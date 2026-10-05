"""Native JAX DES FFTLog, preserving the reference's variable integer grids.

Bluestein convolution evaluates a variable-length DFT in a fixed-capacity buffer.
Only the workspace capacity is static; the physical grid and DFT length follow the
NumPy reference at each point. Gradients are piecewise, within each integer grid.
"""
import numpy as np
import jax
import jax.numpy as jnp


def _radix2_fft(values, inverse=False):
    """Batched radix-2 FFT using JAX butterflies on the final axis.

    Bluestein workspaces have power-of-two lengths. Avoid native CPU FFT calls
    here: concurrent ducc0 FFTs can block all XLA Eigen workers while waiting
    for subtasks on the same pool. Elementwise butterflies need no nested FFT
    thread pool and retain JIT, vmap and autodiff support.
    """
    length = values.shape[-1]
    if length < 1 or length & (length - 1):
        raise ValueError('Radix-2 FFT requires a positive power-of-two length')
    stages = length.bit_length() - 1
    index = jnp.arange(length)
    reverse = jnp.zeros_like(index)
    for bit in range(stages):
        reverse = reverse | (((index >> bit) & 1) << (stages - bit - 1))
    values = jnp.asarray(values, dtype=jnp.result_type(values, 1j))[..., reverse]
    sign = 1 if inverse else -1
    roots = jnp.exp(sign * 2j * jnp.pi * index / length).astype(values.dtype)

    def butterfly(stage, current):
        half = jnp.left_shift(1, stage)
        even = index & ~half
        odd = even | half
        phase = roots[(index & (half - 1)) * (length >> (stage + 1))]
        phase = jnp.where((index & half) == 0, phase, -phase)
        return current[..., even] + phase * current[..., odd]

    result = jax.lax.fori_loop(0, stages, butterfly, values)
    return result / length if inverse else result


def variable_fft(values, size, capacity, inverse=False):
    """Unnormalised DFT (or normalised inverse) of the first ``size`` values."""
    length = 1 << (2 * capacity - 1).bit_length()
    index = jnp.arange(capacity)
    sign = 1 if inverse else -1
    chirp = jnp.exp(sign * 1j * jnp.pi * index**2 / size)
    a = jnp.pad(jnp.where(index < size, values * chirp, 0), (0, length - capacity))
    lag = jnp.arange(length)
    lag = jnp.where(lag < capacity, lag, lag - length)
    b = jnp.where(jnp.abs(lag) < size, jnp.exp(-sign * 1j * jnp.pi * lag**2 / size), 0)
    transformed = _radix2_fft(jnp.stack((a, b)))
    result = _radix2_fft(transformed[0] * transformed[1], inverse=True)[:capacity] * chirp
    return result / size if inverse else result


def complex_loggamma(z):
    """Lanczos log Gamma for Re(z)>0 (the domain used by DES's Mellin kernels)."""
    # Shift small real parts away from the first pole; all intermediate branches
    # stay finite, including for reverse-mode differentiation.
    shift = jnp.real(z) < .5
    w = z + shift - 1
    coefficients = (676.5203681218851, -1259.1392167224028, 771.32342877765313,
                    -176.61502916214059, 12.507343278686905, -.13857109526572012,
                    9.9843695780195716e-6, 1.5056327351493116e-7)
    series = jnp.full_like(w, .99999999999980993)
    for i, coefficient in enumerate(coefficients):
        series = series + coefficient / (w + i + 1)
    t = w + 7.5
    result = .5 * jnp.log(2 * jnp.pi) + (w + .5) * jnp.log(t) - t + jnp.log(series)
    return result - jnp.where(shift, jnp.log(z), 0)


def gamma_ratio(mu, q):
    plus, minus = (mu + 1 + q) / 2, (mu + 1 - q) / 2
    regular = jnp.exp(complex_loggamma(plus) - complex_loggamma(minus))
    asymptotic = jnp.exp((plus - .5) * jnp.log(plus) - (minus - .5) * jnp.log(minus) - q
                        + (1 / plus - 1 / minus) / 12
                        + (1 / minus**3 - 1 / plus**3) / 360
                        + (1 / plus**5 - 1 / minus**5) / 1260)
    return jnp.where(jnp.abs(jnp.imag(q)) + jnp.abs(mu) > 200, asymptotic, regular)


def log_extrapolate(values, index, size):
    """Reference log extrapolation with safe inactive branches."""
    def extend(a, b, distance):
        positive = (a > 1e-30) & (b > 1e-30)
        safe_a, safe_b = jnp.where(positive, a, 1.), jnp.where(positive, b, 1.)
        return jnp.where(positive, safe_a * jnp.exp(jnp.log(safe_b / safe_a) * distance), 0.)
    low = extend(values[0], values[1], jnp.minimum(index, 0))
    high = extend(values[size - 1], values[size - 2], -jnp.maximum(index - size + 1, 0))
    return jnp.where(index < 0, low, jnp.where(index >= size, high, values[jnp.clip(index, 0, size - 1)]))


def uniform_spline_integral(values, size, spacing):
    """Integral of the first size knots' not-a-knot cubic on a uniform grid."""
    capacity = values.shape[-1]
    index = jnp.arange(capacity)
    slopes = jnp.diff(values, axis=-1)
    previous = slopes[..., jnp.clip(index - 1, 0, capacity - 2)]
    following = slopes[..., jnp.clip(index, 0, capacity - 2)]
    rhs = 3 * (previous + following)
    rhs = rhs.at[..., 0].set((5 * slopes[..., 0] + slopes[..., 1]) / 2)
    rhs = rhs.at[..., size - 1].set((5 * slopes[..., size - 2] + slopes[..., size - 3]) / 2)
    rhs = jnp.where(index < size, rhs, 0)
    endpoint = (index == 0) | (index == size - 1)
    diag = jnp.where((index < size) & ~endpoint, 4., 1.)
    lower = jnp.where((index > 0) & (index < size), jnp.where(index == size - 1, 2., 1.), 0.)
    upper = jnp.where(index < size - 1, jnp.where(index == 0, 2., 1.), 0.)
    def matvec(v):
        return (diag[:, None] * v + lower[:, None] * jnp.concatenate((jnp.zeros_like(v[:1]), v[:-1]))
                + upper[:, None] * jnp.concatenate((v[1:], jnp.zeros_like(v[:1]))))
    def solve(_, b):
        return jax.lax.linalg.tridiagonal_solve(lower, diag, upper, b)
    def transpose_solve(_, b):
        return jax.lax.linalg.tridiagonal_solve(jnp.concatenate((jnp.zeros(1), upper[:-1])), diag,
                                               jnp.concatenate((lower[1:], jnp.zeros(1))), b)
    derivatives = jax.lax.custom_linear_solve(matvec, rhs.T, solve, transpose_solve).T
    trapezoids = jnp.sum(jnp.where(jnp.arange(capacity - 1) < size - 1,
                                   (values[..., :-1] + values[..., 1:]) / 2, 0), axis=-1)
    return spacing * (trapezoids - (derivatives[..., size - 1] - derivatives[..., 0]) / 12)


def make_plan(coordinate_grid, sigma):
    """Reserve headroom from the input redshift grid, without fixing the active grid.

    Also accepts a distance grid for standalone use. The runtime capacity guard
    rejects geometries exceeding this workspace instead of clipping their grids.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('JAX non-Limber requires a finite positive lens-redshift width')
    step = float(sigma / 3.5)
    nchi = int(np.ceil(np.log(coordinate_grid[-1] / coordinate_grid[0]) / step))
    capacity = max(2 * nchi, nchi + 64, 8)
    extrap = int(np.ceil(1 / step))
    return step, capacity, extrap


def nonlimber_auto(ells, qgal, chis, growth, rates, k, pk, h, bias, plan):
    """DES auto-bin non-Limber C_ell, including RSD, entirely in JAX."""
    from .des import spline
    step, capacity, pad = plan  # reference uses identical extrapolation/padding widths
    span = jnp.log(chis[-1]) - jnp.log(chis[0])
    raw_size = jnp.ceil(span / step).astype(jnp.int32)
    valid = (raw_size >= 4) & (raw_size <= capacity)
    size = jnp.clip(raw_size, 4, capacity)
    spacing = span / (size - 1)
    index = jnp.arange(capacity)
    # Inactive coordinates are clamped, so extrapolated tails cannot overflow.
    chi = jnp.exp(jnp.log(chis[0]) + jnp.minimum(index, size - 1) * spacing)
    weight = jnp.nan_to_num(spline(chi, chis, qgal * growth * chis), nan=0.)
    rate = spline(chi, chis, rates)
    rsd = weight * rate / jnp.where(bias != 0, bias, 1.)
    fft_capacity = capacity + 4 * pad
    fft_size = size + 4 * pad
    fft_size = fft_size - fft_size % 2
    j = jnp.arange(fft_capacity)
    x = jnp.exp(jnp.log(chis[0]) + (j - 2 * pad) * spacing)
    mode = jnp.arange(fft_capacity // 2 + 1)
    eta = 2 * jnp.pi * mode / (fft_size * spacing)
    window_count = jnp.floor(.25 * fft_size / 2)
    theta = (fft_size // 2 - mode) / (window_count - 1)
    window = jnp.where(mode > fft_size // 2 - window_count,
                       theta - jnp.sin(2 * jnp.pi * theta) / (2 * jnp.pi), 1.)
    window = jnp.where(mode <= fft_size // 2, window, 0.)
    ell = jnp.asarray(ells)[:, None]
    # Compute only the physical output grid, padded to capacity for static shapes.
    offset = 2 * pad - size % 2
    output_index = offset + jnp.minimum(index, size - 1)
    y = (ell + 1) / jnp.exp(jnp.log(chis[0]) + (fft_size - 1 - output_index - 2 * pad) * spacing)
    y0 = (ell + 1) / x[fft_size - 1]
    phase = jnp.exp(-1j * eta[None, :] * jnp.log(x[0] * y0))
    def transform(w, nu):
        extended = log_extrapolate(w, j - 2 * pad, size)
        extended = jnp.where((j >= pad) & (j < size + 3 * pad) & (j < fft_size), extended, 0.)
        coeff = variable_fft(extended * x**(-nu), fft_size, fft_capacity)[:len(mode)] * window
        z = nu + 1j * eta[None, :]
        if nu == 1.:
            mellin = 2**z * gamma_ratio(ell + .5, z - 1.5)
        else:
            mellin = 2**(z - 2) * (z - 1) * (z - 2) * gamma_ratio(ell + .5, z - 3.5)
        half = jnp.conj(coeff[None, :] * phase * mellin)
        mirror = jnp.clip(fft_size - j, 0, len(mode) - 1)
        full = jnp.where(j <= fft_size // 2, half[:, jnp.minimum(j, len(mode) - 1)], jnp.conj(half[:, mirror]))
        # np.irfft discards imaginary parts of DC and the Nyquist component.
        full = jnp.where((j == 0) | (j == fft_size // 2), full.real, full)
        result = jax.vmap(lambda v: variable_fft(v, fft_size, fft_capacity, inverse=True))(full).real
        return result[:, output_index] * y**(-nu) * jnp.sqrt(jnp.pi) / 4
    integral = transform(weight, 1.) - transform(rsd, 1.1)
    power = jnp.exp(spline(jnp.log(y / h), jnp.log(k), jnp.log(pk))) / h**3
    result = 2 / jnp.pi * uniform_spline_integral(y**3 * power * integral**2, size, spacing)
    return jnp.where(valid, result, jnp.nan)
