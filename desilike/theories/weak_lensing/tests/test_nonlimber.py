"""Variable-length FFTLog agrees with the independent NumPy/SciPy reference."""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import CubicSpline
from scipy.special import loggamma

from desilike.theories.weak_lensing.nonlimber import (
    _radix2_fft, variable_fft, complex_loggamma, uniform_spline_integral, make_plan, nonlimber_auto,
)


def test_batched_radix2_fft_and_gradient():
    rng = np.random.default_rng(12)
    for length in (1, 16, 1024):
        values = rng.normal(size=(3, length)) + 1j * rng.normal(size=(3, length))
        forward = jax.jit(jax.vmap(_radix2_fft))
        inverse = jax.jit(lambda array: _radix2_fft(array, inverse=True))
        np.testing.assert_allclose(forward(values), np.fft.fft(values), rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(inverse(values), np.fft.ifft(values), rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(inverse(forward(values)), values, rtol=2e-12, atol=2e-12)
    # Parseval gives an independent reverse-mode derivative for complex FFTs.
    real = jnp.asarray(values.real)
    gradient = jax.jit(jax.grad(lambda array: jnp.sum(jnp.abs(_radix2_fft(array))**2)))(real)
    np.testing.assert_allclose(gradient, 2 * length * real, rtol=2e-12, atol=2e-10)


def test_variable_fft_and_spline_integral():
    capacity = 64
    a = np.random.default_rng(2).normal(size=capacity)
    fft = jax.jit(lambda a, n: variable_fft(a, n, capacity))
    inverse = jax.jit(lambda a, n: variable_fft(a, n, capacity, inverse=True))
    for n in (9, 10, 37, 60):
        np.testing.assert_allclose(fft(a, n)[:n], np.fft.fft(a[:n]), atol=1e-11)
        np.testing.assert_allclose(inverse(a, n)[:n], np.fft.ifft(a[:n]), atol=1e-11)
        y = np.stack((a, a**2))
        actual = uniform_spline_integral(jnp.asarray(y), jnp.asarray(n), .13)
        expected = CubicSpline(np.arange(n) * .13, y[:, :n], axis=-1).integrate(0, (n - 1) * .13)
        np.testing.assert_allclose(actual, expected, atol=1e-11)


def test_complex_gamma():
    z = np.linspace(.05, 100, 50)[:, None] + 1j * np.linspace(-100, 100, 53)[None, :]
    np.testing.assert_allclose(np.exp(complex_loggamma(z) - loggamma(z)), 1, atol=2e-12)


def test_nonlimber_adaptive_grid_and_gradient():
    chis = np.geomspace(20., 3000., 60)
    q = np.exp(-.5 * ((chis - 900) / 220)**2) / 500
    growth = 1 / (1 + chis / 4000)
    rates = .5 + chis / 12000
    k = np.geomspace(1e-5, 1e4, 200)
    pk = 1e4 * k**.8 / (1 + (k / .03)**2)**1.3
    ells = np.array([1., 1.75, 8., 40., 150.])
    sigma, bias = .18, 1.5
    plan = make_plan(chis, sigma)
    fn = jax.jit(lambda ch, b: nonlimber_auto(ells, q, ch, growth, rates, k, pk, .67, b, plan))
    sizes = []
    # Frozen from the removed, independently validated NumPy/SciPy implementation.
    # Rows use stretches .91, 1, 1.12; columns follow ells above.
    reference = np.array([
        [1.5512519901014644e-6, 1.6421861263312943e-6, 1.7493508653931276e-6, 3.747887545803582e-7, 3.94990154388333e-8],
        [9.338586738559944e-7, 1.006486617600695e-6, 1.3485032797390822e-6, 4.848502734465933e-7, 5.642537643264185e-8],
        [4.5132951117467125e-7, 4.960843622347507e-7, 8.150928401620938e-7, 5.932430723555761e-7, 8.831997797440035e-8],
    ])
    for stretch, target in zip((.91, 1., 1.12), reference):
        ch = chis * (chis / chis[0])**(stretch - 1)
        sizes.append(int(np.ceil(np.log(ch[-1] / ch[0]) / plan[0])))
        np.testing.assert_allclose(fn(ch, bias), target, rtol=2e-8, atol=1e-12)
    assert len(set(sizes)) == 3  # Different active FFT lengths, one compiled function.
    derivative = jax.jacrev(lambda b: fn(chis, b))(bias)
    fd = (fn(chis, bias + 1e-5) - fn(chis, bias - 1e-5)) / 2e-5
    np.testing.assert_allclose(derivative, fd, rtol=1e-6, atol=1e-12)
    # A capacity overrun must never silently change the active reference grid.
    outside = chis * (chis / chis[0])**4
    assert np.isnan(np.asarray(fn(outside, bias))).all()
