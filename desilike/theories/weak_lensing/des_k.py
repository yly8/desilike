"""Bin-normalized non-Limber approximation matching the reference des-k.py."""
import jax.numpy as jnp

from .des import DESLensingKernels, DESNonLimber, DESWeakLensing3x2pt, _NonLimberInputs, spline
from .nonlimber import nonlimber_auto


class DESKDependentLensingKernels(DESLensingKernels):
    """Retain linear P(k, z_mean) and normalize growth separately in each bin."""
    _outputs = DESLensingKernels._outputs + ('growth_eff', 'growth_rate_eff', 'pk_linear_mean')

    def __call__(self):
        super().__call__()
        logpower = jnp.log(self.spectrum())
        # Means are evaluated after the photo-z shift, before the width stretch,
        # exactly as in des-k.py. Power leaves use h/Mpc and (Mpc/h)^3.
        self.pk_linear_mean = jnp.exp(spline(self.zmean_lens, self.z_pk, logpower))
        logpower_growth = spline(jnp.log(.05 / self.h), jnp.log(self.k), logpower.T)
        logpower_mean = spline(self.zmean_lens, self.z_pk, logpower_growth)
        self.growth_eff = jnp.exp(.5 * (logpower_growth[None, 1:] - logpower_mean[:, None]))
        loga = -jnp.log1p(jnp.asarray(self.z))
        self.growth_rate_eff = spline(loga, loga[::-1], jnp.log(self.growth_eff[:, ::-1]).T, derivative=1).T
        return self


class _KDependentNonLimberInputs(_NonLimberInputs):
    _outputs = ('h', 'chis', 'growth_eff', 'growth_rate_eff', 'pk_linear_mean', 'qgal')


class DESKDependentNonLimber(DESNonLimber):
    def __init__(self, kernels, biases):
        self.inputs, self.biases = _KDependentNonLimberInputs(kernels), biases

    def __call__(self):
        kernels = self.inputs
        self.cl = jnp.stack([
            nonlimber_auto(self.ells, kernels.qgal[bin_idx], kernels.chis,
                          kernels.growth_eff[bin_idx], kernels.growth_rate_eff[bin_idx],
                          self.k, kernels.pk_linear_mean[bin_idx], kernels.h, bias.value, plan)
            for bin_idx, (bias, plan) in enumerate(zip(self.biases, self.plans))
        ])
        return self


class DESKDependentWeakLensing3x2pt(DESWeakLensing3x2pt):
    """DES NLA theory with the bin-normalized approximation from des-k.py.

    For each lens auto-bin, the non-Limber linear term uses P(k, z_mean)
    and g_eff(z) = sqrt(P(0.05/Mpc, z) / P(0.05/Mpc, z_mean)). RSD uses
    d ln(g_eff) / d ln(a), with no additional k-dependent growth factor.
    This is a separable approximation, not a full unequal-time P(k, z1, z2).
    All constructor settings and likelihood integration follow the base theory.
    Limber=True reduces to the base theory. Cross-bin clustering retains the
    base theory's Limber treatment, as the reference only applies this to auto-bins.
    """
    _kernel_cls = DESKDependentLensingKernels
    _nonlimber_cls = DESKDependentNonLimber
