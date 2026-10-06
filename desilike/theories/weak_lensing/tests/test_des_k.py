"""Independent checks of bin-normalized, scale-dependent growth."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from desilike import build
from desilike.theories.weak_lensing import DESKDependentWeakLensing3x2pt, DESWeakLensing3x2pt
from desilike.theories.weak_lensing.tests.helpers import AnalyticDES, ScaleDependentDES


def test_bin_means_growth_power_and_photoz_gradient():
    theory = DESKDependentWeakLensing3x2pt(cosmo=ScaleDependentDES(params={}), nwbins=2, k=np.geomspace(1e-5, 8e3, 256))
    kernels = theory.kernels
    pipe = build(kernels, output=lambda: (kernels.zmean_lens, kernels.growth_eff, kernels.growth_rate_eff, kernels.pk_linear_mean))
    params = {'DES_DzL1': .012, 'DES_szL1': 1.15}
    means, growth, rates, power = jax.tree_util.tree_map(np.asarray, jax.jit(pipe)(params))
    zs = kernels.z
    shifted = zs - params['DES_DzL1']
    nz = InterpolatedUnivariateSpline(zs, kernels.zbin_w_sp[0])(shifted)
    nz[shifted < 0] = 0
    expected_mean = np.average(zs, weights=nz)
    np.testing.assert_allclose(means[0], expected_mean, atol=1e-13)
    # Width stretching must not redefine the normalization redshift.
    np.testing.assert_allclose(pipe({**params, 'DES_szL1': .85})[0], means, atol=1e-13)
    exponent = 1 + .03 * np.log(.05 / .7)
    np.testing.assert_allclose(growth, np.exp(-(zs[None, :] - means[:, None]) * exponent), rtol=1e-11)
    np.testing.assert_allclose(rates, np.broadcast_to((1 + zs) * exponent, rates.shape), rtol=3e-5)
    wave_number = theory._k
    expected_power = 1e4 * wave_number**-.7 * (1 + wave_number / 20)**-4 * np.exp(
        -2 * means[:, None] * (1 + .03 * np.log(wave_number)))
    np.testing.assert_allclose(power, expected_power, rtol=1e-11)
    objective = lambda shift: jnp.sum(jnp.log(pipe({**params, 'DES_DzL1': shift})[-1][0]))
    shift, step = params['DES_DzL1'], 1e-5
    derivative = jax.jit(jax.grad(objective))(shift)
    finite = (objective(shift + step) - objective(shift - step)) / (2 * step)
    np.testing.assert_allclose(derivative, finite, rtol=2e-6)


def test_separable_limit_and_limber_compatibility():
    for limber in (False, True):
        outputs = []
        for cls in (DESWeakLensing3x2pt, DESKDependentWeakLensing3x2pt):
            theory = cls(cosmo=AnalyticDES(params={}), Limber=limber, nwbins=1, nzbins=1, k=np.geomspace(1e-5, 8e3, 256))
            pipe = build(theory, output=lambda theory=theory: (theory.wtheta, theory.gammat, theory.xip, theory.xim))
            outputs.append(jax.tree_util.tree_map(np.asarray, pipe()))
        for original, normalized in zip(*outputs):
            # For separable P(k,z), the bin normalization cancels algebraically.
            np.testing.assert_allclose(normalized, original, rtol=2e-8, atol=1e-12)
