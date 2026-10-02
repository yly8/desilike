"""Regressions for the DES migration to refactor-jax."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from desilike import build
from desilike.theories.primordial_cosmology import PrimordialCosmology
from desilike.theories.weak_lensing import DESWeakLensing3x2pt


def test_cubic_matches_reference_scipy():
    from scipy.interpolate import CubicSpline
    from desilike.theories.weak_lensing.des import spline
    x = np.array([0., .2, .7, 1., 2.])
    y = np.stack([np.sin(x), np.cos(x)], axis=-1)
    query = np.linspace(-.1, 2.1, 31)
    np.testing.assert_allclose(spline(query, x, y), CubicSpline(x, y, axis=0)(query), atol=1e-14)
    np.testing.assert_allclose(jax.jit(lambda values: spline(query, x, values))(y),
                               CubicSpline(x, y, axis=0)(query), atol=1e-14)
    # Values and moving knots both appear in photo-z / stretch derivatives.
    def objective(shift):
        return jnp.sum(spline(query, jnp.asarray(x) * (1 + shift), jnp.asarray(y) * (1 - shift)))
    step = 1e-5
    finite = (objective(step) - objective(-step)) / (2 * step)
    np.testing.assert_allclose(jax.grad(objective)(0.), finite, rtol=1e-7)



def test_requirements_select_redshift_and_k_axes():
    cosmo = PrimordialCosmology(params={})
    z, k = np.array([0., .5, 1.]), np.array([.01, .1, 1., 10.])
    cosmo.add_requirements({'fourier.pk': [dict(of='delta_m', z=z, k=k)]})
    key = next(iter(cosmo._requirements))
    cosmo._results[key] = jnp.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(cosmo.get('fourier.pk', of='delta_m', z=z[[0, 2]], k=k[[1, 3]]), [[1, 3], [9, 11]])
    np.testing.assert_array_equal(cosmo.get('fourier.pk', of='delta_m', z=.5, k=k[[1, 3]]), [5, 7])


class AnalyticDES(PrimordialCosmology):
    """Cheap, positive spectra to exercise the full DES kernel and graph."""
    def __call__(self):
        for key, spec in self._requirements.items():
            method = key[0]
            z = spec.get('z', np.array([0.]))
            if method == 'fourier.pk':
                k = spec['k']
                boost = 1. + k if spec['static']['non_linear'] else np.ones_like(k)
                self._results[key] = 1e4 * k[None, :] ** -.7 * (1. + k[None, :] / 20.)**-4 * boost / (1. + z[:, None]) ** 2
            elif method == 'background.comoving_radial_distance':
                self._results[key] = 2000. * z
            elif method == 'background.efunc':
                self._results[key] = np.sqrt(.3 * (1. + z)**3 + .7)
            else:
                self._results[key] = {'h': .7, 'Omega_m': .3, 'Omega_b': .05, 'Omega_cdm': .2487, 'Omega_ncdm_tot': .0013, 'omega_b': .022, 'omega_cdm': .12, 'm_ncdm_tot': .06}[method.split('.')[1]]


@pytest.mark.parametrize('fourier', ['legendre', 'binned_bessels'])
def test_des_kernel_and_jit_parameter_response(fourier):
    cosmo = AnalyticDES(params={})
    theory = DESWeakLensing3x2pt(cosmo=cosmo, ia_model='NLA', Limber=True, fourier=fourier)
    pipe = build(theory, output=lambda: theory.xip)
    reference = np.asarray(pipe())
    assert reference.shape == (4, 4, 20)
    assert np.isfinite(reference).all()
    assert np.any(reference != 0.)
    base_m = float(pipe.params['DES_m1'].value)
    shifted = np.asarray(jax.jit(pipe)({'DES_m1': base_m + .02}))
    np.testing.assert_allclose(shifted[0, 0], reference[0, 0] * ((1 + base_m + .02) / (1 + base_m))**2, rtol=1e-9)
    np.testing.assert_allclose(shifted[1, 1], reference[1, 1], rtol=1e-12)
    gradient = jax.grad(lambda m: jnp.sum(pipe({'DES_m1': m})[0, 0]))(base_m)
    np.testing.assert_allclose(gradient, 2 * reference[0, 0].sum() / (1 + base_m), rtol=1e-6)


def test_point_mass_precision_and_shear_ratios():
    from desilike.likelihoods.weak_lensing import BaseDESY3Likelihood
    theory = DESWeakLensing3x2pt(cosmo=AnalyticDES(params={}), ia_model='NLA', Limber=True)
    likelihood = BaseDESY3Likelihood(theory=theory, use_sr=True)
    pipe = build(likelihood)
    value = float(pipe())
    assert np.isfinite(value)
    assert likelihood.flattheory.shape == likelihood.data_vector.shape
    # Marginalization must actually remove information in the gammat block.
    correction = likelihood.covinv_orig - np.asarray(likelihood.precision)
    indices = np.array([i for i, item in enumerate(likelihood.used_items) if item[0] == 2])
    assert np.linalg.norm(correction[np.ix_(indices, indices)]) > 0.
    assert np.linalg.norm(correction - correction.T) / np.linalg.norm(correction) < 1e-9
    assert np.isfinite(float(jax.jit(pipe)({'DES_m1': .01})))


def test_camb_nla_nonlimber_integration():
    pytest.importorskip('camb')
    from desilike.likelihoods.weak_lensing import BaseDESY3Likelihood
    likelihood = BaseDESY3Likelihood(use_sr=True)
    pipe = build(likelihood)
    reference = float(pipe())
    shifted = float(jax.jit(pipe)({'omega_cdm': .125}))
    assert np.isfinite(reference) and np.isfinite(shifted)
    assert not np.isclose(reference, shifted)


def test_custom_data_directory_is_shared(tmp_path, monkeypatch):
    import shutil
    from desilike.theories.weak_lensing.data import resolve_data_dir
    from desilike.likelihoods.weak_lensing import BaseDESY3Likelihood
    custom = tmp_path / 'custom data'
    shutil.copytree(resolve_data_dir(), custom)
    # Different measurements make a silent fallback to bundled data detectable.
    path = custom / 'DES_3YR_final_xip.dat'
    measurements = np.loadtxt(path)
    measurements[:, 3] *= 2
    np.savetxt(path, measurements)
    monkeypatch.chdir(tmp_path)
    theory = DESWeakLensing3x2pt(cosmo=AnalyticDES(params={}), data_dir='custom data', Limber=True)
    likelihood = BaseDESY3Likelihood(theory=theory, use_sr=True)
    assert likelihood.data_dir == theory.data_dir == str(custom.resolve())
    np.testing.assert_allclose(likelihood.data_arrays[0][0, 0], measurements[:20, 3])
    # An explicit likelihood directory updates a supplied theory as well.
    other_theory = DESWeakLensing3x2pt(cosmo=AnalyticDES(params={}), Limber=True)
    other = BaseDESY3Likelihood(theory=other_theory, data_dir=custom, use_sr=True)
    pipe = build(other)
    assert other.theory.data_dir == str(custom.resolve())
    assert np.isfinite(float(pipe()))
    np.testing.assert_allclose(other.data_vector, likelihood.data_vector)


def test_data_directory_expands_user_and_reports_missing(tmp_path):
    from desilike.theories.weak_lensing.data import resolve_data_dir
    from pathlib import Path
    assert resolve_data_dir('~') == Path.home().resolve()
    with pytest.raises(FileNotFoundError, match='DES data directory does not exist'):
        resolve_data_dir(tmp_path / 'missing')
