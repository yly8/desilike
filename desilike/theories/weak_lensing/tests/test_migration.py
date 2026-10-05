"""Regressions for the DES migration to refactor-jax."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from desilike import build
from desilike.theories.primordial_cosmology import PrimordialCosmology
from desilike.theories.weak_lensing import DESWeakLensing3x2pt
from desilike.theories.weak_lensing.tests.helpers import AnalyticDES


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


def test_data_directory_expands_user_and_reports_missing(tmp_path):
    from desilike.theories.weak_lensing.data import resolve_data_dir
    from pathlib import Path
    assert resolve_data_dir('~') == Path.home().resolve()
    with pytest.raises(FileNotFoundError, match='DES data directory does not exist'):
        resolve_data_dir(tmp_path / 'missing')


def test_legendre_values_and_derivatives_across_scipy_versions(monkeypatch):
    from scipy import special
    from numpy.polynomial.legendre import Legendre
    from desilike.theories.weak_lensing.base import _legendre_values_and_derivatives

    degree = 12
    coordinate = np.array([-1., -.3, .7, .999, 1.])
    expected = np.array([
        [Legendre.basis(order)(coordinate) for order in range(degree + 1)],
        [Legendre.basis(order).deriv()(coordinate) for order in range(degree + 1)],
    ])
    if hasattr(special, 'legendre_p_all'):
        # Exercise the modern path even if the deprecated name also exists.
        monkeypatch.delattr(special, 'lpn', raising=False)
    actual = np.asarray(_legendre_values_and_derivatives(degree, coordinate))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('pair', [(-1, 0), (4, 0), (0, 1.5)])
def test_invalid_bin_pairs_are_rejected(pair):
    pairs = {'xip': [pair], 'xim': [], 'gammat': [], 'wtheta': []}
    with pytest.raises(ValueError, match='Invalid xip bin pair'):
        DESWeakLensing3x2pt(cosmo=AnalyticDES(params={}), bin_pairs=pairs)


@pytest.mark.parametrize('options', [{'nzbins': 5}, {'nwbins': 0}, {'nzbins': 1.5}])
def test_invalid_bin_counts_are_rejected(options):
    with pytest.raises(ValueError, match='must be an integer'):
        DESWeakLensing3x2pt(cosmo=AnalyticDES(params={}), **options)


@pytest.mark.parametrize('k', [[0., .1, 1., 10.], [.01, 1., .1, 10.], [.01, .1, 1., np.nan], [.1, 1.]])
def test_invalid_power_spectrum_grid_is_rejected(k):
    with pytest.raises(ValueError, match='k must contain'):
        DESWeakLensing3x2pt(cosmo=AnalyticDES(params={}), k=k)
