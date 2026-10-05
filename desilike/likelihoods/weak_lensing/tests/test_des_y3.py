"""DES Y3 likelihood, data selection, JAX outputs and marginalization regressions."""
import numpy as np
import pytest
import jax

from desilike import build
from desilike.theories.weak_lensing import DESWeakLensing3x2pt
from desilike.theories.weak_lensing.tests.helpers import AnalyticDES
from desilike.likelihoods.weak_lensing import BaseDESY3Likelihood


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


@pytest.mark.parametrize('use_sr', [False, True])
def test_shear_ratio_outputs_survive_jit(use_sr):
    likelihood = BaseDESY3Likelihood(
        theory=DESWeakLensing3x2pt(cosmo=AnalyticDES(params={}), Limber=True), use_sr=use_sr)
    pipe = build(likelihood, output=lambda: likelihood)
    expected = pipe()
    actual = jax.jit(pipe)({})
    np.testing.assert_allclose(actual.logpdf, expected.logpdf, rtol=1e-10)
    np.testing.assert_allclose(actual.theory_ratios, expected.theory_ratios, rtol=1e-10)
    assert actual.theory_ratios.shape == ((9,) if use_sr else (0,))
    residual = actual.flattheory - likelihood.data_vector
    np.testing.assert_allclose(actual.chi2_3x2pt, residual @ actual.precision @ residual, rtol=1e-10)
    np.testing.assert_allclose(-2 * actual.logpdf, actual.chi2_3x2pt + actual.chi2_sr)


def test_missing_required_pair_is_rejected():
    theory = DESWeakLensing3x2pt(cosmo=AnalyticDES(params={}), Limber=True)
    pairs = {name: list(values) for name, values in theory.bin_pairs.items()}
    pairs['xip'].remove((0, 0))
    theory.update(bin_pairs=pairs)
    with pytest.raises(ValueError, match=r'missing.*xip'):
        BaseDESY3Likelihood(theory=theory)


def test_incompatible_theory_bins_are_rejected():
    theory = DESWeakLensing3x2pt(cosmo=AnalyticDES(params={}), Limber=True, nzbins=2)
    with pytest.raises(ValueError, match='bin counts'):
        BaseDESY3Likelihood(theory=theory)


def test_reordered_dataset_types_preserve_likelihood(tmp_path):
    import shutil
    from desilike.theories.weak_lensing.data import resolve_data_dir
    original = BaseDESY3Likelihood(theory=DESWeakLensing3x2pt(cosmo=AnalyticDES(params={}), Limber=True))
    expected = float(build(original)())
    custom = tmp_path / 'reordered'
    shutil.copytree(resolve_data_dir(), custom)
    names = ['xip', 'xim', 'gammat', 'wtheta']
    order = ['gammat', 'xip', 'wtheta', 'xim']
    blocks = {}
    start = 0
    for name in names:
        count = len(np.loadtxt(custom / f'DES_3YR_final_{name}.dat'))
        blocks[name] = np.arange(start, start + count)
        start += count
    permutation = np.concatenate([blocks[name] for name in order])
    covariance = np.loadtxt(custom / 'DES_3YR_final_cov.dat')
    np.savetxt(custom / 'DES_3YR_final_cov.dat', covariance[np.ix_(permutation, permutation)])
    config = custom / 'DES_3YR_final.dataset'
    config.write_text(config.read_text().replace('data_types = xip xim gammat wtheta', 'data_types = ' + ' '.join(order)))
    reordered = BaseDESY3Likelihood(
        theory=DESWeakLensing3x2pt(cosmo=AnalyticDES(params={}), Limber=True), data_dir=custom)
    actual = float(build(reordered)())
    np.testing.assert_allclose(actual, expected, rtol=1e-9)


@pytest.mark.parametrize('cut', ['soft', 'standard'])
@pytest.mark.parametrize('use_sr', [False, True])
def test_required_pairs_follow_scale_cuts(tmp_path, cut, use_sr):
    import re
    import shutil
    from desilike.theories.weak_lensing.data import resolve_data_dir

    custom = tmp_path / cut
    shutil.copytree(resolve_data_dir(), custom)
    config = custom / 'DES_3YR_final.dataset'
    config.write_text(re.sub(r'^data_selection\s*=.*$', f'data_selection = DES_3YR_final_{cut}_cut.dat',
                             config.read_text(), flags=re.MULTILINE))
    theory = DESWeakLensing3x2pt(cosmo=AnalyticDES(params={}), Limber=True)
    likelihood = BaseDESY3Likelihood(theory=theory, data_dir=custom, use_sr=use_sr)
    # This pair has retained angles in soft, but is fully excluded in standard.
    retained_xim = {(i, j) for typ, i, j, _ in likelihood.used_items if likelihood.data_types[typ] == 'xim'}
    assert ((0, 1) in retained_xim) == (cut == 'soft')
    pairs = {name: set() for name in theory.bin_pairs}
    for typ, first, second, _ in likelihood.used_items:
        pairs[likelihood.data_types[typ]].add((first, second))
    if use_sr:
        pairs['gammat'].update((i, j) for i in range(3) for j in range(4))
    # Omitted, fully cut pairs are legal. Only retained data and SR pairs matter.
    theory.update(bin_pairs={name: sorted(values) for name, values in pairs.items()})
    BaseDESY3Likelihood(theory=theory, data_dir=custom, use_sr=use_sr)
    pairs['xip'].remove((0, 0))
    theory.update(bin_pairs={name: sorted(values) for name, values in pairs.items()})
    with pytest.raises(ValueError, match=r'missing.*xip'):
        BaseDESY3Likelihood(theory=theory, data_dir=custom, use_sr=use_sr)
