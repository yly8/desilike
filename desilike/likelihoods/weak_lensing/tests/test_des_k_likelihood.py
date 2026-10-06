"""The k-dependent theory works through the standard DES likelihood graph."""
import jax
import numpy as np

from desilike import build
from desilike.likelihoods.weak_lensing import BaseDESY3Likelihood
from desilike.theories.weak_lensing import DESKDependentWeakLensing3x2pt
from desilike.theories.weak_lensing.des_k import DESKDependentLensingKernels
from desilike.theories.weak_lensing.tests.helpers import ScaleDependentDES


def test_k_dependent_likelihood_jit_and_sampler_batch():
    theory = DESKDependentWeakLensing3x2pt(cosmo=ScaleDependentDES(params={}), k=np.geomspace(1e-5, 8e3, 256))
    likelihood = BaseDESY3Likelihood(theory=theory, use_sr=True)
    pipe = build(likelihood)
    assert isinstance(likelihood.theory.kernels, DESKDependentLensingKernels)
    shifts = np.array([.005, .012])
    eager = np.array([float(pipe({'DES_DzL1': shift, 'DES_szL1': 1.12})) for shift in shifts])
    batched = jax.jit(jax.vmap(lambda shift: pipe({'DES_DzL1': shift, 'DES_szL1': 1.12})))(shifts)
    assert np.isfinite(eager).all()
    assert not np.isclose(*eager, rtol=1e-6)
    np.testing.assert_allclose(batched, eager, rtol=1e-10, atol=1e-7)
