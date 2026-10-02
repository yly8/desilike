"""Short integration checks; these chains do not establish convergence.
Run from any directory: python /absolute/path/to/this/script.py
Uses the installed cosmoprimo and lsstypes, not temporary source copies.
"""
import os
from pathlib import Path
import sys
import json
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '4'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cosmoprimo
import lsstypes
import numpy as np
import jax
from desilike import build, get_params, Posterior, samplers
from desilike.base import replace
from desilike.emulators import Emulator, Space
from desilike.samples import MCSamples
from desilike.theories.weak_lensing import DESWeakLensing3x2pt
from desilike.likelihoods.weak_lensing import BaseDESY3Likelihood

jax.config.update('jax_enable_x64', True)
OUT = Path(__file__).resolve().parent
report = {'configuration': {'Weyl': True, 'Limber': False, 'use_sr': True},
          'purpose': 'Integration smoke test, not convergence or speed comparison',
          'python': sys.executable, 'cosmoprimo': cosmoprimo.__file__,
          'lsstypes': lsstypes.__file__, 'jax': jax.__version__, 'cases': {}}
print(json.dumps(report, indent=2), flush=True)


def configure(like, names):
    params = get_params(like)
    for p in params.select(derived=False):
        if hasattr(p, 'fixed'):
            p.update(fixed=p.name not in names)
    # Narrow ranges keep this interface test near the fiducial point.
    if 'h' in names:
        params['h'].update(prior={'limits': [0.6731, 0.6741]},
                           ref={'dist': 'norm', 'loc': 0.6736, 'scale': 0.0001})
    params['DES_m1'].update(prior={'limits': [-0.02, 0.02]},
                            ref={'dist': 'norm', 'loc': -0.006, 'scale': 0.002})


def run_case(label, like, names, steps):
    start = time.perf_counter()
    posterior = build(Posterior(like))
    sampler = samplers.Sampler(posterior, kernel=samplers.Emcee(nwalkers=8),
                               rng=42, batch_size=2)
    assert set(p.name for p in sampler.varied_params) == set(names)
    chain = sampler.run(min_steps=steps, max_steps=steps, check_every=steps,
                        gelman_rubin=None, burn_in=0)
    sampling_seconds = time.perf_counter() - start
    vals = {name: np.asarray(chain[name].value) for name in names}
    logs = {name: np.asarray(chain[name].value) for name in
            ('logposterior', 'loglikelihood', 'logprior')}
    for array in list(vals.values()) + list(logs.values()):
        assert np.isfinite(array).all()
    for array in vals.values():
        assert np.ptp(array) > 0
        assert np.any(np.diff(array, axis=0) != 0), 'walkers never moved'
    np.testing.assert_allclose(logs['logposterior'], logs['loglikelihood'] + logs['logprior'], atol=1e-8)
    # Re-evaluate a stored point through the compiled graph, independently of sampler blobs.
    point = {name: float(array.ravel()[-1]) for name, array in vals.items()}
    value, derived = posterior(point, return_derived=True)
    np.testing.assert_allclose(float(value), logs['logposterior'].ravel()[-1], atol=1e-7)
    np.testing.assert_allclose(float(derived['loglikelihood']), logs['loglikelihood'].ravel()[-1], atol=1e-7)
    filename = OUT / f'{label}.h5'
    chain.write(filename)
    loaded = MCSamples.read(filename)
    np.testing.assert_array_equal(loaded['logposterior'].value, logs['logposterior'])
    result = {'parameters': names, 'steps': steps, 'walkers': 8,
              'stored_shape': list(logs['logposterior'].shape),
              'seconds_including_setup_and_compilation': sampling_seconds,
              'loglikelihood_range': [float(logs['loglikelihood'].min()), float(logs['loglikelihood'].max())],
              'finite': True, 'walkers_moved': True, 'posterior_recheck': True,
              'save_reload': True, 'chain': str(filename)}
    print(label, json.dumps(result), flush=True)
    report['cases'][label] = result
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2))


# Exact likelihood: changing h forces CAMB updates, DES_m1 exercises nuisance response.
like = BaseDESY3Likelihood(theory=DESWeakLensing3x2pt(Weyl=True, Limber=False), use_sr=True)
configure(like, ['h', 'DES_m1'])
run_case('exact', like, ['h', 'DES_m1'], steps=6)

# Emulator integration: only m1 is emulated; all other parameters remain fixed.
theory = DESWeakLensing3x2pt(Weyl=True, Limber=False)
like = BaseDESY3Likelihood(theory=theory, use_sr=True)
configure(like, ['DES_m1'])
truth = build(like)
point = {'DES_m1': 0.003}
reference = float(truth(point))
emu = Emulator(theory, Space(bounds={'DES_m1': (-0.02, 0.02)})).train(budget=1)
replace(like, theory, emu.to_calculator())
prediction = float(build(like)(point))
np.testing.assert_allclose(prediction, reference, atol=1e-7, rtol=1e-9)
report['emulator_delta_chi2'] = -2 * (prediction - reference)
run_case('emulated', like, ['DES_m1'], steps=12)
print('ALL CHECKS PASSED', flush=True)
