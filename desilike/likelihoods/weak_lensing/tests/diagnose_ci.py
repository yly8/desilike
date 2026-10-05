"""Manual Linux CI probe for the CAMB/JAX hang; not collected as a pytest test."""
import faulthandler
import os
import subprocess
import threading
import time
from importlib.metadata import version

import numpy as np
import pytest
import jax

from desilike import build
from desilike.likelihoods.weak_lensing import BaseDESY3Likelihood
from desilike.theories.primordial_cosmology import CosmoprimoCosmology


START = time.monotonic()


def report(message):
    print(f'{time.monotonic() - START:.3f}s [{threading.current_thread().name}] {message}', flush=True)


def integration_probe():
    original = CosmoprimoCosmology.__call__

    def logged_call(self):
        report('CAMB callback ENTER')
        result = original(self)
        report('CAMB callback EXIT')
        return result

    CosmoprimoCosmology.__call__ = logged_call
    faulthandler.dump_traceback_later(45, repeat=True)
    try:
        report('BUILD')
        likelihood = BaseDESY3Likelihood(use_sr=True)
        pipe = build(likelihood)
        report('EAGER')
        reference = float(pipe())
        report(f'EAGER complete: {reference}')
        parameters = {'omega_cdm': .125}
        report('LOWER')
        lowered = jax.jit(pipe).lower(parameters)
        report('COMPILE')
        compiled = lowered.compile()
        report('EXECUTE')
        def native_stacks():
            report('NATIVE STACKS: execution exceeded 60 seconds')
            subprocess.run(['sudo', 'gdb', '-batch', '-ex', 'set pagination off',
                            '-ex', 'thread apply all bt 12', '-p', str(os.getpid())],
                           timeout=40, check=False)
        watchdog = threading.Timer(60, native_stacks)
        watchdog.daemon = True
        watchdog.start()
        try:
            result = compiled(parameters)
        finally:
            watchdog.cancel()
        report('DISPATCH returned; synchronize')
        shifted = float(result)
        report(f'COMPLETE: {shifted}')
        assert np.isfinite(reference) and np.isfinite(shifted)
        assert not np.isclose(reference, shifted)
    finally:
        faulthandler.cancel_dump_traceback_later()
        CosmoprimoCosmology.__call__ = original


class ProbePlugin:
    def pytest_collection_modifyitems(self, items):
        for item in items:
            if item.name == 'test_camb_nla_nonlimber_integration':
                item.obj = integration_probe


if __name__ == '__main__':
    for name in ['jax', 'jaxlib', 'camb', 'numpy', 'scipy', 'coverage', 'pytest']:
        report(f'{name}={version(name)}')
    report(f"mode={os.environ['WL_PROBE_MODE']}")
    paths = []
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        report(f'{name}={os.environ.get(name, "unset")}')
    if not os.environ['WL_PROBE_MODE'].startswith('isolated'):
        paths = ['desilike/emulators/tests/test_api.py', 'desilike/likelihoods/tests/test_bao.py',
                 'desilike/likelihoods/tests/test_bbn.py', 'desilike/likelihoods/tests/test_cmb.py',
                 'desilike/likelihoods/tests/test_supernovae.py']
    paths.append('desilike/likelihoods/weak_lensing/tests/test_des_y3.py')
    raise SystemExit(pytest.main(['-v', '-s', '--cov=desilike', '--cov-report=', '--timeout=300',
                                 '--timeout-method=thread', *paths], plugins=[ProbePlugin()]))
