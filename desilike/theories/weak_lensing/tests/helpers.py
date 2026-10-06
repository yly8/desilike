"""Analytic cosmology for weak-lensing tests without a Boltzmann solver."""
import numpy as np
from desilike.theories.primordial_cosmology import PrimordialCosmology


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


class ScaleDependentDES(AnalyticDES):
    def __call__(self):
        super().__call__()
        for key, spec in self._requirements.items():
            if key[0] == 'fourier.pk':
                redshift, wave_number = spec['z'], spec['k']
                # Log power is linear in redshift: interpolation has an exact oracle.
                self._results[key] *= (1 + redshift[:, None])**2 * np.exp(
                    -2 * redshift[:, None] * (1 + .03 * np.log(wave_number[None, :])))
