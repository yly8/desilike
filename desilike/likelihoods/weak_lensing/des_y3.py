import os

import numpy as np
import jax.numpy as jnp
from getdist import IniFile
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline
try:
    from scipy.integrate import simpson
except ImportError:
    from scipy.integrate import simps as simpson
import astropy.units as u
import astropy.constants as const
import pickle
import copy

from desilike.base import GaussianLikelihood
from desilike.parameter import Variable
from desilike.theories.weak_lensing.data import resolve_data_dir

trapz = getattr(np, "trapezoid", None) or np.trapz
from desilike.theories.weak_lensing import DESWeakLensing3x2pt


class BaseDESY3Likelihood(GaussianLikelihood):

    _is_external = False

    def __init__(self, use_sr=False, theory=None, cosmo=None, data_dir=None):
        # A supplied theory owns its data location unless explicitly overridden here.
        inherited = getattr(theory, 'data_dir', None) if data_dir is None else data_dir
        self.data_dir = str(resolve_data_dir(inherited))
        if theory is None:
            theory = DESWeakLensing3x2pt(cosmo=cosmo, data_dir=self.data_dir)
        else:
            updates = {'data_dir': self.data_dir}
            if cosmo is not None:
                updates['cosmo'] = cosmo
            theory.update(**updates)
        self.theory = theory
        self.use_sr = use_sr
        self.load_data(self.data_dir)
        self.data_vector = self.make_vector(self.data_arrays)
        self.flatdata = Variable(f'{type(self).__name__}.flatdata', value=self.data_vector)

    def load_data(self, data_dir):
        ini = IniFile(os.path.join(data_dir, 'DES_3YR_final.dataset'))
        self.indices = []
        self.used_indices = []
        self.used_items = []
        self.fullcov = np.loadtxt(ini.relativeFileName('cov_file'))
        ntheta = ini.int('num_theta_bins')
        theta = np.loadtxt(ini.relativeFileName('theta_bins_file'))
        self.theta_edges = theta[:,0]
        self.theta_edges = np.append(self.theta_edges, theta[-1,1])
        self.theta_bins = theta[:,2]
        self.intrinsic_alignment_model = ini.string('intrinsic_alignment_model')
        self.data_types = ini.string('data_types').split()
        self.used_types = ini.list('used_data_types', self.data_types)
        with open(ini.relativeFileName('data_selection'), encoding="utf-8") as f:
            header = f.readline()
            assert (header.strip() == '#  type bin1 bin2 theta_min theta_max')
            lines = f.readlines()
        ranges = {}
        for tp in self.data_types:
            ranges[tp] = np.empty((6, 6), dtype=object)
        for line in lines:
            items = line.split()
            if items[0] in self.used_types:
                bin1, bin2 = [int(x) - 1 for x in items[1:3]]
                ranges[items[0]][bin1][bin2] = [np.float64(x) for x in items[3:]]
        self.ranges = ranges
        self.nzbins = ini.int('num_z_bins')  # for lensing sources
        self.nwbins = ini.int('num_gal_bins', 0)  # for galaxies
        maxbin = max(self.nzbins, self.nwbins)
        cov_ix = 0
        self.bin_pairs = []
        self.data_arrays = []
        self.thetas = []
        for i, tp in enumerate(self.data_types):
            xi = np.loadtxt(ini.relativeFileName(f'measurements[{tp}]'))
            bin1 = xi[:, 0].astype(int) - 1
            bin2 = xi[:, 1].astype(int) - 1
            tbin = xi[:, 2].astype(int) - 1
            corr = np.empty((maxbin, maxbin), dtype=object)
            corr[:, :] = None
            self.data_arrays.append(corr)
            self.bin_pairs.append([])
            for f1, f2, ix, dat in zip(bin1, bin2, tbin, xi[:, 3]):
                self.indices.append((i, f1, f2, ix))
                if not (f1, f2) in self.bin_pairs[i]:
                    self.bin_pairs[i].append((f1, f2))
                    corr[f1, f2] = np.zeros(ntheta)
                corr[f1, f2][ix] = dat
                if ranges[tp][f1, f2] is not None:
                    mn, mx = ranges[tp][f1, f2]
                    if mn < self.theta_bins[ix] < mx:
                        self.thetas.append(self.theta_bins[ix])
                        self.used_indices.append(cov_ix)
                        self.used_items.append(self.indices[-1])
                cov_ix += 1
        nz_source = np.loadtxt(ini.relativeFileName('nz_file'))
        self.zmid = nz_source[:, 1]
        nz_lens = np.loadtxt(ini.relativeFileName('nz_gal_file'))
        assert (np.array_equal(nz_lens[:, 1], self.zmid))
        self.std_z = []
        for b in range(self.nwbins):
            mean_z = trapz(self.zmid * nz_lens[:, b + 3], self.zmid) / trapz(nz_lens[:, b + 3], self.zmid)
            variance_z = trapz(nz_lens[:, b + 3] * (self.zmid - mean_z)**2, self.zmid) / trapz(nz_lens[:, b + 3], self.zmid)
            self.std_z.append(np.sqrt(variance_z))
        self.zmax = self.zmid[-1]

        #shear-ratio data
        if self.use_sr:
            sr_file = ini.relativeFileName('sr_file')
            with open(sr_file, "rb") as f:
                ratio_data = pickle.load(f)
            self.sr_nbin_source = ratio_data['nbin_source']  # 4, we use all the source bins
            self.sr_nbin_lens = ratio_data['nbin_lens'] # 3, because we don't use all the lens bins
            self.sr_nratios_per_lens = ratio_data['nratios_per_lens'] # 3, because there are 3 independent ratios we can construct given 4 source bins, per each lens bin.

            self.sr_data = ratio_data['measured_ratios']
            sr_cov = ratio_data['ratio_cov']
            self.sr_covinv = np.linalg.inv(sr_cov)
            sr_theta = ratio_data['theta_data']
            sr_theta_max = [25.4, 18.26, 13.03, 10.87, 9.66, 9.04]
            sr_theta_min = [[8.47, 6.07, 4.34, 2.5, 2.5, 2.5], [8.47, 6.07, 4.34, 2.5, 2.5, 2.5], [2.5, 2.5, 4.34, 2.5, 2.5, 2.5]]
            ind_cov_data = ratio_data['inv_cov_individual_ratios']

            # Generate scale masks for each bin pair.
            # These are used in the calculation of each ratio from the range of points
            self.sr_masks = {}
            for sc in range(self.sr_nratios_per_lens):
                for l in range(self.sr_nbin_lens):
                    t_min = sr_theta_min[sc][l]
                    t_max = sr_theta_max[l]
                    self.sr_masks[(sc, l)] = (sr_theta > t_min) & (sr_theta <= t_max)

            self.inv_cov_individual_ratios = {}
            sr_n = 0
            for l in range(self.sr_nbin_lens):
                for sc in range(self.sr_nratios_per_lens):
                    mask = self.sr_masks[sc, l]
                    srP = ind_cov_data[sr_n][mask][:, mask]
                    self.inv_cov_individual_ratios[sc, l] = srP
                    sr_n += 1

    def __post_init__(self, *args, **kwargs):
        self.covmat = self.fullcov[np.ix_(self.used_indices, self.used_indices)]
        self.covinv_orig = np.linalg.inv(self.covmat)
        self.errors = copy.deepcopy(self.data_arrays)
        for cov_ix, (type_ix, f1, f2, ix) in enumerate(self.indices):
            self.errors[type_ix][f1, f2][ix] = np.sqrt(self.fullcov[cov_ix, cov_ix])
        self.theta_bins_radians = self.theta_bins / 60 * np.pi / 180
        self.theta_edges_radians = self.theta_edges / 60 * np.pi / 180

        self.zs = self.zmid[self.zmid <= self.zmax]

        #add point_mass to \gamma_t
        sigcrit_inv_fac = (4 * np.pi * const.G)/(const.c**2)
        self.sigcrit_inv_fac_Mpc_Msun = (sigcrit_inv_fac.to(u.Mpc/u.M_sun)).value

        def get_Dcom_array(zarray, Omega_m, H0, Omega_L=None):
            if Omega_L is None:
                Omega_L = 1. - Omega_m
            c = 299792.458
            Dcom_array = np.zeros(len(zarray))
            for j in range(len(zarray)):
                zf = zarray[j]
                res1 = scipy.integrate.quad(lambda z: (c / H0) * (1 / (np.sqrt(Omega_L + Omega_m * ((1 + z) ** 3)))), 0, zf)
                Dcom = res1[0]
                Dcom_array[j] = Dcom
            return Dcom_array

        z_distance = np.linspace(0.0,6.2,1000)
        chi_distance = get_Dcom_array(z_distance, 0.3, 69.0)
        chi_of_z = InterpolatedUnivariateSpline(z_distance, chi_distance * 0.69 )
        # Setup the variables needed for sigma_crit_inverse
        self.chi_lens = chi_of_z(self.zs)
        self.chi_source = chi_of_z(self.zs)
        self.chi_lmat = np.tile(self.chi_lens.reshape(len(self.zs), 1), (1, len(self.zs)))
        self.chi_smat = np.tile(self.chi_source.reshape(1, len(self.zs)), (len(self.zs), 1))
        self.num = self.chi_smat - self.chi_lmat
        ind_lzero = np.where(self.num <= 0)
        self.num[ind_lzero] = 0

        # Simpson integration is linear: cache its weights on the fixed z grid.
        weights = simpson(np.eye(len(self.zs)), x=self.zs, axis=1)
        self._sigma_kernel = (1e13 * self.sigcrit_inv_fac_Mpc_Msun *
                              ((1 + self.zs) / self.chi_lens)[:, None] *
                              (self.num / self.chi_smat) * weights[:, None] * weights[None, :])
        ndata = len(self.used_items)
        lens_ids = sorted({int(item[1]) for item in self.used_items if item[0] == 2})
        self._pm_templates = np.zeros((len(lens_ids), ndata))
        self._pm_lens = np.zeros(ndata, dtype=int)
        self._pm_source = np.zeros(ndata, dtype=int)
        for index, (typ, lens, source, theta) in enumerate(self.used_items):
            if typ == 2:
                self._pm_templates[lens_ids.index(int(lens)), index] = self.theta_bins_radians[theta]**-2
                self._pm_lens[index], self._pm_source[index] = lens, source

    def get_ratio_from_gammat(self, gammat1, gammat2, inv_cov):
        ones = jnp.ones(len(gammat1))
        return ((gammat1 / gammat2) @ inv_cov @ ones) / (ones @ inv_cov @ ones)

    def make_vector(self, arrays):
        return jnp.stack([jnp.asarray(arrays[typ][f1, f2])[theta]
                          for typ, f1, f2, theta in self.used_items])

    def __call__(self):
        self.sigma_crit_inv = jnp.einsum('li,ij,sj->ls', self.theory.nz_lens,
                                        self._sigma_kernel, self.theory.nz_source)
        covinv = jnp.asarray(self.covinv_orig)
        if len(self._pm_templates):
            templates = self._pm_templates * self.sigma_crit_inv[self._pm_lens, self._pm_source]
            vc = templates @ covinv
            x = 1e-8 * jnp.eye(templates.shape[0]) + vc @ templates.T
            self.precision = covinv - vc.T @ jnp.linalg.solve(x, vc)
        else:
            self.precision = covinv
        self.covinv = self.precision
        self.flattheory = self.make_vector([self.theory.xip, self.theory.xim,
                                           self.theory.gammat, self.theory.wtheta])
        self.chi2_3x2pt = -2 * super().__call__()
        self.chi2_sr = jnp.asarray(0.)
        if self.use_sr:
            ratios = []
            for lens in range(self.sr_nbin_lens):
                reference = self.theory.gammat[lens, self.sr_nbin_source - 1]
                for source in range(self.sr_nbin_source - 1):
                    mask = self.sr_masks[source, lens]
                    ratios.append(self.get_ratio_from_gammat(self.theory.gammat[lens, source][mask],
                                                              reference[mask], self.inv_cov_individual_ratios[source, lens]))
            self.theory_ratios = jnp.stack(ratios)
            delta = self.theory_ratios - self.sr_data
            self.chi2_sr = delta @ self.sr_covinv @ delta
        self.logpdf = -.5 * (self.chi2_3x2pt + self.chi2_sr)
        return self.logpdf

    def tree_flatten(self):
        return [self.logpdf, self.flattheory, self.precision, self.chi2_3x2pt,
                self.chi2_sr, self.sigma_crit_inv], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        (obj.logpdf, obj.flattheory, obj.precision, obj.chi2_3x2pt,
         obj.chi2_sr, obj.sigma_crit_inv) = children
        obj.covinv = obj.precision
        return obj
