"""NLA DES 3x2pt: native JAX projection with a separate reference FFTLog node."""
from pathlib import Path
import numpy as np
import yaml
import jax
import jax.numpy as jnp
from scipy.interpolate import InterpolatedUnivariateSpline

from desilike.base import Calculator
from desilike.parameter import Parameter, VariableCollection
from desilike.theories.primordial_cosmology import CosmoprimoCosmology
from .base import get_fourier, nonLimber, trapz
from .data import resolve_data_dir


data_path = str(resolve_data_dir())  # Compatibility alias; constructors resolve the current package location.


def _parameters():
    with open(Path(__file__).with_name('weak_lensing.yaml')) as stream:
        config = next(yaml.safe_load_all(stream))['params']
    params = VariableCollection()
    for name, original in config.items():
        attrs = dict(original) if isinstance(original, dict) else {'value': original}
        delta = attrs.pop('delta', None)
        if delta is not None:
            attrs['fd'] = {'eps': delta[0] if isinstance(delta, list) else delta}
        if 'value' not in attrs and 'loc' in attrs.get('ref', {}):
            attrs['value'] = attrs['ref']['loc']
        params.set(Parameter(name, **attrs))
    return params


def _spline_coefficients(x, y):
    """Not-a-knot coefficients using an O(n) differentiable tridiagonal solve.

    Avoid the dense n-by-n system used by older cubic2 implementations:
    spectra have thousands of knots and hundreds of redshift columns.
    """
    x, y = jnp.asarray(x), jnp.asarray(y)
    n = len(x)
    if n < 4:
        raise ValueError('The DES cubic spline requires at least four knots')
    values = y.reshape((n, -1))
    dx = jnp.diff(x)
    slopes = jnp.diff(values, axis=0) / dx[:, None]
    diag = jnp.concatenate((dx[1:2], 2 * (dx[:-1] + dx[1:]), dx[-2:-1]))
    upper = jnp.concatenate(((x[2] - x[0])[None], dx[:-1], jnp.zeros(1)))
    lower = jnp.concatenate((jnp.zeros(1), dx[1:], (x[-1] - x[-3])[None]))
    d0, dn = x[2] - x[0], x[-1] - x[-3]
    rhs0 = ((dx[0] + 2 * d0) * dx[1] * slopes[0] + dx[0]**2 * slopes[1]) / d0
    rhsn = (dx[-1]**2 * slopes[-2] + (2 * dn + dx[-1]) * dx[-2] * slopes[-1]) / dn
    rhs = jnp.concatenate((rhs0[None], 3 * (dx[1:, None] * slopes[:-1] + dx[:-1, None] * slopes[1:]), rhsn[None]))
    def matvec(v):
        return (diag[:, None] * v + lower[:, None] * jnp.concatenate((jnp.zeros_like(v[:1]), v[:-1]))
                + upper[:, None] * jnp.concatenate((v[1:], jnp.zeros_like(v[:1]))))
    def solve(_, b):
        return jax.lax.linalg.tridiagonal_solve(lower, diag, upper, b)
    def transpose_solve(_, b):
        return jax.lax.linalg.tridiagonal_solve(jnp.concatenate((jnp.zeros(1), upper[:-1])), diag,
                                               jnp.concatenate((lower[1:], jnp.zeros(1))), b)
    derivatives = jax.lax.custom_linear_solve(matvec, rhs, solve=solve, transpose_solve=transpose_solve)
    a = (derivatives[:-1] + derivatives[1:] - 2 * slopes) / dx[:, None]**2
    b = (3 * slopes - 2 * derivatives[:-1] - derivatives[1:]) / dx[:, None]
    shape = (n - 1,) + y.shape[1:]
    return tuple(v.reshape(shape) for v in (values[:-1], derivatives[:-1], b, a))


def spline(xq, x, y, derivative=0):
    """Native JAX equivalent of scipy InterpolatedUnivariateSpline (including extrapolation)."""
    xq, x, y = jnp.asarray(xq), jnp.asarray(x), jnp.asarray(y)
    index = jnp.clip(jnp.searchsorted(x, xq) - 1, 0, len(x) - 2)
    t = (xq - x[index]).reshape(xq.shape + (1,) * (y.ndim - 1))
    value, slope, b, a = (v[index] for v in _spline_coefficients(x, y))
    if derivative == 1:
        return slope + t * (2 * b + 3 * t * a)
    return value + t * (slope + t * (b + t * a))


def _spline_columns(xq, x, y):
    """Evaluate each redshift column on its own Limber wave-number grid."""
    index = jnp.clip(jnp.searchsorted(x, xq) - 1, 0, len(x) - 2)
    t = xq - x[index]
    col = jnp.arange(y.shape[1])[None, :]
    value, slope, b, a = (v[index, col] for v in _spline_coefficients(x, y))
    return value + t * (slope + t * (b + t * a))


def ell_grid(fourier):
    if fourier == 'legendre':
        return np.concatenate((np.linspace(1., 4., 5), np.geomspace(5., 1e5, 80)))
    return np.hstack((np.arange(2., 96., 4.), np.geomspace(100., 40000., 50)))


class _ArrayOutputs(Calculator):
    def tree_flatten(self):
        return [getattr(self, name) for name in self._outputs], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        for name, value in zip(cls._outputs, children):
            setattr(obj, name, value)
        return obj


class DESLensingKernels(_ArrayOutputs):
    """Native JAX redshift kernels and Limber power integrands.

    All P(k) leaves use k in h/Mpc and P in (Mpc/h)^3. The integration itself
    uses Mpc and physical k, as in the reference DES likelihood.
    """
    _outputs = ('h', 'chis', 'Hs', 'growth', 'growth_rate', 'pk_linear0', 'nz_lens',
                'nz_source', 'qgal', 'qsw', 'tmp', 'tmpnonlimber', 'tmplens', 'tmpmw')

    def __init__(self, cosmo, params, data_dir, des_model, nzbins, nwbins, fourier, Weyl, k):
        self.cosmo, self.params = cosmo, params
        self.data_dir, self.des_model = data_dir, des_model
        self.nzbins, self.nwbins = nzbins, nwbins
        self.fourier, self.Weyl, self.k = fourier, Weyl, k

    def __post_init__(self, *args, **kwargs):
        year = self.des_model.split('_')[1]
        source = np.loadtxt(Path(self.data_dir) / f'DES_{year}_final_nz_source.dat')
        lens = np.loadtxt(Path(self.data_dir) / f'DES_{year}_final_nz_lens.dat')
        if not np.array_equal(source[:, 1], lens[:, 1]):
            raise ValueError('Source and lens redshift grids must agree')
        self.z = source[:, 1]
        self.z_pk = np.concatenate(([0.], self.z))
        self.zbin_sp = source[:, 3:3 + self.nzbins].T
        self.zbin_w_sp = lens[:, 3:3 + self.nwbins].T
        # Reference FFTLog resolution uses the original, unshifted lens distributions.
        self.std_z = []
        for nz in self.zbin_w_sp:
            mean = trapz(self.z * nz, self.z) / trapz(nz, self.z)
            self.std_z.append(np.sqrt(trapz(nz * (self.z - mean)**2, self.z) / trapz(nz, self.z)))
        self.ls_cl = ell_grid(self.fourier)
        spectra = [dict(of='delta_m', non_linear=False), dict(of='delta_m', non_linear=True)]
        if self.Weyl:
            spectra += [dict(of='phi_plus_psi', non_linear=False),
                        dict(of=('delta_m', 'phi_plus_psi'), non_linear=False)]
        self.cosmo.add_requirements({
            'fourier.pk': [dict(z=self.z_pk, k=self.k, extrap_kmax_physical=8000., **spec) for spec in spectra],
            'background.comoving_radial_distance': [dict(z=self.z)],
            'background.efunc': [dict(z=self.z)],
            'params.h': None, 'params.Omega_b': None, 'params.Omega_cdm': None,
            'params.Omega_ncdm_tot': None})

    def spectrum(self, of='delta_m', non_linear=False):
        return jnp.asarray(self.cosmo.get('fourier.pk', z=self.z_pk, k=self.k, of=of,
                                         non_linear=non_linear, extrap_kmax_physical=8000.))

    def __call__(self):
        self.h = self.cosmo['h']
        h = self.h
        # Reference CAMB omegam includes the total massive-neutrino energy,
        # whereas cosmoprimo['Omega_m'] subtracts its pressure contribution.
        omegam = self.cosmo['Omega_b'] + self.cosmo['Omega_cdm'] + self.cosmo['Omega_ncdm_tot']
        zs = jnp.asarray(self.z)
        self.chis = chis = self.cosmo.get('background.comoving_radial_distance', z=self.z) / h
        self.Hs = Hs = 100 * h * self.cosmo.get('background.efunc', z=self.z) / 299792.458
        dchis = jnp.concatenate(((chis[1:2] + chis[:1]) / 2, (chis[2:] - chis[:-2]) / 2, chis[-1:] - chis[-2:-1]))
        linear, nonlinear = self.spectrum(), self.spectrum(non_linear=True)
        self.pk_linear0 = linear[0]
        logk = jnp.log(self.k)
        # Reference growth is at physical k=0.05/Mpc and normalized at exactly z=0.
        growth_pk = jnp.exp(spline(jnp.log(.05 / h), logk, jnp.log(linear).T))
        self.growth = growth = jnp.sqrt(growth_pk[1:] / growth_pk[0])
        loga = -jnp.log1p(zs)
        self.growth_rate = spline(loga, loga[::-1], jnp.log(growth)[::-1], derivative=1)
        lensing = jnp.where(chis[None, :] >= chis[:, None],
                            (1 - chis[:, None] / chis[None, :]) * dchis[None, :], 0.)
        p = {name: param.value for name, param in self.params.items()}
        lens_nz = []
        for b in range(self.nwbins):
            zshift = zs - p[f'DES_DzL{b + 1}']
            nz = jnp.where(zshift < 0, 0., spline(zshift, zs, self.zbin_w_sp[b]))
            nz = nz / jnp.trapezoid(nz, zshift)
            zmean = jnp.sum(zs * nz) / jnp.sum(nz)
            z_bias = p[f'DES_szL{b + 1}'] * (zs - zmean) + zmean
            nz = spline(zs, z_bias, jnp.where(z_bias < 0, 0., nz))
            lens_nz.append(nz / jnp.trapezoid(nz, zs))
        self.nz_lens = jnp.stack(lens_nz)
        prefactor = 3 * omegam * h**2 * (1e5 / 299792458.)**2 * chis * (1 + zs) / 2
        n_chi = Hs * self.nz_lens
        magnification = jnp.asarray([.43, .30, 1.75, 1.94, 1.56, 2.96])[:self.nwbins]
        bias = jnp.stack([p[f'DES_b{i + 1}'] for i in range(self.nwbins)])
        self.qgal = n_chi * bias[:, None] + (n_chi @ lensing.T) * prefactor * magnification[:, None]
        alignment = p['DES_AIA1'] * ((1 + zs) / (1 + p['DES_z0IA']))**p['DES_alphaIA1'] * .0134 / growth
        alignment = alignment / (chis * (1 + zs) * 3 * h**2 * (1e5 / 299792458.)**2 / 2)
        source_nz, source_chi = [], []
        for b in range(self.nzbins):
            zshift = zs - p[f'DES_DzS{b + 1}']
            nz = spline(zshift, zs, self.zbin_sp[b])
            nz = nz / jnp.trapezoid(nz, zshift)
            source_nz.append(nz)
            source_chi.append(jnp.where(zshift < 0, 0., Hs * nz))
        self.nz_source = jnp.stack(source_nz)
        n_chi = jnp.stack(source_chi)
        wq = n_chi @ lensing.T - alignment * n_chi
        self.qsw = (chis if self.Weyl else prefactor) * wq
        physical_k = (jnp.asarray(self.ls_cl)[:, None] + .5) / chis
        kh = physical_k / h
        def sample(pk):
            return jnp.exp(_spline_columns(jnp.log(kh), logk, jnp.log(pk[1:]).T))
        pl, pnl = sample(linear), sample(nonlinear)
        # Cobaya's DES logp requests extrap_kmax=8000/Mpc (not a cutoff at 100).
        weight = jnp.where((physical_k >= 1e-4) & (physical_k < 8000.), dchis / chis**2, 0.)
        self.tmp = weight * pnl / h**3
        self.tmpnonlimber = weight * (pnl - pl) / h**3
        if self.Weyl:
            # cosmoprimo stores phi+psi and a positive matter-potential cross power;
            # CAMB/Cobaya use W=k_phys^2(phi+psi)/2 and a negative delta-W cross.
            self.tmplens = weight * (pnl / pl) * sample(self.spectrum('phi_plus_psi')) * physical_k**4 / (4 * h**3)
            self.tmpmw = weight * (pnl / pl) * sample(self.spectrum(('delta_m', 'phi_plus_psi'))) * physical_k**2 / (2 * h**3)
        else:
            self.tmplens = self.tmpmw = self.tmp
        return self


class _NonLimberInputs(_ArrayOutputs):
    """Transfer only FFTLog inputs across the host callback boundary."""
    _outputs = ('h', 'chis', 'growth', 'growth_rate', 'pk_linear0', 'qgal')

    def __init__(self, kernels):
        self.kernels = kernels

    def __call__(self):
        for name in self._outputs:
            setattr(self, name, getattr(self.kernels, name))
        return self


class DESNonLimber(_ArrayOutputs):
    """Reference SciPy/FFTLog, with cosmology-dependent integer grid sizes.

    Kept as an external node to preserve the reference discretization exactly.
    desilike supplies parameter finite-difference derivatives across this node.
    """
    _is_external = True
    _outputs = ('cl',)

    def __init__(self, kernels, biases):
        self.inputs, self.biases = _NonLimberInputs(kernels), biases

    def __post_init__(self, *args, **kwargs):
        self.k = self.inputs.kernels.k
        self.std_z = self.inputs.kernels.std_z
        self.ells = self.inputs.kernels.ls_cl[self.inputs.kernels.ls_cl < 200]

    def __call__(self):
        kernels = self.inputs
        chis, growth, rates = [np.asarray(getattr(kernels, name)) for name in ('chis', 'growth', 'growth_rate')]
        logpk = InterpolatedUnivariateSpline(np.log(self.k), np.log(np.asarray(kernels.pk_linear0)))
        pk0 = lambda k: np.exp(logpk(np.log(k)))
        rate_spline = InterpolatedUnivariateSpline(chis, rates)
        self.cl = np.stack([nonLimber(self.ells, np.asarray(q), np.asarray(q), chis, growth,
                                     pk0, float(kernels.h), sigma, rate_spline,
                                     float(bias.value), float(bias.value))
                            for q, sigma, bias in zip(np.asarray(kernels.qgal), self.std_z, self.biases)])
        return self


class DESWeakLensing3x2pt(_ArrayOutputs):
    """NLA DES theory. Kernels and projections are native JAX; FFTLog is external."""
    _outputs = ('xip', 'xim', 'gammat', 'wtheta', 'ell', 'cl_xip', 'cl_xim',
                'cl_gammat', 'cl_wtheta', 'chis', 'Hs', 'nz_lens', 'nz_source')

    def __init__(self, cosmo=None, fiducial='DESI', des_model='DES_3YR',
                 ia_model=None, Limber=None, Weyl=False, fourier=None,
                 nzbins=None, nwbins=None, bin_pairs=None, data_dir=None,
                 engine='camb', params=None, k=None):
        if des_model not in ('DES_1YR', 'DES_3YR'):
            raise ValueError('Supported DES models are DES_1YR and DES_3YR')
        year = des_model.split('_')[1]
        self.des_model = des_model
        self.ia_model = ia_model or 'NLA'
        if self.ia_model != 'NLA':
            raise ValueError('Only the NLA intrinsic-alignment model is supported')
        self.Limber = (year == '1YR') if Limber is None else Limber
        self.Weyl = Weyl
        self.fourier = fourier or ('binned_bessels' if year == '1YR' else 'legendre')
        if self.fourier not in ('binned_bessels', 'legendre'):
            raise ValueError('fourier must be binned_bessels or legendre')
        self.nzbins = 4 if nzbins is None else nzbins
        self.nwbins = (5 if year == '1YR' else 6) if nwbins is None else nwbins
        self.bin_pairs = bin_pairs if bin_pairs is not None else {
            'xip': [(i, j) for i in range(self.nzbins) for j in range(i, self.nzbins)],
            'xim': [(i, j) for i in range(self.nzbins) for j in range(i, self.nzbins)],
            'gammat': [(i, j) for i in range(self.nwbins) for j in range(self.nzbins)],
            'wtheta': [(i, i) for i in range(self.nwbins)]}
        self.data_dir = str(resolve_data_dir(data_dir))
        defaults = _parameters()
        for param in VariableCollection(params): defaults.set(param)
        self.params = {p.basename: p for p in defaults if p.basename.startswith('DES_')}
        cosmoparams = VariableCollection([p for p in defaults if not p.basename.startswith('DES_')])
        self.cosmo = cosmo if cosmo is not None else CosmoprimoCosmology(
            engine=engine, fiducial=fiducial, params=cosmoparams,
            precision={'calc_params': {'non_linear': 'takahashi', 'kmax_pk': 150.,
                                       'z_pk': np.linspace(0., 3., 100)}})
        self._k = np.geomspace(1e-5, 8e3, 4096) if k is None else np.asarray(k)
        kernel_params = {name: p for name, p in self.params.items() if not name.startswith('DES_m')}
        self.kernels = DESLensingKernels(self.cosmo, kernel_params, self.data_dir, des_model,
                                         self.nzbins, self.nwbins, self.fourier, Weyl, self._k)
        self.nonlimber = None if self.Limber else DESNonLimber(
            self.kernels, [self.params[f'DES_b{i + 1}'] for i in range(self.nwbins)])

    def __post_init__(self, *args, **kwargs):
        self.z = self.kernels.z
        self._ls_cl = self.kernels.ls_cl
        self._ell, cache = get_fourier(self.fourier, data_dir=self.data_dir)
        if self.fourier == 'legendre':
            self._wtransform, self._gtransform, self._ptransform, self._mtransform = cache
        else:
            self._ptransform, self._gtransform, self._mtransform = cache
            self._wtransform = self._ptransform
        self._masks = {}
        for name, shape in [('xip', (self.nzbins, self.nzbins)), ('xim', (self.nzbins, self.nzbins)),
                            ('gammat', (self.nwbins, self.nzbins)), ('wtheta', (self.nwbins, self.nwbins))]:
            mask = np.zeros(shape)
            for i, j in self.bin_pairs[name]: mask[i, j] = 1.
            self._masks[name] = mask

    def __call__(self):
        k = self.kernels
        def project(tmp, q1, q2): return jnp.einsum('lz,iz,jz->lij', tmp, q1, q2)
        rawp = project(k.tmplens, k.qsw, k.qsw)
        rawg = project(k.tmpmw, k.qgal, k.qsw)
        raww = project(k.tmp, k.qgal, k.qgal)
        if self.nonlimber is not None:
            nlow = int(np.sum(self._ls_cl < 200))
            rawfront = project(k.tmpnonlimber, k.qgal, k.qgal)
            diagonal = jnp.eye(self.nwbins, dtype=raww.dtype)[None, :, :]
            low = rawfront[:nlow] + self.nonlimber.cl.T[:, :, None]
            raww = jnp.concatenate((raww[:nlow] * (1 - diagonal) + low * diagonal, raww[nlow:]))
        def angular(raw):
            return jnp.moveaxis(spline(self._ell, self._ls_cl, raw), 0, -1)
        self.cl_xip = angular(rawp) * self._masks['xip'][..., None]
        self.cl_xim = angular(rawp) * self._masks['xim'][..., None]
        self.cl_gammat = angular(rawg) * self._masks['gammat'][..., None]
        self.cl_wtheta = angular(raww) * self._masks['wtheta'][..., None]
        m = jnp.stack([1 + self.params[f'DES_m{i + 1}'].value for i in range(self.nzbins)])
        self.xip = (self.cl_xip @ self._ptransform) * (m[:, None] * m[None, :])[..., None]
        self.xim = (self.cl_xim @ self._mtransform) * (m[:, None] * m[None, :])[..., None]
        self.gammat = (self.cl_gammat @ self._gtransform) * m[None, :, None]
        self.wtheta = self.cl_wtheta @ self._wtransform
        self.ell = jnp.asarray(self._ell, dtype='f8')
        for name in ('chis', 'Hs', 'nz_lens', 'nz_source'): setattr(self, name, getattr(k, name))
        return self
