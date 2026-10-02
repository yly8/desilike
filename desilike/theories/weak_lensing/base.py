"""Reference FFTLog and static angular-transform helpers (NumPy/SciPy)."""
from functools import lru_cache
from pathlib import Path
from .data import resolve_data_dir

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import special
from cosmoprimo import constants as const

_spline = InterpolatedUnivariateSpline
trapz = getattr(np, 'trapezoid', None) or np.trapz


def interp1d(xq, x, f, method='cubic'):
    """Reference DES not-a-knot cubic spline, with polynomial extrapolation."""
    from scipy.interpolate import CubicSpline
    return CubicSpline(x, f, axis=0, extrapolate=True)(xq)


#nonLimber calculation depend on 1911.11947
def nonLimber(ells, kernel1_vals, kernel2_vals, chis, D_growth, PK, h, sigma, f_interp=None, bias1=None, bias2=None):
    chi_pad_upper=1.
    chi_pad_lower=1.
    chi_extrap_upper=1.
    chi_extrap_lower=1.

    if np.isnan(sigma):
        dlogchi = np.log(chis[-1]/chis[-2])
    else:
        dlogchi = sigma/3.5
    log_chimin, log_chimax = np.log(chis[0]), np.log(chis[-1])
    nchi = np.ceil((log_chimax-log_chimin)/dlogchi).astype(int)
    log_chi_vals = np.linspace(log_chimin, log_chimax, nchi)
    chi_vals = np.exp(log_chi_vals)

    if chi_pad_upper>0.:
        assert chi_pad_lower==chi_pad_upper
        N_pad = (np.ceil(float(chi_pad_upper)/dlogchi)).astype(int)
    else:
        N_pad = 0
    if chi_extrap_upper>0.:
        N_extrap_upper = (np.ceil(float(chi_extrap_upper)/dlogchi)).astype(int)
    else:
        N_extrap_upper = 0
    if chi_extrap_lower>0.:
        N_extrap_lower = (np.ceil(float(chi_extrap_lower)/dlogchi)).astype(int)
    else:
        N_extrap_lower = 0

    if np.all(kernel1_vals == kernel2_vals):
        w1p = kernel1_vals * D_growth * chis
        #w1 = _spline(chis, w1p)(chi_vals)
        w1 = interp1d(chi_vals, chis, w1p, method='cubic')
        w1 = np.nan_to_num(w1, nan=0.0)
        w2 = w1
    else:
        w1p = kernel1_vals * D_growth * chis
        w2p = kernel2_vals * D_growth * chis
        #w1 = _spline(chis, w1p)(chi_vals)
        #w2 = _spline(chis, w2p)(chi_vals)
        w1 = interp1d(chi_vals, chis, w1p, method='cubic')
        w1 = np.nan_to_num(w1, nan=0.0)
        w2 = interp1d(chi_vals, chis, w2p, method='cubic')
        w2 = np.nan_to_num(w2, nan=0.0)
    if f_interp is not None:
        if bias1 !=0 and bias2 != 0:
            f_vals = f_interp(chi_vals)
            w1_rsd = w1 * f_vals/bias1
            w2_rsd = w2 * f_vals/bias2
        else:
            f_vals = f_interp(chi_vals)
            w1_rsd = w1 * f_vals
            w2_rsd = w2 * f_vals

    cell = []
    for i_ell, ell in enumerate(ells):
        if np.all(kernel1_vals == kernel2_vals):
            k_vals, I_1 = FFT(chi_vals, w1, ell, N_extrap_upper, N_extrap_lower, N_pad)
            I_2 = I_1
            if f_interp is not None:
                k_vals_check, I_1_rsd = FFT(chi_vals, w1_rsd, ell, N_extrap_upper, N_extrap_lower, N_pad, nu=1.1)
                assert np.allclose(k_vals_check, k_vals)
                I_1 = I_1 - I_1_rsd
                I_2 = I_1
        else:
            k_vals, I_1 = FFT(chi_vals, w1, ell, N_extrap_upper, N_extrap_lower, N_pad)
            k_vals, I_2 = FFT(chi_vals, w2, ell, N_extrap_upper, N_extrap_lower, N_pad)
            if f_interp is not None:
                k_vals_check, I_1_rsd = FFT(chi_vals, w1_rsd, ell, N_extrap_upper, N_extrap_lower, N_pad, nu=1.1)
                k_vals_check, I_2_rsd = FFT(chi_vals, w2_rsd, ell, N_extrap_upper, N_extrap_lower, N_pad, nu=1.1)
                assert np.allclose(k_vals_check, k_vals)
                I_1 = I_1 - I_1_rsd
                I_2 = I_2 - I_2_rsd
        logk_vals = np.log(k_vals)
        pk_vals = PK(k_vals/h)/h**3
        integrand_vals = k_vals * k_vals * k_vals * pk_vals * I_1 * I_2
        #Spline and integrate the integrand.
        integrand_interp = InterpolatedUnivariateSpline(logk_vals, integrand_vals)
        integral = integrand_interp.integral(logk_vals.min(), logk_vals.max())
        integral *= 2./np.pi
        cell.append(integral)
    return np.array(cell)

def FFT(x, fx, ell, N_extrap_high = 0, N_extrap_low = 0, N_pad = 0, nu = 1.0):
    dlnx = np.log(x[1]/x[0])
    c_window_width = 0.25

    # extrapolate x and f(x) linearly in log(x), and log(f(x))
    x = log_extrap(x, N_extrap_low, N_extrap_high)
    fx = log_extrap(fx, N_extrap_low, N_extrap_high)
    N = np.size(x)

    # zero-padding
    if(N_pad):
        pad = np.zeros(N_pad)
        x = log_extrap(x, N_pad, N_pad)
        fx = np.hstack((pad, fx, pad))
        N += 2*N_pad
        N_extrap_high += N_pad
        N_extrap_low += N_pad

    if(N%2==1): # Make sure the array sizes are even
        x= x[:-1]
        fx=fx[:-1]
        N -= 1
        if(N_pad):
            N_extrap_high -=1

    f_b=fx * x**(-nu)
    c_m=np.fft.rfft(f_b)
    m=np.arange(0,N//2+1)
    c_m = c_m * c_window(m, int(c_window_width*N//2.) )
    eta_m = 2*np.pi/(float(N)*dlnx) * m

    z_ar = nu + 1j*eta_m
    y = (ell+1.) / x[::-1]
    if nu ==1:
        h_m = c_m * (x[0]*y[0])**(-1j*eta_m) * g_l(ell, z_ar)
    else:
        h_m = c_m * (x[0]*y[0])**(-1j*eta_m) * g_l_2(ell, z_ar)

    Fy = np.fft.irfft(np.conj(h_m)) * y**(-nu) * np.sqrt(np.pi)/4.
    if np.any(np.isnan(Fy)):
        print("found nans in Fy for ell=%s"%ell)
    return (y[N_extrap_high:N-N_extrap_low], Fy[N_extrap_high:N-N_extrap_low])

def c_window(n, n_cut):
    n_right = n[-1] - n_cut
    mask = (n > n_right)
    theta_right=(n[-1]-n)/float(n[-1]-n_right-1)
    taper = theta_right - 1/(2*np.pi) * np.sin(2*np.pi * theta_right)
    W = np.where(mask, taper, 1.0)
    return W

def log_extrap(x, N_extrap_low, N_extrap_high):
    low_x = np.array([])
    high_x = np.array([])
    if(N_extrap_low):
        if x[1] <= 1e-30 or x[0] <= 1e-30:
            low_x = np.zeros(N_extrap_low)
        else:
            dlnx_low = np.log(x[1]/x[0])
            low_x = x[0] * np.exp(dlnx_low * np.arange(-N_extrap_low, 0) )
    if(N_extrap_high):
        if x[-1] <= 1e-30 or x[-2] <= 1e-30:
            high_x = np.zeros(N_extrap_high)
        else:
            dlnx_high= np.log(x[-1]/x[-2])
            high_x = x[-1] * np.exp(dlnx_high * np.arange(1, N_extrap_high+1) )
    x_extrap = np.hstack((low_x, x, high_x))
    return x_extrap

def g_l(l, z_array):
    gl = 2.**z_array * g_m_vals(l+0.5,z_array-1.5)
    return gl

def g_l_2(l,z_array):
    gl2 = 2.**(z_array-2) *(z_array -1)*(z_array -2)* g_m_vals(l+0.5,z_array-3.5)
    return gl2

def g_m_vals(mu, q):
    imag_q= np.imag(q)
    g_m=np.zeros(q.size, dtype=complex)
    cut =200
    asym_q=q[np.absolute(imag_q)+ np.absolute(mu) >cut]
    asym_plus=(mu+1+asym_q)/2.
    asym_minus=(mu+1-asym_q)/2.

    q_good=q[ (np.absolute(imag_q)+ np.absolute(mu) <=cut) & (q!=mu + 1 + 0.0j)]

    alpha_plus=(mu+1+q_good)/2.
    alpha_minus=(mu+1-q_good)/2.

    g_m[(np.absolute(imag_q)+ np.absolute(mu) <=cut) & (q!= mu + 1 + 0.0j)] =special.gamma(alpha_plus)/special.gamma(alpha_minus)

    # asymptotic form
    g_m[np.absolute(imag_q)+ np.absolute(mu)>cut] = np.exp( (asym_plus-0.5)*np.log(asym_plus) - (asym_minus-0.5)*np.log(asym_minus) - asym_q \
        +1./12 *(1./asym_plus - 1./asym_minus) +1./360.*(1./asym_minus**3 - 1./asym_plus**3) +1./1260*(1./asym_plus**5 - 1./asym_minus**5) )

    g_m[np.where(q==mu+1+0.0j)[0]] = 0.+0.0j
    return g_m

def Pl_rec_binav(ells, cost_min, cost_max):
    """Calculate average Pl"""
    Pl_binav = np.zeros(len(ells))
    Pl_binav[0] = 1.
    # coefficients that are a function of ell only
    ell = ells[1:]
    coeff = 1./(2.*ell+1.)
    # computation of legendre polynomials
    # --- this computes all polynomials of order 0 to ell_max+1 and for all ell's
    lpns_min = special.lpn(ell[-1]+1, cost_min)[0]
    lpns_max = special.lpn(ell[-1]+1, cost_max)[0]
    # terms in the numerator of average Pl
    term_lm1 = lpns_max[:-2] - lpns_min[:-2]
    term_lp1 = lpns_max[2:] - lpns_min[2:]
    # denominator in average Pl
    dcost = cost_max-cost_min
    # computation of bin-averaged Pl(ell)
    Pl_binav[ell] = coeff * (term_lp1 - term_lm1) / dcost
    return Pl_binav

def get_legfactors_00_binav(ells, theta_edges):
    n_ell, n_theta = len(ells), len(theta_edges)-1
    #theta_edges = theta_bin_means_to_edges(thetas) # this does geometric mean
    legfacs = np.zeros((n_theta, n_ell))
    ell_factor = np.zeros(len(ells))
    ell_factor[1:] = (2 * ells[1:] + 1) / 4. / np.pi
    for it, t in enumerate(theta_edges[1:]):
        t_min = theta_edges[it]
        t_max = t
        cost_min = np.cos(t_min) # thetas are already converted to radians
        cost_max = np.cos(t_max)
        Pl = Pl_rec_binav(ells, cost_min, cost_max)
        legfacs[it] = Pl * ell_factor
    return legfacs

def P2l_rec_binav(ells, cost_min, cost_max):
    """Calculate P2l using recurrence relation for normalised P2l"""
    P2l_binav = np.zeros(len(ells))
    P2l_binav[0] = 0.
    P2l_binav[1] = 0.
    # coefficients that are a function of ell only
    ell = ells[2:]
    coeff_lm1 = ell+2./(2.*ell+1.)
    coeff_lp1 = 2./(2.*ell+1.)
    coeff_l   = 2.-ell
    # computation of legendre polynomials
    # --- this computes all polynomials of order 0 to ell_max+1 and for all ell's
    lpns_min = special.lpn(ell[-1]+1, cost_min)[0][1:]
    lpns_max = special.lpn(ell[-1]+1, cost_max)[0][1:]
    # terms in the numerator of average P2l
    term_lm1 = coeff_lm1 * (lpns_max[:-2]-lpns_min[:-2])
    term_lp1 = coeff_lp1 * (lpns_max[2:]-lpns_min[2:])
    term_l   = coeff_l   * (cost_max*lpns_max[1:-1]-cost_min*lpns_min[1:-1])
    # denominator in average P2l
    dcost = cost_max-cost_min
    # computation of bin-averaged P2l(ell)
    P2l_binav[ell] = (term_lm1 + term_l - term_lp1) / dcost
    return P2l_binav

def get_legfactors_02_binav(ells, theta_edges):
    n_ell, n_theta = len(ells), len(theta_edges)-1
    #theta_edges = theta_bin_means_to_edges(thetas) # this does geometric mean
    legfacs = np.zeros((n_theta, n_ell))
    ell_factor = np.zeros(len(ells))
    ell_factor[1:] = (2 * ells[1:] + 1) / 4. / np.pi / ells[1:] / (ells[1:] + 1)
    for it, t in enumerate(theta_edges[1:]):
        t_min = theta_edges[it]
        t_max = t
        cost_min = np.cos(t_min) # thetas are already converted to radians
        cost_max = np.cos(t_max)
        P2l = P2l_rec_binav(ells, cost_min, cost_max)
        legfacs[it] = P2l * ell_factor
    return legfacs

def Gp_plus_minus_Gm_binav(ells, cost_min, cost_max):
    """Calculate bin-averaged G_{l,2}^{+/-}"""
    Gp_plus_Gm  = np.zeros(len(ells))
    Gp_minus_Gm = np.zeros(len(ells))

    # for ell=0,1 it is 0
    Gp_plus_Gm[0:1]  = 0.
    Gp_minus_Gm[0:1] = 0.

    # for the rest of ell's compute equation (5.8) in https://arxiv.org/abs/1911.11947
    ell = ells[2:]
    #---coefficients including only P_l
    coeff_lm1    = -ell*(ell-1.)/2. * (ell+2./(2.*ell+1)) - (ell+2.)
    coeff_lp1    =  ell*(ell-1.)/(2.*ell+1.)
    coeff_l      = -ell*(ell-1.)*(2.-ell)/2.
    coeff_l_plus = -2.*(ell-1.)
    #---coefficients including dP_l/dx
    coeff_dlm1_plus = -2.*(ell+2.)
    coeff_xdl_plus  = 2.*(ell-1.)
    coeff_xdlm1     = ell+2.
    coeff_dl        = 4.-ell

    # computation of legendre polynomials
    #---this computes all polynomials of order 0 to ell_max+1 and for all ell's
    lpns_min  = special.lpn(ell[-1]+1, cost_min)[0][1:]
    lpns_max  = special.lpn(ell[-1]+1, cost_max)[0][1:]
    dlpns_min = special.lpn(ell[-1]+1, cost_min)[1][1:]
    dlpns_max = special.lpn(ell[-1]+1, cost_max)[1][1:]

    # denominator in average
    dcost = cost_max-cost_min

    # numerator in average
    #---common part in both plus and minus
    common_part  = coeff_lm1*(lpns_max[:-2]-lpns_min[:-2])
    common_part += coeff_l*(cost_max*lpns_max[1:-1] - cost_min*lpns_min[1:-1])
    common_part += coeff_lp1*(lpns_max[2:]-lpns_min[2:])
    common_part += coeff_dl*(dlpns_max[1:-1]-dlpns_min[1:-1])
    common_part += coeff_xdlm1*(cost_max*dlpns_max[:-2]-cost_min*dlpns_min[:-2])
    #---plus
    Gp_plus_Gm_extra  = coeff_xdl_plus*(cost_max*dlpns_max[1:-1]-cost_min*dlpns_min[1:-1])
    Gp_plus_Gm_extra += coeff_l_plus*(lpns_max[1:-1] - lpns_min[1:-1])
    Gp_plus_Gm_extra += coeff_dlm1_plus*(dlpns_max[:-2]-dlpns_min[:-2])
    Gp_plus_Gm[2:] = common_part + Gp_plus_Gm_extra
    Gp_plus_Gm /= dcost
    #---minus
    Gp_minus_Gm_extra = -Gp_plus_Gm_extra
    Gp_minus_Gm[2:] = common_part + Gp_minus_Gm_extra
    Gp_minus_Gm /= dcost

    return Gp_plus_Gm, Gp_minus_Gm

def get_legfactors_22_binav(ells, theta_edges):
    n_ell, n_theta = len(ells), len(theta_edges)-1
    #theta_edges = theta_bin_means_to_edges(thetas) # this does geometric mean
    leg_factors_p  = np.zeros((n_theta, n_ell))
    leg_factors_m = np.zeros((n_theta, n_ell))
    ell_factor = np.zeros(len(ells))
    ell_factor[2:] = (2 * ells[2:] + 1) / 2. / np.pi / ells[2:] / ells[2:] / (ells[2:]+1.) / (ells[2:]+1.)
    for it, t in enumerate(theta_edges[1:]):
        t_min = theta_edges[it]
        t_max = t
        cost_min = np.cos(t_min) # thetas are already converted to radians
        cost_max = np.cos(t_max)
        gp, gm = Gp_plus_minus_Gm_binav(ells, cost_min, cost_max)
        leg_factors_p[it] = gp * ell_factor
        leg_factors_m[it] = gm * ell_factor
    return [leg_factors_p, leg_factors_m]

def apply_filter(ell_max, high_l_filter, legfacs):
    f = sin_filter( ell_max, ell_max*high_l_filter)
    return legfacs * np.tile(f, (legfacs.shape[0],1))

def sin_filter(ell_max, ell_right):
    ells = np.arange(ell_max+1)
    y = (ell_max-ells)/(ell_max-ell_right)
    return np.where( ells>ell_right, y - np.sin(2*np.pi*y)/2/np.pi, np.ones(len(ells)) )


@lru_cache(maxsize=8)
def get_fourier(fourier, acc=1, l_max=40000, data_dir=None):
    data_dir = resolve_data_dir(data_dir)
    if fourier == 'legendre':
        theta = np.loadtxt(data_dir / 'DES_3YR_final_theta_bins.dat')
        theta_edges = theta[:,0]
        theta_edges = np.append(theta_edges, theta[-1,1])
        theta_bins = theta[:,2]
        theta_bins_radians = theta_bins / 60 * np.pi / 180
        theta_edges_radians = theta_edges / 60 * np.pi / 180
    elif fourier == 'binned_bessels':
        theta_bins = np.loadtxt(data_dir / 'DES_3YR_final_theta_bins.dat')
        theta_bins_radians = theta_bins[:, 2] / 60 * np.pi / 180

    # Note hankel assumes integral starts at ell=0
    # (though could change spline to zero at zero).
    # At percent level it matters what is assumed
    if fourier == 'binned_bessels':
        # Approximate bessel integral as binned smooth C_L against integrals of
        # bessel in each bin. Here we crudely precompute an approximation to the
        # bessel integral by brute force
        dls = np.diff(np.unique((np.exp(np.linspace(
            np.log(1.), np.log(l_max), int(500 * acc)))).astype(int)))
        groups = []
        ell = 2  # ell_min
        ls_bessel = np.zeros(dls.size)
        for i, dlx in enumerate(dls):
            ls_bessel[i] = (2 * ell + dlx - 1) / 2.
            groups.append(np.arange(ell, ell + dlx))
            ell += dlx
        js = np.empty((3, ls_bessel.size, len(theta_bins_radians)))
        bigell = np.arange(0, l_max + 1, dtype=np.float64)
        for i, theta in enumerate(theta_bins_radians):
            bigx = bigell * theta
            for ix, nu in enumerate([0, 2, 4]):
                bigj = special.jn(nu, bigx) * bigell / (2 * np.pi)
                for j, g in enumerate(groups):
                    js[ix, j, i] = np.sum(bigj[g])
        bessel_cache = js[0, :, :], js[1, :, :], js[2, :, :]
        return ls_bessel, bessel_cache
    elif fourier == 'legendre':
        high_l_filter = 0.75
        ls_legender = np.arange(l_max + 1)
        P_l = get_legfactors_00_binav(ls_legender, theta_edges_radians)
        P_l = apply_filter(l_max, high_l_filter, P_l)
        P_l_2 = get_legfactors_02_binav(ls_legender, theta_edges_radians)
        P_l_2 = apply_filter(l_max, high_l_filter, P_l_2)
        Gp_l, Gm_l = get_legfactors_22_binav(ls_legender, theta_edges_radians)
        Gp_l = apply_filter(l_max, high_l_filter, Gp_l)
        Gm_l = apply_filter(l_max, high_l_filter, Gm_l)
        legendre_cache = P_l.T, P_l_2.T, Gp_l.T, Gm_l.T
        return ls_legender, legendre_cache
