"""Static angular-transform tables, prepared once outside the JAX evaluation graph."""
from functools import lru_cache
from pathlib import Path
from .data import resolve_data_dir

import numpy as np
from scipy import special
from cosmoprimo import constants as const

trapz = getattr(np, 'trapezoid', None) or np.trapz


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
