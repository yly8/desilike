# DES weak lensing on refactor-jax

Migrated from the local `jax` branch (`67c61b1`) onto upstream `refactor-jax`
(`cf153007`). The original branch is retained. DES Y3 uses NLA intrinsic
alignment; the TATT implementation, its three extra nuisance parameters and
FAST-PT dependency have been removed.

Install this branch and its dependencies in the Python environment you use:

```sh
python -m pip install --upgrade 'cosmoprimo @ git+https://github.com/cosmodesi/cosmoprimo' 'lsstypes @ git+https://github.com/adematti/lsstypes'
python -m pip install -e '.[weak-lensing]'
```

The full non-Limber graph was validated with JAX 0.6.2 (CPU, float64). On this
Apple Silicon machine JAX 0.4.38 crashes while executing that compiled graph;
the `weak-lensing` extra therefore requires JAX >= 0.6.2 and compatible
interpax/Equinox versions. Validation used interpax 0.3.15 and Equinox 0.13.8.

The old cosmoprimo installation used with `jax` may lack APIs required by
`refactor-jax` (for example `fd_stencil`). Check any paths in your desilike
installer configuration if it selects an older installation.

```python
import jax
from desilike import build
from desilike.likelihoods.weak_lensing import BaseDESY3Likelihood

likelihood = BaseDESY3Likelihood(use_sr=True)
loglike = build(likelihood)
print(loglike())
print(jax.jit(loglike)({'omega_cdm': 0.125, 'DES_AIA1': 0.5}))
```

The default theory uses CAMB with Takahashi nonlinear power, NLA, non-Limber
clustering and the Legendre transform. `use_sr` defaults to `False`. Data files
are bundled in `des3`. All readers resolve this directory from the installed
package, so moving/reinstalling desilike or changing the working directory does
not require editing paths. A custom `data_dir` accepts `str`/`Path`, relative
paths and `~`. Relative custom paths are interpreted from the working directory.
A likelihood inherits its supplied theory's data directory; an explicit
likelihood `data_dir` configures both the likelihood and theory consistently.
Dataset entries are resolved relative to the `.dataset` file.

```python
# Portable bundled-data default:
likelihood = BaseDESY3Likelihood(use_sr=True)
# Optional user data directory:
likelihood = BaseDESY3Likelihood(data_dir="~/data/des-y3", use_sr=True)
```

The comparison notebook discovers the checkout at runtime and uses bundled
data for desilike. The separate reference repository is located relative to
the checkout's parent by default; set `DES_Y3_REFERENCE` to its `des.py` path
and, if needed, `DES_Y3_REFERENCE_DATA` to its data directory when it lives
elsewhere. These settings affect only reference comparisons.
 For a shared
cosmology provider, pass a `CosmoprimoCosmology` with
`precision={'calc_params': {'non_linear': 'takahashi', 'kmax_pk': 150.,
'z_pk': np.linspace(0., 3., 100)}}` for the comparison precision used here.

Theory parameters are graph nodes, and power spectra/background quantities
are requested through the new cosmology requirements API on every evaluation.
Use `build(calculator)(parameter_values)` instead of the old direct parameter
call, and `update(...)` instead of `init.update(...)` before building.
The optional Cosmosis adapter takes explicit nuisance `params` and a Cosmosis
installation/configuration; it is not needed for the native DES calculation.

The NLA redshift kernels, not-a-knot cubic interpolation, Limber integrands,
angular projection, and likelihood (including point-mass marginalization and
shear ratios) use native JAX operations, including non-Limber FFTLog. There is
one numerical implementation; call the built pipeline directly for JAX eager,
or wrap it with `jax.jit` for compiled execution. The former NumPy/SciPy
non-Limber backend and `nonlimber_backend` option have been removed.
NumPy/SciPy are still used for file loading and parameter-independent setup
(e.g. fixed angular-transform tables); CAMB remains an external provider.

The native path uses Bluestein convolution to evaluate variable-length DFTs in
fixed-capacity workspaces. Active integer grid lengths, extrapolation, padding,
RSD Mellin kernels and cubic integration follow the reference at every point;
the grid is not frozen at a fiducial cosmology. Workspace sizes are chosen from
the input redshift grid with headroom; exceeding capacity returns NaN rather
than silently truncating the physical grid. Gradients are piecewise within an
integer-grid region; grid transitions themselves are not differentiable.
The power-of-two convolution FFTs use native JAX radix-2 butterflies, including
under `jit` and `vmap`. This avoids an intermittent CPU FFT deadlock observed
with jaxlib 0.11.2 on Linux: concurrent ducc0 FFT calls occupied the Eigen worker
pool while waiting for nested tasks in that same pool. The implementation does
not change process-wide thread settings or use a NumPy/SciPy runtime fallback.
The physical FFTLog grid and normalization are unchanged. Butterfly execution
can be slower than the native FFT backend; benchmark the full likelihood on
the target machine when selecting sampler batch sizes.
CAMB remains external and uses parameter finite differences. Nuisance parameters
through the native non-Limber node now have a native autodiff path.
Unused bin pairs are zero-filled.

The numerical conventions follow the local reference DES Y3 implementation:
physical k=0.05/Mpc for growth, exact z=0 growth normalization, original lens
redshift widths for FFTLog, cubic-spline extrapolation, and a physical
8000/Mpc extrapolation cutoff. The default CAMB calculation uses kmax=150 h/Mpc
and 100 redshifts from 0 to 3 before extrapolation. A shared cosmology should
use these precision settings too. The matter density in the lensing kernel
includes total massive-neutrino energy, matching CAMB's `omegam`; cosmoprimo's
`Omega_m` subtracts the neutrino pressure contribution and is not identical.

Point-mass bin selections are explicit arrays/static templates. The reference's
NumPy integer bin labels also give elementwise list comparisons; the earlier
claim that these comparisons necessarily disabled marginalization was incorrect.
The covariance correction and chi-squared convention match the reference,
including its omission of a covariance log-determinant term.

The executed `desilike_test.ipynb` at the repository root contains the staged
CAMB, angular spectrum, correlation, Weyl-switch and chi-squared comparison
for 13 LCDM cosmologies, followed by JIT and gradient checks. The original
notebook is backed up under `bak/`. Tables, plots and source fingerprints are
saved under `diagnostics/weak_lensing_consistency/`.

Run the targeted regression tests with:

```sh
python -m pytest -q desilike/theories/weak_lensing/tests desilike/likelihoods/weak_lensing/tests
```

These cover the reference cubic interpolation, two-dimensional cosmology indexing,
both transforms, parameter sensitivity, JIT, finite-difference gradients,
point-mass correction and shear ratios. An additional CAMB integration check
uses the bundled DES Y3 data with NLA/non-Limber/shear ratios and checks the
JIT response to a change in `omega_cdm`.

## Notebooks

- Validation and timing: [English](../../../nb/desilike_test_wl.ipynb).
- Usage tutorial: [English](../../../nb/desilike_weak_lensing_tutorial.ipynb).

The English editions retain the numerical comparisons and saved results of their Chinese counterparts. Setup uses installed dependencies unless explicitly overridden through `DESILIKE_LSSTYPES_PATH` or `DESILIKE_COSMOPRIMO_PATH`. The validation notebook requires the external DES reference; the tutorial uses bundled data only. Execution commands are in [nb/README.md](../../../nb/README.md).

Regression tests live in both `theories/weak_lensing/tests` and `likelihoods/weak_lensing/tests`, and are discovered by the existing CI suites. They include invalid bin/grid settings, required likelihood bin pairs, covariance/data ordering, complete JIT node outputs, and shear-ratio on/off behavior. The Y3 likelihood requires matching source/lens bin counts and all bin pairs used by its selected data (including shear ratios when enabled).

## Bin-normalized k-dependent non-Limber theory

`DESKDependentWeakLensing3x2pt` implements the separable approximation in the
local reference `des-k.py`. It has the same constructor and nuisance parameters
as `DESWeakLensing3x2pt` and works with the existing likelihood:

```python
from desilike.theories.weak_lensing import DESKDependentWeakLensing3x2pt

likelihood = BaseDESY3Likelihood(
    theory=DESKDependentWeakLensing3x2pt(Limber=False, Weyl=True), use_sr=True)
loglike = build(likelihood)
print(jax.jit(loglike)({'DES_DzL1': 0.012, 'DES_szL1': 1.12}))
```

For each lens auto-bin, the linear non-Limber integral uses `P(k, z_mean)`
instead of `P(k, 0)`, and the radial kernel uses
`g_eff(z) = sqrt(P(k_ref, z) / P(k_ref, z_mean))`, with physical
`k_ref = 0.05/Mpc`. The RSD growth rate is `d ln(g_eff) / d ln(a)`;
there is no additional k-dependent multiplier on the RSD transform.
`z_mean` is computed after the lens photo-z shift and before the width stretch.
Power and growth normalization are interpolated with differentiable JAX splines.
For separable growth the normalization cancels and reproduces the original theory.

This is not a full arbitrary unequal-time `P(k,z1,z2)` calculation. As in the
reference, only the non-Limber clustering auto-bins use the new approximation.
Cross-bin clustering retains the existing Limber treatment. The nonlinear
Limber correction, shear, galaxy–galaxy lensing, intrinsic alignments and
likelihood conventions are unchanged; `Limber=True` reproduces the base theory.

Section 11 of both validation notebooks compares the new theory against the
original `DESWeakLensing3x2pt`, sharing the same cosmology node and settings.
It reports physical changes in angular spectra, correlations and chi-squared
for 13 LCDM cosmologies plus a lens photo-z/width perturbation, with Weyl off/on.
Each theory also receives an eager/JIT consistency check. Tables and bin-pair
plots are saved to `diagnostics/weak_lensing_consistency/k_dependence_models/`.

Section 12 separately validates the implementation against `des-k.py`, using
independent matched CAMB calculations. Set `DES_Y3_K_REFERENCE` if `des-k.py`
is not beside the original `des.py`. The reference file is only needed for
this notebook validation, not at runtime in the theory or likelihood.
Reference accuracy tables are saved to
`diagnostics/weak_lensing_consistency/k_dependence/`.
