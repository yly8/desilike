# Notebooks

- basic_examples.ipynb: desilike basics, how to create a calculator (theory or likelihood), define parameters, fit and sample likelihood
- kaiser_implementation_examples.ipynb: implementation of a Kaiser theory, down to the likelihood
- bao_examples.ipynb: BAO fit, plot BAO wiggles, estimate detection level, sample the BAO likelihood, inference of Omega_m
- fs_shapefit_examples.ipynb: full shape fits with the shapefit parameterization
- fs_direct_examples.ipynb: full shape fits with the direct parameterization (i.e. base cosmological parameters)
- compression_examples.ipynb: constraints on cosmological parameters from BAO and shapefit constraints, comparison to direct constraints
- png_examples.ipynb: local primordial non-gaussianity fits
- turnover_examples.ipynb: turnover scale fits, inference of H0

Note: compression_examples.ipynb reads the chains written to `_tests/` by bao_examples.ipynb, fs_shapefit_examples.ipynb and fs_direct_examples.ipynb, so run those first.

- [desilike_test_wl.ipynb](desilike_test_wl.ipynb): DES Y3 weak-lensing validation against the external DES reference, including JAX eager/JIT timing and the bin-normalized k-dependent theory comparison in section 11.
- [desilike_weak_lensing_tutorial.ipynb](desilike_weak_lensing_tutorial.ipynb): computing weak-lensing spectra, correlations, and likelihoods with bundled data.

These are the English weak-lensing notebooks. Run/edit them at these paths (the former root-level `_en.ipynb` names are obsolete). Both locate the checkout from the kernel working directory, including when started in `nb/`.

From the repository root, execute either notebook with its current path:

```sh
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 nb/desilike_test_wl.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 nb/desilike_weak_lensing_tutorial.ipynb
```

The validation notebook also requires the external DES reference described in its setup section. Section 11 compares the k-dependent and original desilike theories with identical inputs. Section 12 separately validates the implementation against `des-k.py` beside the original reference; set `DES_Y3_K_REFERENCE` to override its location. Each section can be run independently after the setup cells in sections 0–1.
