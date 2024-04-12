# PolyBin3D
PolyBin3D is a Python code that estimates the binned power spectrum and bispectrum for 3D fields such as the distributions of matter and galaxies, using the algorithms of [Philcox 2020](https://arxiv.org/abs/2012.09389), [Philcox 2021](https://arxiv.org/abs/2107.06287), [Ivanov et al. 2023](https://arxiv.org/abs/2302.04414) and [Philcox & Flöss 2024](https://arxiv.org/abs/2404.07249). It is a sister code to [PolyBin](https://github.com/oliverphilcox/PolyBin), which computes the polyspectra of data on the two-sphere and is a modern reimplementation of the former [Spectra-Without-Windows](https://github.com/oliverphilcox/Spectra-Without-Windows) code. 

For each statistic, two estimators are available: the standard (ideal) estimators, which do not take into account the mask, and window-deconvolved estimators. In the second case, we require computation of a Fisher matrix; this depends on binning and the mask, but does not need to be recomputed for each new simulation.

The code supports GPU acceleration using JAX, which can be enabled using the `backend` argument in the `base` class, as demonstrated below.

PolyBin contains the following modules:
- `pspec`: Binned power spectra
- `bspec`: Binned bispectra

The basic usage of the power spectrum class is the following:
```
# Import code
import PolyBin3D as pb
import numpy as np

# Load base class
base = pb.PolyBin3D(boxsize, # dimensions of box 
                    gridsize, # dimensions of Fourier-space grid, 
                    boxcenter=[0,0,0], # center of simulation box
                    pixel_window='tsc', # pixel window function
                    backend='fftw', # backend for performing FFTs ('fftw' for cpu, 'jax' for gpu)
                    nthreads=4, # number of CPUs for performing FFTs (only applies to 'fftw' backend)
                    sightline='global') # redshift-space axis                    

# Load power spectrum class
pspec = pb.PSpec(base, 
                 k_bins, # k-bin edges
                 lmax, # Legendre multipoles
                 mask, # real-space mask
                 applySinv, # filter to apply to data
                )

# Compute Fisher matrix and shot-noise using Monte Carlo simulations (should usually be parallelized)
fish, shot_num = pspec.compute_fisher(10, N_cpus=1, verb=True)

# Compute windowed power spectra
Pk_ideal = pspec.Pk_ideal(data) 

# Compute unwindowed power spectra, using the Fisher matrix we just computed
Pk_unwindowed = pspec.Pk_unwindowed(data, fish=fish, shot_num=shot_num, subtract_shotnoise=False)
```

Bispectra can be computed similarly:
```
# Load bispectrum class
bspec = pb.BSpec(base, 
                 k_bins, # k-bin edges
                 lmax, # Legendre multipoles
                 mask, # real-space mask
                 applySinv, # filter to apply to data
                )

# Compute Fisher matrix using Monte Carlo simulations (should usually be parallelized)
fish = bspec.compute_fisher(10, N_cpus=1, verb=True)

# Compute windowed bispectra
Bk_ideal = bspec.Bk_ideal(data) 

# Compute unwindowed bispectra using the Fisher matrix we just computed
Bk_unwindowed = bspec.Bk_unwindowed(data, fish=fish, include_linear_term=False)
```

Further details are described in the tutorials, which describe
- [Tutorial 1](Tutorial%201%20-%20Pk%20from%20Simulations.ipynb): introduction to PolyBin3D, and computing the power spectrum from simulations
- [Tutorial 2](Tutorial%202%20-%20Validating%20the%20Unwindowed%20Pk%20Estimators.ipynb): validation of the window-deconvolved power spectrum estimators
- [Tutorial 3](Tutorial%203%20-%20BOSS%20Pk%20Multipoles.ipynb): application of the power spectrum esitmators to the BOSS DR12 dataset
- [Tutorial 4](Tutorial%204%20-%20Bk%20from%20Simulations.ipynb): introduction to computing bispectra
- [Tutorial 5](Tutorial%205%20-%20Validating%20the%20Unwindowed%20Bk%20Estimators.ipynb): validation of the window-deconvolved bispectrum estimators

## Authors
- [Oliver Philcox](mailto:ohep2@cantab.ac.uk) (Columbia / Simons Foundation)
- [Thomas Flöss](mailto:tsfloss@gmail.com) (University of Groningen)

## Dependencies
- Python 2/3
- numpy, scipy
- fftw [for FFTs]
- Nbodykit [not required, but useful for testing]
- JAX (for GPU acceleration, see [here](https://jax.readthedocs.io/en/latest/installation.html) for installation instructions.)

## References
**Code references:**
1. **Philcox, O. H. E. & Flöss, T.: "PolyBin3D: A Suite of Optimal and Efficient Power Spectrum and Bispectrum Estimators for Large-Scale Structure", (2024) ([arXiv](https://arxiv.org/abs/2404.07249))**
2. Philcox, O. H. E., "Cosmology Without Window Functions: Quadratic Estimators for the Galaxy Power Spectrum", (2020) ([arXiv](https://arxiv.org/abs/2012.09389))
3. Philcox, O. H. E., "Cosmology Without Window Functions: Cubic Estimators for the Galaxy Bispectrum", (2021) ([arXiv](https://arxiv.org/abs/2107.06287))
4. Ivanov, M. M., Philcox, O. H. E., et al. "Cosmology with the Galaxy Bispectrum Multipoles: Optimal Estimation and Application to BOSS Data" (2023) ([arXiv](https://arxiv.org/abs/2302.04414))

**Some works using data from PolyBin3D (or its predecessor)**
- Philcox & Ivanov (2021, [arXiv](https://arxiv.org/abs/2112.04515)): Combined constraints on LambdaCDM from the BOSS power spectrum and bispectrum.
- Cabass et al. (2022, [arXiv](https://arxiv.org/abs/2201.07238)): Constraints on single-field inflation from the BOSS power spectrum and bispectrum.
- Cabass et al. (2022, [arXiv](https://arxiv.org/abs/2204.01781)): Constraints on multi-field inflation from the BOSS power spectrum and bispectrum.
- Nunes et al. (2022, [arXiv](https://arxiv.org/abs/2203.08093)): Constraints on dark-sector interactions from the BOSS galaxy power spectrum.
- Rogers et al. (2023, [arXiv](https://arxiv.org/abs/2301.08361)): Ultra-light axions and the S8 tension: joint constraints from the cosmic microwave background and galaxy clustering.
- Ivanov et al. (2023, [arXiv](https://arxiv.org/abs/2302.04414)): Cosmology with the Galaxy Bispectrum Multipoles: Optimal Estimation and Application to BOSS Data.
- Moretti et al. (2023, [arXiv](https://arxiv.org/abs/2306.09275)): Constraints on the growth index and neutrino mass from the BOSS power spectrum.
- He et al. (2023, [arXiv](https://arxiv.org/abs/2309.03956)): Self-Interacting Neutrinos in Light of Large-Scale Structure Data.
- Camarena et al. (2023, [arXiv](https://arxiv.org/abs/2309.03941)): The two-mode puzzle: Confronting self-interacting neutrinos with the full shape of the galaxy power spectrum 
