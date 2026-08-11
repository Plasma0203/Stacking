# MaNGA Per-Galaxy Spaxel Stacking and Emission-Line Fitting

This pipeline stacks MaNGA spaxels within individual galaxies and measures weak emission lines from the stacked spectra.

## Scripts

1. `galaxy_stacking.py`  
   Builds continuum-subtracted, velocity-aligned, aperture-weighted stacked spectra for each galaxy.

2. `gaussian_fit_stacked.py`  
   Fits Gaussian emission-line models to the stacked spectra and writes flux measurements.

## Required input files

Place these files in the working directory:

```text
uniform1_mass_sfr_sample.csv
dapall-v3_1_1-3.1.0.fits
