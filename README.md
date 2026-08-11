# MaNGA Stacking and Gaussian Line-Fitting Pipeline

## Overview

This pipeline has two scripts:

1. `galaxy_stacking.py`  
   Creates continuum-subtracted, velocity-aligned stacked spectra for each galaxy.

2. `gaussian_fit_stacked.py`  
   Fits Gaussian emission-line models to the stacked spectra.

Run the stacking script first, then run the Gaussian-fitting script.

## Required Input Files

Place these files in the working directory:

```text
uniform1_mass_sfr_sample.csv
dapall-v3_1_1-3.1.0.fits
```

The CSV file must contain:

```text
plateifu
SFR_1RE
```

## Requirements

Install the required Python packages:

```bash
pip install numpy pandas scipy astropy matplotlib extinction photutils spectres sdss-marvin
```

The scripts use Marvin remote access to public MaNGA DR17 data.

## How to Run

First run:

```bash
python galaxy_stacking.py
```

This creates stacked spectra in:

```text
Stacked_Spectra/
```

Then run:

```bash
python gaussian_fit_stacked.py
```

## Outputs

The stacking script creates one FITS file per galaxy:

```text
Stacked_Spectra/<plateifu>_stacked_spec.fits.gz
```

The Gaussian-fitting script creates:

```text
gaussian_fit_results.csv
diagnostics_linear_only.pdf
```

## Apertures

Each galaxy is stacked in three apertures:

```text
nuclear_pix
disk_pix_to_Re
total_Re
```

## Gaussian-Fit Results

`gaussian_fit_results.csv` contains emission-line measurements for each galaxy and aperture, including:

```text
objid
region
line
flux
flux_err
snr
chi2_red
upper_limit
```

For Halpha, the output also includes continuum and equivalent-width measurements.

## Notes

- `galaxy_stacking.py` must be run before `gaussian_fit_stacked.py`.
- Existing stacked FITS files are skipped automatically.
- `spectres` is recommended because it performs flux-conserving resampling.
