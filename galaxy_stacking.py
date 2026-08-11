#!/usr/bin/env python
"""
Per-galaxy spaxel stacking for MaNGA cubes.

For each galaxy in the input catalogue this builds continuum-subtracted,
velocity-aligned, aperture-weighted LINEAR stacked spectra in three apertures
(nuclear, disk-to-Re, total within Re) and writes them to a compressed FITS
table. The stacks are intended for weak emission-line measurements and
ionization diagnostics.

Stacking is done over spaxels WITHIN each galaxy (one stacked spectrum per
galaxy), not across galaxies. The deprojection helper is adapted from code by
Amir Musaeva.

Pipeline per galaxy
-------------------
1. Pull the DRP cube, DAP maps, and DAP model cube from Marvin (remote).
2. Correct observed flux and the stellar-continuum model for Galactic
   extinction (O'Donnell 1994, Rv = 3.1), then subtract the stellar model.
3. Shift every spaxel to rest frame using the systemic redshift and the
   DAP Halpha velocity field.
4. Resample each spaxel onto a common log-wavelength grid with a
   flux-conserving algorithm (spectres), falling back to linear
   interpolation if spectres is unavailable.
5. Co-add spaxels with fractional aperture weights (linear mean), propagate
   the noise from the DRP inverse variance, and inflate it by the empirical
   MaNGA covariance correction (Law et al. 2016).

Inputs (expected in the working directory)
------------------------------------------
uniform1_mass_sfr_sample.csv : columns plateifu, SFR_1RE
dapall-v3_1_1-3.1.0.fits     : DAPall table (used for NSA structural params)

Output
------
Stacked_Spectra/<plateifu>_stacked_spec.fits.gz : one BinTableHDU per galaxy
    with columns wave, fluxL_<aperture>, noiseL_<aperture> for each of the
    three apertures, plus per-aperture spaxel counts in the header.

Requirements
------------
marvin, astropy, numpy, scipy, pandas, extinction, photutils, spectres
"""

import os
import warnings

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from scipy import constants as cons
from scipy import interpolate

import marvin
from marvin import config
from marvin.tools import Maps, Cube, ModelCube
import extinction  # type: ignore
from photutils.aperture import CircularAperture, EllipticalAperture

# Flux-conserving resampling. If spectres is missing we fall back to linear
# interpolation, which does not conserve flux across bins but keeps the script
# runnable.
try:
    from spectres import spectres
    USE_SPECTRES = True
except ImportError:
    USE_SPECTRES = False

warnings.filterwarnings("ignore")


# ============================================================================
# Configuration
# ============================================================================

SAMPLE_CSV = 'uniform1_mass_sfr_sample.csv'   # catalogue of galaxies to stack
DAP_PATH = 'dapall-v3_1_1-3.1.0.fits'         # DAPall table
OUTPUT_DIR = 'Stacked_Spectra'                # per-galaxy FITS outputs

DR_RELEASE = 'DR17'

# Common rest-frame wavelength grid. min/max bracket the optical lines we care
# about; the 3900-point log spacing matches the native MaNGA sampling closely
# enough for line work.
MIN_WAVE = 3650.0
MAX_WAVE = 9150.0
N_GRID = 3900

# Aperture geometry (arcsec). MaNGA spaxels are 0.5 arcsec.
PIXEL_SCALE = 0.5
R_NUCLEAR_ARCSEC = 1.25   # radius of the circular nuclear aperture
R_DISK_INNER_ARCSEC = 1.25  # inner cut for the disk aperture (floored at 2.5 pix)

# Force single-threaded BLAS so many galaxies can be run in parallel processes
# without oversubscribing cores. Set False if running a single galaxy.
SINGLE_THREAD = True


# ============================================================================
# Derived constants
# ============================================================================

if SINGLE_THREAD:
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

C_KMS = cons.speed_of_light / 1000.0  # speed of light in km/s

# Log-spaced grid, and its linear counterpart (spectres wants linear wavelengths)
LOG_GRID = np.arange(
    np.log10(MIN_WAVE + 4),
    np.log10(MAX_WAVE - 4),
    (np.log10(MAX_WAVE) - np.log10(MIN_WAVE)) / N_GRID
)
LINEAR_GRID = 10 ** LOG_GRID


# ============================================================================
# Resampling
# ============================================================================

def resample_spectrum(old_wave, old_flux, new_wave, old_ivar=None):
    """
    Resample a single spaxel spectrum onto ``new_wave``.

    Uses the flux-conserving spectres algorithm when available, otherwise
    linear interpolation. Wavelengths must be LINEAR (not log). Inputs are
    sorted and de-duplicated first, as spectres requires strictly increasing,
    unique wavelengths.

    Parameters
    ----------
    old_wave : array
        Source wavelengths (linear), any order.
    old_flux : array
        Source flux, same length as ``old_wave``.
    new_wave : array
        Target wavelength grid (linear).
    old_ivar : array, optional
        Source inverse variance. If given, it is propagated to the new grid
        and returned; otherwise the second return value is None.

    Returns
    -------
    new_flux : array
        Resampled flux (zero-filled outside the input coverage).
    new_ivar : array or None
        Resampled inverse variance, or None if ``old_ivar`` was not supplied.
    """
    sort_idx = np.argsort(old_wave)
    old_wave = old_wave[sort_idx]
    old_flux = old_flux[sort_idx]
    if old_ivar is not None:
        old_ivar = old_ivar[sort_idx]

    # Drop duplicate wavelengths (spectres rejects them)
    _, unique_idx = np.unique(old_wave, return_index=True)
    unique_idx = np.sort(unique_idx)
    old_wave = old_wave[unique_idx]
    old_flux = old_flux[unique_idx]
    if old_ivar is not None:
        old_ivar = old_ivar[unique_idx]

    if len(old_wave) < 3:
        return np.zeros_like(new_wave), np.zeros_like(new_wave)

    if USE_SPECTRES:
        try:
            new_flux = spectres(new_wave, old_wave, old_flux, fill=0, verbose=False)
            if old_ivar is not None:
                # Resample in variance space, then invert. Bins outside the
                # input coverage get a huge variance (-> zero ivar).
                old_var = np.where(old_ivar > 0, 1.0 / old_ivar, 1e20)
                new_var = spectres(new_wave, old_wave, old_var, fill=1e20, verbose=False)
                new_ivar = np.where(new_var < 1e19, 1.0 / new_var, 0.0)
            else:
                new_ivar = None
            return new_flux, new_ivar
        except Exception:
            pass  # fall through to linear interpolation

    # Fallback: linear interpolation (does NOT conserve flux)
    f_flux = interpolate.interp1d(old_wave, old_flux, bounds_error=False, fill_value=0)
    new_flux = f_flux(new_wave)
    if old_ivar is not None:
        f_ivar = interpolate.interp1d(old_wave, old_ivar, bounds_error=False, fill_value=0)
        new_ivar = f_ivar(new_wave)
    else:
        new_ivar = None
    return new_flux, new_ivar


# ============================================================================
# Aperture construction
# ============================================================================

def build_aperture_masks(ny, nx, Re_pix, b_over_a, pa_rad):
    """
    Build fractional aperture weight maps for the three stacking regions.

    Apertures are centred on the cube centre. Fractional (anti-aliased) masks
    are used so partially covered spaxels contribute in proportion to their
    overlap.

    Parameters
    ----------
    ny, nx : int
        Cube spatial dimensions.
    Re_pix : float
        Effective radius in pixels.
    b_over_a : float
        Axis ratio (NSA_ELPETRO_BA).
    pa_rad : float
        Aperture position angle in radians.

    Returns
    -------
    dict
        {'nuclear_pix', 'disk_pix_to_Re', 'total_Re'} -> (ny, nx) weight maps.
    """
    x0 = (nx - 1) / 2.0
    y0 = (ny - 1) / 2.0

    r_nuc = R_NUCLEAR_ARCSEC / PIXEL_SCALE
    r_in_disk = max(2.5, R_DISK_INNER_ARCSEC / PIXEL_SCALE)

    nuclear = CircularAperture((x0, y0), r=r_nuc)
    inner = CircularAperture((x0, y0), r=r_in_disk)
    ellipse = EllipticalAperture((x0, y0), a=Re_pix, b=Re_pix * b_over_a, theta=pa_rad)

    def to_map(ap):
        m = ap.to_mask(method='exact').to_image((ny, nx))
        return m if m is not None else np.zeros((ny, nx))

    nuclear_mask = to_map(nuclear)
    inner_mask = to_map(inner)
    total_mask = to_map(ellipse)

    # Disk = elliptical Re aperture minus the inner circular core.
    # Clip so the boundary between the two masks cannot go negative.
    disk_mask = np.clip(total_mask - inner_mask, 0, 1)

    return {
        "nuclear_pix": nuclear_mask,
        "disk_pix_to_Re": disk_mask,
        "total_Re": total_mask,
    }


def covariance_factor(n_spx):
    """
    Empirical MaNGA covariance correction (Law et al. 2016).

    Neighbouring MaNGA spaxels are correlated by cube reconstruction, so the
    formal propagated noise underestimates the true uncertainty in a stacked
    aperture. This multiplicative factor inflates the noise accordingly.
    """
    if n_spx <= 1:
        return 1.0
    if n_spx < 100:
        return 1 + 1.62 * np.log10(n_spx)
    return 4.2


# ============================================================================
# Main loop
# ============================================================================

def main():
    # Load the galaxy catalogue
    sample = pd.read_csv(SAMPLE_CSV)
    sample['plateifu'] = sample['plateifu'].astype(str)
    ids = sample['plateifu'].values
    sfr_1re = sample['SFR_1RE'].values

    # Configure Marvin for remote public DR17 access
    config.setRelease(DR_RELEASE)
    marvin.config.mode = 'remote'
    marvin.config.access = 'public'

    # DAPall table: source of NSA structural parameters and redshift
    dapall = fits.open(DAP_PATH)[1].data

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not USE_SPECTRES:
        print("WARNING: spectres not installed; using linear interpolation "
              "(flux not conserved). Install with: pip install spectres")

    for o, obj in enumerate(ids):

        # Skip galaxies already processed, so reruns don't re-download cubes
        outfile = f"{OUTPUT_DIR}/{obj}_stacked_spec.fits.gz"
        if os.path.exists(outfile):
            print(f"Skipping {obj} (already done)")
            continue

        try:
            # ---- Fetch data products -------------------------------------
            maps = Maps(plateifu=obj, mode='remote')
            cube = Cube(plateifu=obj, mode='remote')
            modelcube = ModelCube(plateifu=obj, mode='remote')

            plate = maps.plate
            ifudesign = maps.ifu

            # Locate this galaxy in DAPall for structural params and redshift
            w_obj = (dapall['PLATE'] == plate) & (dapall['IFUDESIGN'] == ifudesign)
            z = dapall['NSA_Z'][w_obj][0]

            wave = cube.flux.wavelength.value  # observed wavelengths (linear)
            flux = cube.flux.value             # (nwave, ny, nx)
            flux_ivar = cube.flux.ivar

            # ---- Galactic extinction correction --------------------------
            # NSA g-band extinction -> E(B-V) -> A_V, O'Donnell 1994 curve.
            ext = maps.nsa['extinction']
            Ag = ext[1]
            ebv = Ag / 3.793
            Rv = 3.1
            Av = Rv * ebv
            A_lambda = extinction.odonnell94(wave, Av, Rv)

            # Deredden the observed flux BEFORE subtracting the stellar model
            flux = flux * 10 ** (0.4 * A_lambda)[:, None, None]

            # Stellar continuum = full DAP model minus the emission-line model
            stellar_model = modelcube.full_fit.value - modelcube.emline_fit.value
            # Apply the SAME extinction correction to the model before subtracting
            stellar_model = stellar_model * 10 ** (0.4 * A_lambda)[:, None, None]

            flux = flux - stellar_model  # continuum-subtracted emission spectrum

            # ---- Velocity alignment --------------------------------------
            # Rest-frame wavelength per spaxel: remove systemic redshift and the
            # local Halpha velocity so lines stack without rotational smearing.
            vel = maps.emline_gvel_ha_6564
            wl_z = wave / (1 + z)
            wl = wl_z[:, None, None] / (1 + vel.value / C_KMS)

            # ---- Aperture masks ------------------------------------------
            ny, nx = flux.shape[1], flux.shape[2]
            Re = dapall['NSA_ELPETRO_TH50_R'][w_obj][0]
            Re_pix = Re / PIXEL_SCALE
            b_over_a = dapall['NSA_ELPETRO_BA'][w_obj][0]
            pa_rad = np.deg2rad(90 - dapall['NSA_ELPETRO_PHI'][w_obj][0])

            aperture_masks = build_aperture_masks(ny, nx, Re_pix, b_over_a, pa_rad)

            # ---- Stack per aperture --------------------------------------
            out = Table()
            out['wave'] = LINEAR_GRID.astype('float32')
            header_counts = {}

            for bin_name, weights in aperture_masks.items():

                # Accumulators over the common grid
                stack_flux = np.zeros(len(LOG_GRID))   # sum of weight * flux
                stack_count = np.zeros(len(LOG_GRID))  # sum of weights (normaliser)
                stack_var = np.zeros(len(LOG_GRID))    # sum of weight * variance
                n_spx = 0.0                            # total aperture weight

                for y, x in np.argwhere(weights > 0):
                    # argwhere returns (row, col) = (y, x); flux is (nwave, ny, nx)
                    wave_sp = wl[:, y, x]
                    flux_sp = flux[:, y, x]
                    ivar_sp = flux_ivar[:, y, x]

                    good = (ivar_sp > 0) & (wave_sp > 0) & np.isfinite(flux_sp)
                    if np.sum(good) < 10:
                        continue

                    flux_interp, ivar_interp = resample_spectrum(
                        wave_sp[good], flux_sp[good], LINEAR_GRID, old_ivar=ivar_sp[good]
                    )

                    if ivar_interp is None:
                        continue
                    valid = ivar_interp > 0
                    w_spax = weights[y, x]

                    # Aperture-weighted linear (mean) stack
                    stack_flux[valid] += flux_interp[valid] * w_spax
                    stack_count[valid] += w_spax
                    stack_var[valid] += (1.0 / ivar_interp[valid]) * w_spax

                    n_spx += w_spax

                # Normalise the weighted sum to a weighted mean
                has_data = stack_count > 0
                stack_flux[has_data] /= stack_count[has_data]

                # Propagated noise, then covariance-inflated
                stack_noise = np.zeros_like(stack_flux)
                stack_noise[has_data] = np.sqrt(stack_var[has_data]) / stack_count[has_data]
                stack_noise[has_data] *= covariance_factor(n_spx)

                out[f"fluxL_{bin_name}"] = stack_flux.astype('float32')
                out[f"noiseL_{bin_name}"] = stack_noise.astype('float32')

                # Effective spaxel count for this aperture (scalar -> header)
                header_counts[bin_name] = float(np.nanmean(stack_count[has_data])) \
                    if has_data.any() else 0.0

            # ---- Write output --------------------------------------------
            hdu = fits.BinTableHDU(out)
            hdu.header['OBJID'] = obj
            hdu.header['SFR'] = float(sfr_1re[o])
            for bin_name, val in header_counts.items():
                # FITS keywords are <=8 chars; use a short prefix per aperture
                key = 'NSPX_' + bin_name.split('_')[0][:3].upper()
                hdu.header[key] = val

            hdu.writeto(outfile, overwrite=True)
            print(f"Saved: {outfile}")

        except Exception as e:
            print(f"Error with {obj}: {e}")


if __name__ == '__main__':
    main()
