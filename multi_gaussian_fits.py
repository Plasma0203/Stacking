#!/usr/bin/env python
"""
Emission-line measurement from per-galaxy stacked spectra.

Reads the linear stacked spectra produced by galaxy_stacking.py and measures
emission-line fluxes with local Gaussian fits, one fit set per aperture per
galaxy. Blended complexes (Halpha+[NII], the [OII] doublet, the [SII] doublet)
are fit jointly; isolated lines are fit with a single Gaussian on a local
continuum. For the Halpha complex the local continuum is also used to derive an
Halpha equivalent width.

A line is a detection at S/N >= 3; below that a 3-sigma upper limit is recorded
and the row is flagged. All fluxes are multiplied back up by the aperture's
effective spaxel count (N_spx), converting the mean stacked flux to a summed
aperture flux.

Input
-----
INPUT_DIR/<plateifu>_stacked_spec.fits.gz, each a BinTableHDU with columns
    wave, fluxL_<aperture>, noiseL_<aperture>
and header keywords OBJID and NSPX_<APER> written by galaxy_stacking.py.

Output
------
gaussian_fit_results.csv : one row per (galaxy, aperture, line)
diagnostics_linear_only.pdf : per-galaxy, per-aperture fit panels

Requirements
------------
astropy, numpy, scipy, pandas, matplotlib
"""

import os
import warnings

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")


# ============================================================================
# Configuration
# ============================================================================

# Must match OUTPUT_DIR in galaxy_stacking.py
INPUT_DIR = 'Stacked_Spectra'

RESULTS_CSV = 'gaussian_fit_results.csv'
DIAGNOSTICS_PDF = 'diagnostics_linear_only.pdf'

APERTURES = ['nuclear_pix', 'disk_pix_to_Re', 'total_Re']

SNR_DETECTION = 3.0   # S/N threshold separating detections from upper limits

# Lines fit per aperture. Value is the rest-frame centre in Angstrom. Halpha and
# the two doublets are special-cased below; the rest are single-Gaussian fits.
LINES = {
    "Hb 4861": 4861,
    "OIII 5007": 5007,
    "OI 6300": 6300,
    "Ha 6563": 6563,
    "SII 6716": 6716,
    "SII 6731": 6731,
    "OII 3727": 3728,
    "NeIII 3869": 3869,
    "HeII 4686": 4686,
}


# ============================================================================
# Line-profile models
# ============================================================================

def gaussian(x, A, mu, sigma, c0):
    """Single Gaussian on a flat continuum."""
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c0


def double_gaussian_OII(x, A_3727, A_3729, mu_3727, sigma, c0, c1):
    """
    [OII] 3726,3729 doublet: two Gaussians with a shared width and a fixed
    physical separation, on a linear local continuum centred at 3727.5 A.
    """
    delta_OII = 3728.82 - 3726.03  # fixed doublet separation, 2.79 A
    mu_3729 = mu_3727 + delta_OII
    continuum = c0 + c1 * (x - 3727.5)
    return (
        A_3727 * np.exp(-(x - mu_3727)**2 / (2 * sigma**2)) +
        A_3729 * np.exp(-(x - mu_3729)**2 / (2 * sigma**2)) +
        continuum
    )


def triple_gaussian_Ha_NII(x, A_Ha, A_NII_6584, mu, sigma, c0, c1):
    """
    Halpha + [NII] 6548,6584 complex, single shared width (matches the DAP).

    The [NII] doublet ratio is fixed by atomic physics
    (A_6548 = A_6584 / 2.96). Line offsets are fixed relative to Halpha.
    Continuum is linear, centred at 6563 A.
    """
    d6548 = 6548.05 - 6562.85
    d6584 = 6583.45 - 6562.85
    A_NII_6548 = A_NII_6584 / 2.96
    continuum = c0 + c1 * (x - 6563)
    return (
        A_Ha * np.exp(-(x - mu)**2 / (2 * sigma**2)) +
        A_NII_6548 * np.exp(-(x - (mu + d6548))**2 / (2 * sigma**2)) +
        A_NII_6584 * np.exp(-(x - (mu + d6584))**2 / (2 * sigma**2)) +
        continuum
    )


def double_gaussian_SII(x, A1, A2, mu1, mu2, sigma, c0):
    """[SII] 6716,6731 doublet: two Gaussians with a shared width, flat continuum."""
    return (
        A1 * np.exp(-(x - mu1)**2 / (2 * sigma**2)) +
        A2 * np.exp(-(x - mu2)**2 / (2 * sigma**2)) +
        c0
    )


# ============================================================================
# Flux, continuum, EW, and their uncertainties
# ============================================================================

def flux_and_error_from_cov(popt, pcov, A_idx, sigma_idx):
    """
    Integrated Gaussian flux F = A * sigma * sqrt(2 pi) and its uncertainty,
    propagated from the (A, sigma) covariance submatrix:

        sigma_F^2 = J C J^T,  J = [dF/dA, dF/dsigma] = [sigma sqrt(2pi), A sqrt(2pi)]
    """
    A = popt[A_idx]
    sigma = popt[sigma_idx]
    flux = A * sigma * np.sqrt(2 * np.pi)

    J = np.array([sigma * np.sqrt(2 * np.pi), A * np.sqrt(2 * np.pi)])
    cov_sub = pcov[np.ix_([A_idx, sigma_idx], [A_idx, sigma_idx])]
    var = J @ cov_sub @ J
    return flux, np.sqrt(np.abs(var))


def continuum_at_wavelength(popt, pcov, c0_idx, c1_idx, lambda_eval, lambda_ref):
    """
    Linear continuum c0 + c1*(lambda - lambda_ref) evaluated at lambda_eval,
    with uncertainty from the (c0, c1) covariance submatrix.
    """
    c0 = popt[c0_idx]
    c1 = popt[c1_idx]
    dx = lambda_eval - lambda_ref
    cont = c0 + c1 * dx

    J = np.array([1.0, dx])
    cov_sub = pcov[np.ix_([c0_idx, c1_idx], [c0_idx, c1_idx])]
    var = J @ cov_sub @ J
    return cont, np.sqrt(np.abs(var))


def equivalent_width(line_flux, line_flux_err, cont, cont_err):
    """
    EW = line_flux / continuum_flux_density, with standard ratio error
    propagation. Returns (nan, nan) if inputs are non-finite or the continuum
    is non-positive.
    """
    if (not np.isfinite(line_flux) or not np.isfinite(line_flux_err)
            or not np.isfinite(cont) or not np.isfinite(cont_err)
            or cont <= 0):
        return np.nan, np.nan

    ew = line_flux / cont
    ew_err = np.sqrt(
        (line_flux_err / cont) ** 2
        + ((line_flux * cont_err) / (cont ** 2)) ** 2
    )
    return ew, ew_err


def upper_limit_flux(noise_window, dlambda, fwhm):
    """3-sigma flux upper limit from the local noise, pixel scale, and line FWHM."""
    return 3 * np.median(noise_window) * dlambda * np.sqrt(fwhm / dlambda)


# ============================================================================
# Single-line fit
# ============================================================================

def fit_gaussian_line(wave, flux, noise, center, window=20):
    """
    Fit one isolated line with a single Gaussian on a flat continuum.

    Returns
    -------
    flux_val, flux_err, popt, chi2_red, x, y, model
        Any of these are nan / None if the fit could not be performed.
    """
    mask = (wave > center - window) & (wave < center + window)
    if np.sum(mask) < 10:
        return np.nan, np.nan, None, np.nan, None, None, None

    x = wave[mask]
    y = flux[mask]
    yerr = noise[mask]

    bounds = ([0, center - 5, 0.5, -np.inf],
              [np.inf, center + 5, 10, np.inf])

    try:
        popt, pcov = curve_fit(
            gaussian, x, y,
            p0=[np.max(y), center, 2.0, np.median(y)],
            sigma=yerr, absolute_sigma=True, bounds=bounds, maxfev=20000
        )
        model = gaussian(x, *popt)
        chi2_red = np.sum(((y - model) / yerr) ** 2) / (len(x) - 4)
        flux_val, flux_err = flux_and_error_from_cov(popt, pcov, 0, 2)
        return flux_val, flux_err, popt, chi2_red, x, y, model
    except Exception:
        return np.nan, np.nan, None, np.nan, None, None, None


# ============================================================================
# Per-file processing
# ============================================================================

def read_nspx(header, region):
    """
    Recover the aperture's effective spaxel count from the FITS header.

    galaxy_stacking.py writes NSPX_<first 3 letters of aperture, upper>, i.e.
    NSPX_NUC / NSPX_DIS / NSPX_TOT. Defaults to 1.0 if the keyword is absent
    (so older files without it still run, just without the N_spx rescaling).
    """
    key = 'NSPX_' + region.split('_')[0][:3].upper()
    return float(header.get(key, 1.0))


def process_file(fits_file, region, pdf, results):
    """Fit every line in one aperture of one galaxy, append rows to ``results``."""
    with fits.open(fits_file) as hdu:
        data = hdu[1].data
        header = hdu[1].header
        wave = data["wave"]
        objid = header.get("OBJID", "UNKNOWN")
        n_spx = read_nspx(header, region)

    flux = data[f'fluxL_{region}']
    noise = data[f'noiseL_{region}']

    dlambda = np.median(np.diff(wave))

    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    axes = axes.flatten()

    for i, (name, center) in enumerate(LINES.items()):
        ax = axes[i]

        # ---- Halpha + [NII] complex --------------------------------------
        if name == "Ha 6563":
            mask = (wave > 6520) & (wave < 6610)
            x, y, yerr = wave[mask], flux[mask], noise[mask]
            bounds = (
                [0, 0, 6563 - 3, 1.0, -np.inf, -np.inf],
                [np.inf, np.inf, 6563 + 3, 5.0, np.inf, np.inf]
            )
            try:
                popt, pcov = curve_fit(
                    triple_gaussian_Ha_NII, x, y,
                    p0=[np.max(y) - np.median(y), (np.max(y) - np.median(y)) / 3,
                        6563, 2.0, np.median(y), 0.0],
                    sigma=yerr, absolute_sigma=True, bounds=bounds, maxfev=20000
                )

                sigma = popt[3]
                fwhm = 2.355 * sigma
                model = triple_gaussian_Ha_NII(x, *popt)
                chi2_red = np.sum(((y - model) / yerr) ** 2) / (len(x) - 6)

                flux_Ha, err_Ha = flux_and_error_from_cov(popt, pcov, 0, 3)
                flux_6584, err_6584 = flux_and_error_from_cov(popt, pcov, 1, 3)
                flux_6548, err_6548 = flux_6584 / 2.96, err_6584 / 2.96

                snr_ha = flux_Ha / err_Ha if err_Ha > 0 else 0
                snr_nii = flux_6584 / err_6584 if err_6584 > 0 else 0

                flux_upper = upper_limit_flux(yerr, dlambda, fwhm)

                # Halpha continuum and equivalent width
                cont_Ha, cont_Ha_err = continuum_at_wavelength(
                    popt, pcov, c0_idx=4, c1_idx=5,
                    lambda_eval=6562.85, lambda_ref=6563.0
                )
                EW_Ha, EW_Ha_err = equivalent_width(flux_Ha, err_Ha, cont_Ha, cont_Ha_err)
                EW_Ha_abs = np.abs(EW_Ha) if np.isfinite(EW_Ha) else np.nan

                EW_Ha_upper = EW_Ha_upper_abs = np.nan
                if np.isfinite(flux_upper) and np.isfinite(cont_Ha) and cont_Ha > 0:
                    EW_Ha_upper = flux_upper / cont_Ha
                    EW_Ha_upper_abs = np.abs(EW_Ha_upper)

                is_upper_ha = snr_ha < SNR_DETECTION
                is_upper_nii = snr_nii < SNR_DETECTION

                ax.plot(x, y, label="data")
                ax.plot(x, yerr, alpha=0.4)
                ax.plot(x, model, label="fit")
                ax.legend(fontsize=7)

                if snr_ha >= SNR_DETECTION or snr_nii >= SNR_DETECTION:
                    label = (f"Ha={flux_Ha:.2e}±{err_Ha:.2e}\n"
                             f"NII={flux_6584:.2e}±{err_6584:.2e}\n"
                             f"cont_Ha={cont_Ha:.2e}\n"
                             f"EW_Ha={EW_Ha_abs:.2f} Å\n"
                             f"χ²={chi2_red:.2f}\n"
                             f"SNR_Ha={snr_ha:.1f}, SNR_NII={snr_nii:.1f}")
                else:
                    label = (f"<Ha {flux_upper:.2e}\n<NII {flux_upper:.2e}\n"
                             f"cont_Ha={cont_Ha:.2e}\n"
                             f"EW_Ha upper={EW_Ha_upper_abs:.2f} Å\n"
                             f"SNR_Ha={snr_ha:.1f}, SNR_NII={snr_nii:.1f}")

                results.append({
                    "objid": objid, "region": region, "line": "Ha 6563",
                    "flux": (flux_Ha if snr_ha >= SNR_DETECTION else flux_upper) * n_spx,
                    "flux_err": (err_Ha if snr_ha >= SNR_DETECTION else flux_upper / 3.0) * n_spx,
                    "snr": snr_ha, "chi2_red": chi2_red, "upper_limit": is_upper_ha,
                    "Ha_continuum_6563": cont_Ha * n_spx,
                    "Ha_continuum_6563_err": cont_Ha_err * n_spx,
                    "EW_Ha_6563": EW_Ha, "EW_Ha_6563_err": EW_Ha_err,
                    "EW_Ha_6563_abs": EW_Ha_abs,
                    "EW_Ha_6563_upper": EW_Ha_upper,
                    "EW_Ha_6563_upper_abs": EW_Ha_upper_abs,
                    "EW_Ha_upper_limit": is_upper_ha,
                })
                results.append({
                    "objid": objid, "region": region, "line": "NII 6584",
                    "flux": (flux_6584 if snr_nii >= SNR_DETECTION else flux_upper) * n_spx,
                    "flux_err": (err_6584 if snr_nii >= SNR_DETECTION else flux_upper / 3.0) * n_spx,
                    "snr": snr_nii, "chi2_red": chi2_red, "upper_limit": is_upper_nii,
                })
                results.append({
                    "objid": objid, "region": region, "line": "NII 6548",
                    "flux": (flux_6548 if snr_nii >= SNR_DETECTION else flux_upper / 2.96) * n_spx,
                    "flux_err": (err_6548 if snr_nii >= SNR_DETECTION else (flux_upper / 2.96) / 3.0) * n_spx,
                    "snr": snr_nii, "chi2_red": chi2_red, "upper_limit": is_upper_nii,
                })
            except Exception:
                label = "fit fail"

            ax.set_title("Hα + NII")
            ax.text(0.03, 0.97, label, transform=ax.transAxes, va='top',
                    fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
            continue

        # ---- [OII] doublet -----------------------------------------------
        if name == "OII 3727":
            mask = (wave > 3710) & (wave < 3745)
            x, y, yerr = wave[mask], flux[mask], noise[mask]

            if len(x) < 10:
                ax.set_title("[OII] doublet")
                ax.text(0.03, 0.97, "insufficient data", transform=ax.transAxes,
                        va='top', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
                continue

            A_init = np.max(y) - np.median(y)
            bounds = (
                [0, 0, 3726.03 - 3, 0.5, -np.inf, -np.inf],
                [np.inf, np.inf, 3726.03 + 3, 5.0, np.inf, np.inf]
            )
            try:
                popt, pcov = curve_fit(
                    double_gaussian_OII, x, y,
                    p0=[A_init, A_init, 3726.03, 2.0, np.median(y), 0.0],
                    sigma=yerr, absolute_sigma=True, bounds=bounds, maxfev=20000
                )

                # If the amplitude ratio lands outside the physical [0.2, 1.6]
                # range, refit from a typical ratio of 0.9
                ratio = popt[1] / popt[0] if popt[0] > 0 else np.nan
                if ratio < 0.2 or ratio > 1.6:
                    popt, pcov = curve_fit(
                        double_gaussian_OII, x, y,
                        p0=[A_init, 0.9 * A_init, 3726.03, 2.0, np.median(y), 0.0],
                        sigma=yerr, absolute_sigma=True, bounds=bounds, maxfev=20000
                    )
                    ratio = popt[1] / popt[0] if popt[0] > 0 else np.nan

                fwhm = 2.355 * popt[3]
                model = double_gaussian_OII(x, *popt)
                chi2_red = np.sum(((y - model) / yerr) ** 2) / (len(x) - 6)

                flux_3727, err_3727 = flux_and_error_from_cov(popt, pcov, 0, 3)
                flux_3729, err_3729 = flux_and_error_from_cov(popt, pcov, 1, 3)
                flux_total = flux_3727 + flux_3729
                err_total = np.sqrt(err_3727**2 + err_3729**2)

                snr_3727 = flux_3727 / err_3727 if err_3727 > 0 else 0
                snr_3729 = flux_3729 / err_3729 if err_3729 > 0 else 0
                snr_total = flux_total / err_total if err_total > 0 else 0

                flux_upper = upper_limit_flux(yerr, dlambda, fwhm)

                ax.plot(x, y, label="data")
                ax.plot(x, yerr, alpha=0.4)
                ax.plot(x, model, label="fit")
                ax.legend(fontsize=7)

                if snr_3727 >= SNR_DETECTION or snr_3729 >= SNR_DETECTION:
                    label = (f"3727={flux_3727:.2e}±{err_3727:.2e}\n"
                             f"3729={flux_3729:.2e}±{err_3729:.2e}\n"
                             f"ratio={ratio:.2f}\nχ²={chi2_red:.2f}")
                else:
                    label = (f"<OII 3727 {flux_upper:.2e}\n"
                             f"<OII 3729 {flux_upper:.2e}\n"
                             f"SNR_3727={snr_3727:.1f}, SNR_3729={snr_3729:.1f}")

                is_upper_3727 = snr_3727 < SNR_DETECTION
                is_upper_3729 = snr_3729 < SNR_DETECTION
                upper_total = is_upper_3727 or is_upper_3729

                results.append({
                    "objid": objid, "region": region, "line": "OII 3727",
                    "flux": (flux_3727 if snr_3727 >= SNR_DETECTION else flux_upper) * n_spx,
                    "flux_err": (err_3727 if snr_3727 >= SNR_DETECTION else flux_upper / 3.0) * n_spx,
                    "snr": snr_3727, "chi2_red": chi2_red, "upper_limit": is_upper_3727,
                })
                results.append({
                    "objid": objid, "region": region, "line": "OII 3729",
                    "flux": (flux_3729 if snr_3729 >= SNR_DETECTION else flux_upper) * n_spx,
                    "flux_err": (err_3729 if snr_3729 >= SNR_DETECTION else flux_upper / 3.0) * n_spx,
                    "snr": snr_3729, "chi2_red": chi2_red, "upper_limit": is_upper_3729,
                })
                results.append({
                    "objid": objid, "region": region, "line": "OII total",
                    "flux": (flux_total if not upper_total else flux_upper * 2) * n_spx,
                    "flux_err": (err_total if not upper_total else (flux_upper * 2) / 3.0) * n_spx,
                    "snr": snr_total, "chi2_red": chi2_red, "upper_limit": upper_total,
                })
            except Exception:
                label = "fit fail"

            ax.set_title("[OII] doublet")
            ax.text(0.03, 0.97, label, transform=ax.transAxes, va='top',
                    fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
            continue

        # ---- [SII] doublet -----------------------------------------------
        if name == "SII 6716":
            mask = (wave > 6700) & (wave < 6750)
            x, y, yerr = wave[mask], flux[mask], noise[mask]
            bounds = (
                [0, 0, 6716 - 5, 6731 - 5, 0.5, -np.inf],
                [np.inf, np.inf, 6716 + 5, 6731 + 5, 10, np.inf]
            )
            try:
                popt, pcov = curve_fit(
                    double_gaussian_SII, x, y,
                    p0=[np.max(y), np.max(y), 6716, 6731, 2.0, np.median(y)],
                    sigma=yerr, absolute_sigma=True
                )

                model = double_gaussian_SII(x, *popt)
                chi2_red = np.sum(((y - model) / yerr) ** 2) / (len(x) - 5)

                f1, e1 = flux_and_error_from_cov(popt, pcov, 0, 4)
                f2, e2 = flux_and_error_from_cov(popt, pcov, 1, 4)

                snr_f1 = f1 / e1 if e1 > 0 else 0
                snr_f2 = f2 / e2 if e2 > 0 else 0

                fwhm = 2.355 * popt[4]
                flux_upper = upper_limit_flux(yerr, dlambda, fwhm)

                ax.plot(x, y)
                ax.plot(x, model)
                ax.plot(x, yerr, alpha=0.4)

                if snr_f1 >= SNR_DETECTION or snr_f2 >= SNR_DETECTION:
                    label = (f"6716={f1:.2e}±{e1:.2e}\n6731={f2:.2e}±{e2:.2e}\n"
                             f"χ²={chi2_red:.2f}\n"
                             f"SNR_6716={snr_f1:.1f}, SNR_6731={snr_f2:.1f}")
                else:
                    label = (f"<SII 6716 {flux_upper:.2e}\n<SII 6731 {flux_upper:.2e}\n"
                             f"SNR_6716={snr_f1:.1f}, SNR_6731={snr_f2:.1f}")

                is_upper_6716 = snr_f1 < SNR_DETECTION
                is_upper_6731 = snr_f2 < SNR_DETECTION

                results.append({
                    "objid": objid, "region": region, "line": "SII 6716",
                    "flux": (f1 if snr_f1 >= SNR_DETECTION else flux_upper) * n_spx,
                    "flux_err": (e1 if snr_f1 >= SNR_DETECTION else flux_upper / 3.0) * n_spx,
                    "snr": snr_f1, "chi2_red": chi2_red, "upper_limit": is_upper_6716,
                })
                results.append({
                    "objid": objid, "region": region, "line": "SII 6731",
                    "flux": (f2 if snr_f2 >= SNR_DETECTION else flux_upper) * n_spx,
                    "flux_err": (e2 if snr_f2 >= SNR_DETECTION else flux_upper / 3.0) * n_spx,
                    "snr": snr_f2, "chi2_red": chi2_red, "upper_limit": is_upper_6731,
                })
            except Exception:
                label = "fit fail"

            ax.set_title("SII doublet")
            ax.text(0.03, 0.97, label, transform=ax.transAxes, va='top',
                    fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
            continue

        # 6731 is measured jointly with 6716 above
        if name == "SII 6731":
            continue

        # ---- Isolated single lines ---------------------------------------
        window = 15 if name == "NeIII 3869" else 20
        flux_val, flux_err, popt, chi2_red, x, y, model = fit_gaussian_line(
            wave, flux, noise, center, window=window
        )

        if popt is not None and np.isfinite(flux_val):
            snr = flux_val / flux_err if flux_err > 0 else 0
            win = noise[(wave > center - window) & (wave < center + window)]

            ax.plot(x, y)
            ax.plot(x, model)
            ax.plot(x, win, alpha=0.4)

            if snr >= SNR_DETECTION:
                label = f"F={flux_val:.2e}±{flux_err:.2e}\nχ²={chi2_red:.2f}\nSNR={snr:.1f}"
                flux_upper = np.nan
            else:
                fwhm = 2.355 * popt[2]
                flux_upper = upper_limit_flux(win, dlambda, fwhm)
                label = f"< {flux_upper:.2e}"

            is_upper = snr < SNR_DETECTION
            results.append({
                "objid": objid, "region": region, "line": name,
                "flux": (flux_val if snr >= SNR_DETECTION else flux_upper) * n_spx,
                "flux_err": (flux_err if snr >= SNR_DETECTION else flux_upper / 3.0) * n_spx,
                "snr": snr, "chi2_red": chi2_red, "upper_limit": is_upper,
            })
        else:
            label = "fit fail"

        ax.set_title(name)
        ax.text(0.03, 0.97, label, transform=ax.transAxes, va='top',
                fontsize=8, bbox=dict(facecolor='white', alpha=0.7))

    fig.suptitle(f"{objid} — {region} (Linear Stack)", fontsize=14)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    fits_files = sorted(
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.endswith(".fits.gz")
    )
    if not fits_files:
        raise FileNotFoundError(
            f"No .fits.gz files in {INPUT_DIR!r}. Run galaxy_stacking.py first."
        )

    print(f"Fitting {len(fits_files)} galaxies x {len(APERTURES)} apertures...")

    all_fit_results = []
    with PdfPages(DIAGNOSTICS_PDF) as pdf:
        for f in fits_files:
            for region in APERTURES:
                process_file(f, region, pdf, all_fit_results)

    pd.DataFrame(all_fit_results).to_csv(RESULTS_CSV, index=False)
    print(f"Saved {RESULTS_CSV} and {DIAGNOSTICS_PDF}")
    print("DONE")


if __name__ == '__main__':
    main()
