import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from pipeline import settings
from pipeline.common.fitting_math__utils import double_gaussian, gaussian
from pipeline.common.log import get_logger
from pipeline.common.plotting_utils import find_closest_index, get_wavelengths_to_fit
from pipeline.common.prefect_utils import pipeline_task
from pipeline.tasks.plotting.wavelength_arc_calibration_plots import plot_refined_spectrum, plot_spectrum

# from pipeline.tasks.processing.make_parameter_matrix import repeat_shift_fit, shifting_spaxel

PEAKS_DICT = get_wavelengths_to_fit()

NUMBER_OF_SPAXELS = 225

ALL_PEAKS = PEAKS_DICT.keys()


@pipeline_task()
def make_flux_array(linespread_path: Path, arc_vector_file: Path) -> np.ndarray:
    """Creates a flux array by convolving the linespread function with the
    model-generated spectrum data.

    Args:
        linespread_path : Path to the linespread file.
        arc_vector_file : Path to the arc vector file.

    Returns:
        np.ndarray: The flux array.
    """  # noqa: D205
    linespread_data = np.load(linespread_path)  # (n_spaxels * 1400, 100)
    spectrum_data = np.load(arc_vector_file)  # (n_spaxels * 1400,)
    n_spaxels = len(spectrum_data) // 1400
    if linespread_data.shape[0] != n_spaxels * 1400:
        raise ValueError(f"linespread has {linespread_data.shape[0]} rows but arc vector implies {n_spaxels * 1400}")

    big_arr = []
    for i in range(n_spaxels):
        arc = spectrum_data[1400 * i : 1400 * (i + 1)]
        kernel = np.nanmean(linespread_data[1400 * i : 1400 * (i + 1)], axis=0)
        big_arr.append(np.convolve(arc, kernel / kernel.sum(), mode="full"))
    return np.array(big_arr)


# double_range isn't the most robust way to do this. I think I should modify it so I can input a list of flags
# the same length as centers that tells you what peak(s) should be treated as doubles
@pipeline_task()
def refine_peak_centers(
    spectrum: np.ndarray, centers: list, window: int = 10, double_range: tuple = (300, 400)
) -> tuple[np.ndarray, np.ndarray]:
    """Args:
        spectrum: The partially fit spectrum to refine by fitting Gaussians around emission
          lines.
        centers: The initial np.argnanmax nearest each emission line.
        window: The window size for the Gaussians.
        double_range: The range of peaks to treat as double lines.

    Returns:
        np.ndarray: The refined peak centers.
    """  # noqa: D205
    x = np.arange(len(spectrum))
    new_centers = []
    fit_params = []

    for c in centers:
        # local fit indices
        i1 = max(int(np.floor(c - window)), 0)
        i2 = min(int(np.ceil(c + window + 1)), len(spectrum))
        x_fit = x[i1:i2]
        y_fit = spectrum[i1:i2]

        # common offset guess
        offset_0 = np.median(y_fit)

        if double_range[0] <= c <= double_range[1] and len(x_fit) >= 5:
            # Double Gaussian fit with safe/consistent p0 and bounds
            amp_guess = max(spectrum[int(round(c))] - offset_0, 1e-6)

            # initial two-peak positions relative to c
            mu1_0 = c - window / 4.0  # TODO: check that this matches the spacing between the doublet
            mu2_0 = c + window / 4.0

            # center/delta parameterization
            center_0 = 0.5 * (mu1_0 + mu2_0)
            delta_0 = mu2_0 - mu1_0

            sigma1_0 = sigma2_0 = max(window / 3.0, 1e-3)

            # ensure amplitudes non-negative and not tiny negative
            amp1_0 = max(amp_guess, 1e-6)
            amp2_0 = max(amp_guess, 1e-6)

            # Clip delta_0 into allowed range [1e-6, 4]
            delta_0 = float(np.clip(delta_0, 1e-6, 4.0))

            parameter_guess = [amp1_0, center_0, sigma1_0, amp2_0, delta_0, sigma2_0, offset_0]

            lower = [0.0, max(c - window, 0), 1e-6, 0.0, 1e-6, 1e-6, -np.inf]
            upper = [np.inf, min(c + window, len(spectrum) - 1), np.inf, np.inf, 4.0, np.inf, np.inf]

            try:
                popt, pcov = curve_fit(
                    double_gaussian, x_fit, y_fit, p0=parameter_guess, bounds=(lower, upper), maxfev=20000
                )
                amp1, center, sigma1, amp2, delta, sigma2, offset = popt
                mu1 = center - delta / 2.0
                mu2 = center + delta / 2.0

                new_centers.append(max(mu1, mu2))
                fit_params.append(
                    {
                        "amp1": float(amp1),
                        "mu1": float(mu1),
                        "sigma1": float(sigma1),
                        "amp2": float(amp2),
                        "mu2": float(mu2),
                        "sigma2": float(sigma2),
                        "offset": float(offset),
                        "type": "double",
                        "popt": popt,
                        "pcov": pcov,
                    }
                )

            except Exception:
                new_centers.append(c)
                fit_params.append(None)

        else:
            # normal single peak fitting
            amp_0 = max(spectrum[int(round(c))] - offset_0, 1e-6)
            mu_0 = float(c)
            sigma_0 = max(window / 2.0, 1e-3)

            p_0_single = [amp_0, mu_0, sigma_0, offset_0]
            lower_single = [0.0, max(c - window, 0), 1e-6, -np.inf]
            upper_single = [np.inf, min(c + window, len(spectrum) - 1), np.inf, np.inf]

            p_0_single = [float(np.clip(p_0_single[i], lower_single[i], upper_single[i])) for i in range(4)]

            try:
                popt, pcov = curve_fit(
                    gaussian, x_fit, y_fit, p0=p_0_single, bounds=(lower_single, upper_single), maxfev=20000
                )
                amp, mu, sigma, offset = popt
                new_centers.append(float(mu))
                fit_params.append(
                    {
                        "amp": float(amp),
                        "mu": float(mu),
                        "sigma": float(sigma),
                        "offset": float(offset),
                        "type": "single",
                        "popt": popt,
                        "pcov": pcov,
                    }
                )
            except Exception:
                new_centers.append(c)
                fit_params.append(None)

    return np.array(new_centers), np.array(fit_params)


@pipeline_task()
def cal_spec(spectrum: np.ndarray, peaks_dict: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Args:
        spectrum: The spectrum to calibrate.
        peaks_dict: Dictionary of peak positions with wavelengths as the keys and values providing location info.

    Returns:
        np.ndarray: The calibrated wavelengths.
        np.ndarray: The polynomial coefficients.
        np.ndarray: The residuals squared.
    """  # noqa: D205
    # TODO: add a lot of robustness checks here
    print("spectrum shape", spectrum.shape)

    improved_peaks = []
    wavelengths = []
    for peak in peaks_dict.keys():
        if peaks_dict[peak]["first_fit"]:
            a, b = peaks_dict[peak]["pixel_start_search"], peaks_dict[peak]["pixel_end_search"]
            improved_peaks.append(a + np.nanargmax(spectrum[a:b]))
            wavelengths.append(peak)
    other_new_centers, _ = refine_peak_centers(spectrum, improved_peaks, window=3)
    plot_refined_spectrum(spectrum, other_new_centers)
    x_points = np.array(range(len(spectrum)))
    wavelengths_array = np.array(wavelengths)
    p_3 = np.polyfit(other_new_centers, wavelengths_array, 3)
    wavelengths_cubic_fit = p_3[0] * x_points**3 + p_3[1] * x_points**2 + p_3[2] * x_points + p_3[3]

    fitted_centers_lbda = np.polyval(p_3, other_new_centers)
    residuals = fitted_centers_lbda - wavelengths_array
    return wavelengths_cubic_fit, p_3, residuals**2


@pipeline_task()
def recal_spec(spectrum, peak_guesses, corresponding_wavelengths) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Args:
        spectrum: The spectrum to recalibrate.
        peak_guesses: The guessed peak positions.
        corresponding_wavelengths: The corresponding wavelengths for the guessed peaks.

    Returns:
        np.ndarray: The recalibrated wavelengths.
        np.ndarray: The polynomial coefficients.
        np.ndarray: The residuals squared.
    """  # noqa: D205
    # this is the part that takes the longest time
    other_new_centers, _ = refine_peak_centers(spectrum, peak_guesses, window=3)
    x_points = np.array(range(len(spectrum)))
    other_lbda = np.array(corresponding_wavelengths)
    p_3 = np.polyfit(other_new_centers, other_lbda, 3)
    wavelengths_3 = p_3[0] * x_points**3 + p_3[1] * x_points**2 + p_3[2] * x_points + p_3[3]

    fitted_centers_lbda = np.polyval(p_3, other_new_centers)
    residuals = fitted_centers_lbda - other_lbda
    return wavelengths_3, p_3, residuals**2


def calibrate_wavelength_arc(
    arcVectorPath, lineSpreadPath, spaxels: list[int] | None = None, output_dir: Path | None = None
):
    """Fit a cubic pixel->wavelength solution for each spaxel in the arc vector.

    Args:
        arcVectorPath: .npy of the concatenated per-spaxel arc vectors (1400 per spaxel).
        lineSpreadPath: .npy of the matching per-row line-spread kernels.
        spaxels: spaxel ids, in the order they were written to the arc vector (default 0..n-1).
        output_dir: where to write the per-spaxel parameter JSONs (default settings.output_path).

    Returns:
        The rectified-spectrum plot image.
    """
    logger = get_logger()
    flux_array = make_flux_array(lineSpreadPath, arcVectorPath)
    spaxels = list(spaxels) if spaxels is not None else list(range(len(flux_array)))
    if len(spaxels) != len(flux_array):
        raise ValueError(f"{len(spaxels)} spaxel ids given but the arc vector holds {len(flux_array)} spaxels")
    output_dir = output_dir if output_dir is not None else settings.output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    wavelength_list = []
    residuals = []
    for idx, spaxel in enumerate(spaxels):
        spec = flux_array[idx]
        waves, _, res = cal_spec(spec, PEAKS_DICT)
        logger.info(f"spaxel {spaxel} early RMS: {np.sqrt(np.nanmean(res))}")
        # figure out where we think the peaks are based on the previous fit, then refine them
        closest_indices = [find_closest_index(waves, p) for p in ALL_PEAKS]
        waves, ps, res = recal_spec(spec, closest_indices, list(ALL_PEAKS))
        wavelength_list.append(waves)
        residuals.extend(res)
        logger.info(f"spaxel {spaxel} done")

        output_path = output_dir / f"wavelength_calibration_parameters_spaxel_{spaxel}.json"
        with output_path.open("w") as f:
            json.dump({"coefficient_order": [3, 2, 1, 0], str(spaxel): ps.tolist()}, f, indent=2)
        logger.info(f"Saved wavelength calibration parameters for spaxel {spaxel} to {output_path}")

    logger.info(f"late RMS: {np.sqrt(np.nanmean(residuals))}")
    return plot_spectrum(np.array(wavelength_list), flux_array)


"""
# TODO: the locations of the output parameter files
# TODO: the spectra
#IGNORE!! this is what should be in process.py
def full_arc_calibration(preprocessed_arc: PreprocessSummary) -> None:
    run_id = str(preprocessed_arc.run_id)
    # preprocessed_arc_path = Path(preprocessed_arc.output_path)
    spaxel_list = list(range(0, 224))
    # do all the translational shifts first
    repeat_shift_fit(spaxel_list, np.arange(-1.5, 1.5, 0.3).tolist(), True, preprocessed_arc, np.zeros(1))
    shift_and_save(spaxel_list, is_translational_shift=True)
    # using ideal translational shifts, do the width shifts
    translation_array = np.load(f"translational_shifts_{preprocessed_arc.run_id}.npy")
    repeat_shift_fit(
        spaxel_list,
        np.arange(0.9, 1.2, 0.05).tolist(),
        False,
        preprocessed_arc,
        np.zeros(1),
        translational_params=translation_array,
    )
    shift_and_save(spaxel_list, is_translational_shift=False)
    # generate the full matrix and vector using oversampling
    width_array = np.load(f"width_shifts_{preprocessed_arc.run_id}.npy")
    shifting_spaxel(
        0,
        run_id,
        0,
        True,
        oversample_factor=4,
        is_partial=False,
        translational_params=translation_array,
        width_params=width_array,
    )


# TODO: make the code in here a top level function (flow) with the entrypoint just calling that function
# TODO: this top level function should take a pydantic configuration object so it can easily integrate with prefect
if __name__ == "__main__":
    base_dir = Path("data/level=raw")
    files = [
        base_dir / "runs/run_id=25_199_028/25_199_028_004_03_B.fits",  # TODO : update file name to actual arc file
    ]
    for file in files:
        p = Path(file)
        name = p.stem + "_"
        # TODO: Where this file come from?
        # TODO: We should pull the code that generates this file into this "flow"
        lineSpreadPath = p.with_name(name + "line_spread.npy")
        arcVectorPath = p.with_name(name + "fit_arc_vector.npy")

        # check that all the necessary files exist
        assert Path(file).exists(), f"File {file} does not exist."
        assert lineSpreadPath.exists(), f"File {lineSpreadPath} does not exist."
        assert arcVectorPath.exists(), f"File {arcVectorPath} does not exist."

        #config = ProcessRunConfig(run_id=file)
        calibrate_wavelength_arc(lineSpreadPath, arcVectorPath)
        #in fits header, add path to calibration data
        #                add residuals per spaxel
        #                add number of peaks that successfully fit
        #                add chi2

        # TODO: next function will accept a path to a json file with all the calibration, science image (as fits rn) """
