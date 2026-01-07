from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from pipeline.common.log import get_logger
from pipeline.common.plotting_utils import double_gaussian, find_closest_index, gaussian, get_wavelengths_to_fit
from pipeline.tasks.loaders import load_images_from_file
from pipeline.tasks.plotting.wavelength_arc_calibration_plots import plot_params, plot_refined_spectrum, plot_spectrum

PEAKS_DICT = get_wavelengths_to_fit()

NUMBER_OF_SPAXELS = 225

ALL_PEAKS = PEAKS_DICT.keys()


def make_flux_array(linespread_path: Path, arc_vector_file: Path) -> np.ndarray:
    """
    Creates a flux array by convolving the linespread function with the
    model-generated spectrum data.
    Args:
        linespread_path : Path to the linespread file.
        arc_vector_file : Path to the arc vector file.
    Returns:
        np.ndarray: The flux array.
    """
    big_arr = []
    # TODO: should check that the loaded file is the size we expect it to be otherwise will have problems
    linespread_file = load_images_from_file(linespread_path)[0].data
    spectra = linespread_file.reshape(NUMBER_OF_SPAXELS, -1)

    spectrum_file = load_images_from_file(arc_vector_file)[0].data
    for i in range(0, NUMBER_OF_SPAXELS):
        avg_cross = np.mean(spectrum_file[1400 * i : 1400 * i + 1], axis=0)
        spectrum = np.convolve(avg_cross, spectra[i])
        big_arr.append(spectrum)
    return np.array(big_arr)


# double_range isn't the most robust way to do this. I think I should modify it so I can input a list of flags
# the same length as centers that tells you what peak(s) should be treated as doubles
def refine_peak_centers(
    spectrum: np.ndarray, centers: list, window: int = 10, double_range: tuple = (300, 400)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        spectrum: The partially fit spectrum to refine by fitting Gaussians around emission lines.
        centers: The initial np.argnanmax nearest each emission line.
        window: The window size for the Gaussians.
        double_range: The range of peaks to treat as double lines.
    Returns:
        np.ndarray: The refined peak centers.
    """
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
            mu1_0 = c - window / 4.0
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


def cal_spec(spectrum: np.ndarray, peaks_dict: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        spectrum: The spectrum to calibrate.
        peaks_dict: Dictionary of peak positions with wavelengths as the keys and values providing location info.
    Returns:
        np.ndarray: The calibrated wavelengths.
        np.ndarray: The polynomial coefficients.
        np.ndarray: The residuals squared.
    """
    # TODO: add a lot of robustness checks here

    improved_peaks = []
    wavelengths = []
    for peak in peaks_dict.keys():
        if peaks_dict[peak].first_fit is True:
            a, b = peaks_dict[peak].pixel_start_search, peaks_dict[peak].pixel_end_search
            improved_peaks.append(a + np.nanargmax(spectrum[a:b]))
            wavelengths.append(peaks_dict[peak].wavelength)
    other_new_centers, p = refine_peak_centers(spectrum, improved_peaks, window=3)
    plot_refined_spectrum(spectrum, other_new_centers)
    x_points = np.array(range(len(spectrum)))
    wavelengths_array = np.array(wavelengths)
    p_3 = np.polyfit(other_new_centers, wavelengths_array, 3)
    wavelengths_cubic_fit = p_3[0] * x_points**3 + p_3[1] * x_points**2 + p_3[2] * x_points + p_3[3]

    fitted_centers_lbda = np.polyval(p_3, other_new_centers)
    residuals = fitted_centers_lbda - wavelengths_array
    return wavelengths_cubic_fit, p_3, residuals**2


def recal_spec(spectrum, peak_guesses, corresponding_wavelengths) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # this is the part that takes the longest time
    other_new_centers, p = refine_peak_centers(spectrum, peak_guesses, window=3)
    x_points = np.array(range(len(spectrum)))
    other_lbda = np.array(corresponding_wavelengths)
    p_3 = np.polyfit(other_new_centers, other_lbda, 3)
    wavelengths_3 = p_3[0] * x_points**3 + p_3[1] * x_points**2 + p_3[2] * x_points + p_3[3]

    fitted_centers_lbda = np.polyval(p_3, other_new_centers)
    residuals = fitted_centers_lbda - other_lbda
    return wavelengths_3, p_3, residuals**2


def calibrate_wavelength_arc(arcPath: Path) -> np.ndarray:
    """
    Args:
        arcPath: Path to the arc file to be calibrated.
    Returns:
        np.ndarray: The calibrated wavelength parameters.
    """
    logger = get_logger()
    logger.info(f"Starting wavelength calibration for arc file: {arcPath}")

    big_arr = make_flux_array(lineSpreadPath, arcVectorPath)
    big_wave = []
    params = []
    residuals = []

    for i in range(NUMBER_OF_SPAXELS):
        spec = big_arr[i]
        waves, ps, res = cal_spec(spec, PEAKS_DICT)
        big_wave.append(waves)
        params.append(ps)
        residuals.extend(res)
        logger.info("early RMS: ", np.sqrt(np.mean(residuals)))
        logger.info("beginning refined fitting")
        residuals = []
        for i in range(big_arr.shape[0]):
            spec = big_arr[i]
            # figure out where we think the peaks are based on the previous fit, then refine them
            closest_indices = [find_closest_index(big_wave[i], p) for p in ALL_PEAKS]
            waves, ps, res = recal_spec(spec, closest_indices, ALL_PEAKS)
            big_wave[i] = waves
            params[i] = ps
            residuals.extend(res)
    params = np.array(params)
    plot_params(params)
    logger.info("late RMS: ", np.sqrt(np.mean(residuals)))
    plot_spectrum(np.array(big_wave), big_arr)
    # TODO: decide what we want to save and/or return
    # np.save(f"{name}fit_wavelength_cal", big_wave)
    logger.info(f"done with arc fits for {p.stem}")
    return params


# TODO: make the code in here a top level function (flow) with the entrypoint just calling that function
# TODO: this top level function should take a pydantic configuration object so it can easily integrate with prefect
if __name__ == "__main__":
    base_dir = Path("/src/pipeline/output/")
    files = [
        base_dir / "runs/run_id=25_056_084/science_red.fits",  # TODO : update file name to actual arc file
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

        # config = PreprocessExposureConfig(primary_file=file)
        calibrate_wavelength_arc(file)
