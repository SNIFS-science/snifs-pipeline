import multiprocessing
import time
from functools import partial
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from pipeline.common.fitting_math__utils import double_gaussian, gaussian
from pipeline.common.log import get_logger
from pipeline.common.plotting_utils import find_closest_index, get_wavelengths_to_fit
from pipeline.flows.preprocess_exposure import PreprocessSummary
from pipeline.tasks.processing.make_parameter_matrix import repeat_shift_fit, shifting_spaxel
from pipeline.tasks.processing.ram_tracker import MemoryMonitor
from pipeline.tasks.processing.refine_peak_centers_quickly import double_gaussian_jac, gaussian_jac
from pipeline.tasks.processing.shift_optimization import shift_and_save

# TIME STRUCTURE cpus: 2,4,6,8,10,12
# index structure : [0:5]
# data structure : (total_time_list #list of 10 runs#, parallel time list #list of 10 runs#, RAM list #list of 10 runs)
# TOTAL_ITER_DOUBLE = []

# TOTAL_ITER_SINGLE = []
# REFINE_RUN_TIME = 0
# REFINE_ITERS = 0

# PARALLEL_TIMER[cpu run index][0-9 for repeats on same setup] = CURRENT_LAP
PARALLEL_TIMER = []
MONITOR = None
# CURRENT_LAP[0].append(total_time)
# CURRENT_LAP[1].append(para_time)
# CURRENT_LAP[2].append(monitor.get_history())
# CURRENT_LAP[3].append(monitor.get_peak_mem())
# CURRENT_LAP[4].append(monitor_parallel_start_ind)
CURRENT_LAP = [[], [], [], [], []]

# UNIMPROVED_TIME = []

# IMPROVED_TIME = []

PROCESSING = 8


PEAKS_DICT = get_wavelengths_to_fit()

NUMBER_OF_SPAXELS = 225

ALL_PEAKS = PEAKS_DICT.keys()


def make_flux_array(linespread_path: Path, arc_vector_file: Path) -> np.ndarray:
    """Creates a flux array by convolving the linespread function with the
    model-generated spectrum data.

    Args:
        linespread_path : Path to the linespread file.
        arc_vector_file : Path to the arc vector file.

    Returns:
        np.ndarray: The flux array.
    """  # noqa: D205
    big_arr = []
    # TODO: should check that the loaded file is the size we expect it to be otherwise will have problems
    linespread_data = np.load(linespread_path)  # load_images_from_file(linespread_path)[0].data
    print(f"Linespread shape: {linespread_data.shape}")
    spectra = linespread_data.reshape(NUMBER_OF_SPAXELS, -1)

    print(f"Spectra shape: {spectra.shape}")

    spectrum_data = np.load(arc_vector_file)  # load_images_from_file(arc_vector_file)[0].data
    for i in range(0, NUMBER_OF_SPAXELS):
        avg_cross = np.mean(spectrum_data[1400 * i : 1400 * i + 1], axis=0)
        spectrum = np.convolve(avg_cross, spectra[i])
        big_arr.append(spectrum)
    return np.array(big_arr)


# double_range isn't the most robust way to do this. I think I should modify it so I can input a list of flags
# the same length as centers that tells you what peak(s) should be treated as doubles
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
                popt, pcov, infodict, errmsg, ier = curve_fit(
                    double_gaussian,
                    x_fit,
                    y_fit,
                    p0=parameter_guess,
                    bounds=(lower, upper),
                    maxfev=20000,
                    full_output=True,  # , jac=double_gaussian_jac
                )

                # TOTAL_ITER_DOUBLE.append(infodict['nfev'])

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
            # modified single peak fitting

            # ORIGINALLL
            amp_0 = max(spectrum[int(round(c))] - offset_0, 1e-6)
            mu_0 = float(c)
            sigma_0 = max(window / 2.0, 1e-3)

            p_0_single = [amp_0, mu_0, sigma_0, offset_0]
            lower_single = [0.0, max(c - window, 0), 1e-6, -np.inf]
            upper_single = [np.inf, min(c + window, len(spectrum) - 1), np.inf, np.inf]

            p_0_single = [float(np.clip(p_0_single[i], lower_single[i], upper_single[i])) for i in range(4)]
            # print("Initial peak guess")
            # print(p_0_single)
            try:
                popt, pcov, infodict, errmsg, ier = curve_fit(
                    gaussian,
                    x_fit,
                    y_fit,
                    p0=p_0_single,
                    bounds=(lower_single, upper_single),
                    maxfev=20000,
                    full_output=True,  # , jac=gaussian_jac
                )

                # TOTAL_ITER_SINGLE.append(infodict['nfev'])

                amp, mu, sigma, offset = popt
                # print("New Params:")
                # print([*popt])
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


def refine_peak_centers_modified(
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
                popt, pcov, infodict, errmsg, ier = curve_fit(
                    double_gaussian,
                    x_fit,
                    y_fit,
                    p0=parameter_guess,
                    bounds=(lower, upper),
                    maxfev=20000,
                    full_output=True,
                    jac=double_gaussian_jac,
                )

                # TOTAL_ITER_DOUBLE.append(infodict['nfev'])

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
            # modified single peak fitting

            # Zoom in on the region around the spectral peak
            left = max(int(c - window), 0)
            right = min(int(c + window) + 1, len(spectrum))

            local_x = np.arange(left, right)
            local_y = spectrum[left:right]

            # get the offset from the local y values
            offset_0 = np.median(local_y)

            # subtract and regularize the offset (note there could be a problem with 0 as lower bound)
            signal = np.clip(local_y - offset_0, 0, None)

            # Prevent divide by zero errors
            if np.sum(signal) > 0:
                # Use the first moment of the local range to estimate center (first moment is like center of mass)
                mu_0 = np.sum(local_x * signal) / np.sum(signal)

                # use variance formula for the local region to estimate true variance
                sigma_0 = np.sqrt(np.sum(signal * (local_x - mu_0) ** 2) / np.sum(signal))
                # naive guess for amplitude
                amp_0 = np.max(signal)

            else:
                mu_0 = float(c)
                sigma_0 = window / 4
                amp_0 = 1e-6

            pix_width = 1.0

            sigma_min = 0.3 * pix_width
            sigma_max = window

            p_0_single = [amp_0, mu_0, sigma_0, offset_0]

            lower_single = [
                0.0,
                max(c - window, 0),
                sigma_min,
                -np.inf,
            ]

            upper_single = [
                np.inf,
                min(c + window, len(spectrum) - 1),
                sigma_max,
                np.inf,
            ]

            p_0_single = [float(np.clip(p_0_single[i], lower_single[i], upper_single[i])) for i in range(4)]

            try:
                popt, pcov, infodict, errmsg, ier = curve_fit(
                    gaussian,
                    x_fit,
                    y_fit,
                    p0=p_0_single,
                    bounds=(lower_single, upper_single),
                    maxfev=20000,
                    full_output=True,
                    jac=gaussian_jac,
                )

                # TOTAL_ITER_SINGLE.append(infodict['nfev'])

                amp, mu, sigma, offset = popt
                # print("New Params:")
                # print([*popt])
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
    """Args:
        spectrum: The spectrum to calibrate.
        peaks_dict: Dictionary of peak positions with wavelengths as the keys and values providing location info.

    Returns:
        np.ndarray: The calibrated wavelengths.
        np.ndarray: The polynomial coefficients.
        np.ndarray: The residuals squared.
    """  # noqa: D205
    # TODO: add a lot of robustness checks here

    improved_peaks = []
    wavelengths = []

    for peak in peaks_dict.keys():
        # print(type(peaks_dict[peak]))
        if peaks_dict[peak]["first_fit"]:  # if peaks_dict[peak].first_fit == True:
            a, b = peaks_dict[peak]["pixel_start_search"], peaks_dict[peak]["pixel_end_search"]
            improved_peaks.append(a + np.nanargmax(spectrum[a:b]))
            wavelengths.append(float(peak))  # peaks_dict[peak].wavelength

    other_new_centers, _ = refine_peak_centers_modified(spectrum, improved_peaks, window=3)

    # TEST
    # plot_refined_spectrum(spectrum, other_new_centers)
    x_points = np.array(range(len(spectrum)))
    wavelengths_array = np.array(wavelengths)
    p_3 = np.polyfit(other_new_centers, wavelengths_array, 3)
    wavelengths_cubic_fit = p_3[0] * x_points**3 + p_3[1] * x_points**2 + p_3[2] * x_points + p_3[3]

    fitted_centers_lbda = np.polyval(p_3, other_new_centers)
    residuals = fitted_centers_lbda - wavelengths_array
    return wavelengths_cubic_fit, p_3, residuals**2


# timer code
# start = time.perf_counter()
# for i in range(10):
#     poo, p = refine_peak_centers(spectrum, improved_peaks, window=3)
# end = time.perf_counter()

# run_time_unimproved = end - start

# start = time.perf_counter()
# for i in range(10):
#     poo, p = refine_peak_centers_modified(spectrum, improved_peaks, window=3)
# end = time.perf_counter()

# run_time_improved = end - start

# print("IMPROVED RUNTIME: ")
# print(run_time_improved)
# print("UNIMPROVED RUNTIME: ")
# print(run_time_unimproved)
# UNIMPROVED_TIME.append(run_time_unimproved/10)
# IMPROVED_TIME.append(run_time_improved/10)


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
    other_new_centers, _ = refine_peak_centers_modified(spectrum, peak_guesses, window=3)
    x_points = np.array(range(len(spectrum)))

    other_lbda = np.array(list(map(float, corresponding_wavelengths)), dtype=float)
    # print(other_lbda)
    p_3 = np.polyfit(other_new_centers, other_lbda, 3)
    wavelengths_3 = p_3[0] * x_points**3 + p_3[1] * x_points**2 + p_3[2] * x_points + p_3[3]

    fitted_centers_lbda = np.polyval(p_3, other_new_centers)
    residuals = fitted_centers_lbda - other_lbda
    return wavelengths_3, p_3, residuals**2


# TRY THIS TO UNDERSTAND TRY TO UNDERSTAND THIS
def calibrate_wavelength_arc(arcPath: Path) -> np.ndarray:
    """Args:
        arcPath: Path to the arc file to be calibrated.

    Returns:
        np.ndarray: The calibrated wavelength parameters.
    """  # noqa: D205
    logger = get_logger()
    logger.info(f"Starting wavelength calibration for arc file: {arcPath}")

    print(lineSpreadPath, arcVectorPath)
    flux_array = make_flux_array(lineSpreadPath, arcVectorPath)
    wavelength_list = []
    params = []
    residuals = []

    spaxels = [flux_array[i] for i in range(NUMBER_OF_SPAXELS)]
    cal_spec_with_dict = partial(
        cal_spec, peaks_dict=PEAKS_DICT
    )  # why does cal_spec take in peaks dict if it stays the same

    # I could do
    # cores_to_use = multiprocessing.cpu_count() - 1
    # pool = multiprocessing.Pool(processes=cores_to_use)

    print("starting parallel processes")
    monitor_parallel_start_ind = MONITOR.get_history_index()
    parallel_start = time.perf_counter()
    with multiprocessing.Pool(processes=PROCESSING) as pool:
        results = pool.map(cal_spec_with_dict, spaxels)
    wavelength_list = [item[0] for item in results]
    params = [item[1] for item in results]
    residuals = [item[2] for item in results]
    logger.info("early RMS: ", np.sqrt(np.mean(residuals)))
    logger.info("beginning refined fitting")
    residuals = []

    monitor_parallel_end_ind = MONITOR.get_history_index()
    parallel_done = time.perf_counter()

    print(f"Successfully processed {len(results)} items.")

    para_time = parallel_done - parallel_start

    CURRENT_LAP[1].append(para_time)

    for i in range(flux_array.shape[0]):
        # print(flux_array.shape[0], len(wavelength_list))

        spec = flux_array[i]
        # figure out where we think the peaks are based on the previous fit, then refine them
        closest_indices = [find_closest_index(wavelength_list[i], p) for p in ALL_PEAKS]
        waves, ps, res = recal_spec(spec, closest_indices, ALL_PEAKS)
        wavelength_list[i] = waves
        params[i] = ps
        residuals.extend(res)

    params = np.array(params)
    # plot_params(params)  # THIS ONE
    logger.info("late RMS: ", np.sqrt(np.mean(residuals)))
    # plot_spectrum(np.array(wavelength_list), flux_array) #THIS ONE SKIP FOR NOW
    # TODO: decide what we want to save and/or return
    # np.save(f"{name}fit_wavelength_cal", big_wave)

    logger.info(f"done with arc fits for {p.stem}")
    MONITOR.stop()
    MONITOR.join()
    CURRENT_LAP[2].append(MONITOR.get_history())
    CURRENT_LAP[3].append(MONITOR.get_peak_mem())
    CURRENT_LAP[4].append((monitor_parallel_start_ind, monitor_parallel_end_ind))

    return params


# TODO: the locations of the output parameter files
# TODO: the spectra IGNORE THIS FOR NOW
def full_arc_calibration(preprocessed_arc: PreprocessSummary) -> None:
    preprocessed_arc_path = Path(preprocessed_arc.output_path)
    spaxel_list = list(range(0, 224))
    # do all the translational shifts first
    repeat_shift_fit(spaxel_list, np.arange(-1.5, 1.5, 0.3).tolist(), True, preprocessed_arc_path, np.zeros(1))
    shift_and_save(spaxel_list, is_translational_shift=True)
    # using ideal translational shifts, do the width shifts
    translation_array = np.load(f"translational_shifts_{preprocessed_arc.run_id}.npy")
    repeat_shift_fit(
        spaxel_list,
        np.arange(0.9, 1.2, 0.05).tolist(),
        False,
        preprocessed_arc_path,
        np.zeros(1),
        translational_params=translation_array,
    )
    shift_and_save(spaxel_list, is_translational_shift=False)
    # generate the full matrix and vector using oversampling
    width_array = np.load(f"width_shifts_{preprocessed_arc.run_id}.npy")
    shifting_spaxel(
        0,
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
    # START TRACKING TOTAL RAM
    # CHANGE PATHS HERE
    base_dir = Path("/home/anousha/snifs-pipeline/data/level=raw")

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

        # config = PreprocessExposureConfig(primary_file=file)
        # calibrate_wavelength_arc(file)

        # timer code
        cpu_test_range = 6
        PARALLEL_TIMER = [[] for k in range(cpu_test_range)]
        for j in range(cpu_test_range):
            PROCESSING = 2 + j * 2

            for i in range(10):
                print(f"COMMENCING RUN {i + 1}")
                MONITOR = MemoryMonitor()
                MONITOR.start()
                start = time.perf_counter()
                calibrate_wavelength_arc(file)
                end = time.perf_counter()
                print(f"TIME FOR RUN {i + 1} is {end - start}")
                CURRENT_LAP[0].append(end - start)
            PARALLEL_TIMER[j].append(CURRENT_LAP)

            CURRENT_LAP = [[], [], [], [], []]
        # for a cpu run
# SAVE PATH FOR NPY FILE
# CHANGE PATHS HERE
np.save(
    "/home/anousha/snifs-pipeline/parallel_time_ram_spaxels_corrected_peak_RAM.npy",
    np.array(PARALLEL_TIMER, dtype=object),
)
