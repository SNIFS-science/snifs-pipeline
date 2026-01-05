import numpy as np
from scipy.optimize import curve_fit

from pipeline.common.plotting_utils import double_gaussian, find_closest_index, gaussian, get_all_peaks
from pipeline.tasks.plotting.wavelength_arc_calibration_plots import plot_params, plot_refined_spectrum, plot_spectrum

ALL_PEAKS = get_all_peaks()

INITIAL_PEAK_ESTIMATES = np.array(
    [
        [300, 390],
        [400, 550],
        [580, 640],
        [641, 705],
        [705, 770],
        [880, 920],
        [1000, 1100],
        [1150, 1250],
        [1400, 1448],
    ]
)

INITIAL_WAVELENGTH_VALUES = np.array([5769.6, 5460.735, 5085.822, 4916, 4799.912, 4358.1, 4045.3, 3651.3, 3131.7])

SPAXEL_NUMBER = 225


# TODO: all of these should be updated to have type hints np.ndarray
# TODO: we should put some basic docstring for all common functions. googledoc style preferred.
# TODO: make sure that the path is a path (not a string)
def make_flux_array(linespread_file, spectrum_file):
    """
    Creates a flux array by convolving the linespread function with the
    model-generated spectrum data.
    Args:
        linespread_file : Linespread file.
        spectrum_file : Arc vector file.
    Returns:
        np.ndarray: The flux array.
    """
    big_arr = []
    # TODO: should check that the loaded file is the size we expect it to be otherwise will have problems
    spectra = linespread_file.reshape(SPAXEL_NUMBER, -1)
    for i in range(0, SPAXEL_NUMBER):
        avg_cross = np.mean(spectrum_file[1400 * i : 1400 * i + 1], axis=0)
        spectrum = np.convolve(avg_cross, spectra[i])
        big_arr.append(spectrum)
    return np.array(big_arr)


# double_range isn't the most robust way to do this. I think I should modify it so I can input a list of flags
# the same length as centers that tells you what peak(s) should be treated as doubles
def refine_peak_centers(spectrum: np.ndarray, centers: list, window: int = 10, double_range: tuple = (300, 400)):
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

            # Ensure p_0 is feasible (clip center into [lower,upper] etc.)
            # TODO: this makes Sam sad.
            # p_0_clipped = [
            #     float(np.clip(p, l, u) if np.isfinite(u) else 1e12) for p, l, u in zip(p_0, lower, upper, strict=True)
            # ]
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

    return np.array(new_centers), fit_params


def cal_spec(spectrum: np.ndarray, est_peaks: np.ndarray, wavelengths: np.ndarray) -> tuple:
    # need to add a lot of robustness checks here

    other_peaks = []
    for peak in est_peaks:
        a, b = peak
        other_peaks.append(a + np.nanargmax(spec[a:b]))
    other_new_centers, p = refine_peak_centers(spectrum, other_peaks, window=3)
    plot_refined_spectrum(spectrum, other_new_centers)
    x_points = np.array(range(len(spectrum)))
    other_lbda = np.array(wavelengths)
    p_3 = np.polyfit(other_new_centers, other_lbda, 3)
    wavelengths_cubic_fit = p_3[0] * x_points**3 + p_3[1] * x_points**2 + p_3[2] * x_points + p_3[3]

    fitted_centers_lbda = np.polyval(p_3, other_new_centers)
    residuals = fitted_centers_lbda - other_lbda
    return wavelengths_cubic_fit, p_3, residuals**2


def recal_spec(spec, peaks, lbda):
    # this is the part that takes the longest time
    other_new_centers, p = refine_peak_centers(spec, peaks, window=3)
    x_points = np.array(range(len(spec)))
    other_lbda = np.array(lbda)
    p_3 = np.polyfit(other_new_centers, other_lbda, 3)
    wavelengths_3 = p_3[0] * x_points**3 + p_3[1] * x_points**2 + p_3[2] * x_points + p_3[3]

    fitted_centers_lbda = np.polyval(p_3, other_new_centers)
    residuals = fitted_centers_lbda - other_lbda
    return wavelengths_3, p_3, residuals**2


# TODO: make the code in here a top level function (flow) with the entrypoint just calling that function
# TODO: this top level function should take a pydantic configuration object so it can easily integrate with prefect
if __name__ == "__main__":
    # class WavelengthSearch(TypedDict):
    #     wavelength_anstroms: float
    #     pixel_start_search: int
    #     pixel_end_search: int

    # class WavelengthSearch2(BaseModel):
    #     wavelength_anstroms: float
    #     pixel_start_search: int
    #     pixel_end_search: int

    # @dataclass
    # class WavelengthConfig:
    #     wavelength_anstroms: float
    #     pixel_start_search: int
    #     pixel_end_search: int

    # wavelengths_to_fit: dict[str, WavelengthSearch2] = {
    #     "mercury_1": WavelengthSearch2(
    #         wavelength_anstroms=5769.6,
    #         pixel_start_search=300,
    #         pixel_end_search=390,
    #     )
    # }

    # I know you do this file selection stuff much better in preprocess_exposure
    # TODO: NEVER USE OS ITS DISGUSTING TO ME
    import os

    # TODO: we should have the main function (flow) take in a specific arc file path
    # TODO: and run over that - and we can loop through all the available arcs outside of the function
    # TODO: in the name==main entrypoint
    rootdir = "/home/anousha/snifs_model/"
    for _subdir, dirs, _files in os.walk(rootdir):
        for dire in dirs:
            name = str(dire)
            # TODO: boooooo for hardcoding this - we should try to stitch it into the preprocess output
            if "P25_" in name and "B" in name:
                print(f"wavelength calibrating {name}")
                # TODO: Where this file come from?
                # TODO: We should pull the code that generates this file into this "flow"
                big_arr = make_flux_array(rootdir + name + "_crossSumFitArc", rootdir + name + "_fit_arc_vector.npy")
                big_wave = []
                params = []
                residuals = []
                # TODO: what is the first dim representing? ah - each spaxel
                for i in range(big_arr.shape[0]):
                    spec = big_arr[i]
                    waves, ps, res = cal_spec(spec, INITIAL_PEAK_ESTIMATES, INITIAL_WAVELENGTH_VALUES)
                    big_wave.append(waves)
                    params.append(ps)
                    residuals.extend(res)
                print("early RMS: ", np.sqrt(np.mean(residuals)))
                big_wave = np.array(big_wave)
                print("refitting")
                residuals = []
                for i in range(big_arr.shape[0]):
                    spec = big_arr[i]
                    closest_indices = [find_closest_index(big_wave[i], p) for p in ALL_PEAKS]
                    waves, ps, res = recal_spec(spec, closest_indices, ALL_PEAKS)
                    big_wave[i] = waves
                    params[i] = ps
                    residuals.extend(res)
                plot_params(np.array(params))
                print("late RMS: ", np.sqrt(np.mean(residuals)))
                plot_spectrum(big_wave, big_arr)
                np.save(f"{name}fitWavelengthCal", big_wave)
                print(f"done with {name}")
