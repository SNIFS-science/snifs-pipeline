import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

from pipeline import settings

PEAKS = [
    5769.6,
    5460.735,
    5085.822,
    4916,
    4358.328,
    4198.317,
    4158.59,
    4077.837,
    4046.563,
    3906.371,
    3663.279,
    3650.153,
    3610.5077,
    3466.1996,
    3261.0548,
    3131.7,
]

PEAK_ESTIMATES = np.array(
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

WAVELENGTH_VALUES = np.array([5769.6, 5460.735, 5085.822, 4916, 4799.912, 4358.1, 4045.3, 3651.3, 3131.7])


# TODO: We should move these common math functions into the common package, maybe in a math_utils.py file
# TODO: all of these should be updated to have type hints np.ndarray
# TODO: we should put some basic docstring for all common functions. googledoc style preferred.
def gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset


# TODO: same as the above
def double_gaussian(x, amp1, center, sigma1, amp2, delta, sigma2, offset):
    # defined this way to control distance between the two
    mu1 = center - delta / 2
    mu2 = center + delta / 2
    return amp1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2) + amp2 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2) + offset


# TODO: make sure that the path is a path (not a string)
def make_array(path, spectrum_path):
    # TODO: make_array name could be more informative
    big_arr = []
    file2 = spectrum_path
    a = np.load(file2)
    # should check that the loaded file is the size we expect it to be otherwise will have problems
    # TODO: should we pull out the magic number into a const?
    spectra = a.reshape(225, -1)
    # TODO: we should load this in as early as possible and pass the data in, rather than file
    file = f"{path}.npy"
    data_cross = np.load(file)
    for i in range(0, 225):
        avg_cross = np.mean(data_cross[1400 * i : 1400 * i + 1], axis=0)
        spectrum = np.convolve(avg_cross, spectra[i])
        big_arr.append(spectrum)
    return np.array(big_arr)


# double_range isn't the most robust way to do this. I think I should modify it so I can input a list of flags
# the same length as centers that tells you what peak(s) should be treated as doubles
def refine_peak_centers(spectrum, centers, window=10, double_range=(300, 400)):
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
            # --- Double Gaussian fit with safe/consistent p0 and bounds ---
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

            p_0 = [amp1_0, center_0, sigma1_0, amp2_0, delta_0, sigma2_0, offset_0]

            lower = [0.0, max(c - window, 0), 1e-6, 0.0, 1e-6, 1e-6, -np.inf]
            upper = [np.inf, min(c + window, len(spectrum) - 1), np.inf, np.inf, 4.0, np.inf, np.inf]

            # Ensure p_0 is feasible (clip center into [lower,upper] etc.)
            # TODO: this makes Sam sad.
            # p_0_clipped = [
            #     float(np.clip(p, l, u) if np.isfinite(u) else 1e12) for p, l, u in zip(p_0, lower, upper, strict=True)
            # ]
            p_0_clipped = [
                float(np.clip(p_0[0], lower[0], upper[0] if np.isfinite(upper[0]) else 1e12)),
                float(np.clip(p_0[1], lower[1], upper[1] if np.isfinite(upper[1]) else 1e12)),
                float(np.clip(p_0[2], lower[2], upper[2] if np.isfinite(upper[2]) else 1e12)),
                float(np.clip(p_0[3], lower[3], upper[3] if np.isfinite(upper[3]) else 1e12)),
                float(np.clip(p_0[4], lower[4], upper[4] if np.isfinite(upper[4]) else 1e12)),
                float(np.clip(p_0[5], lower[5], upper[5] if np.isfinite(upper[5]) else 1e12)),
                float(np.clip(p_0[6], lower[6], upper[6] if np.isfinite(upper[6]) else 1e12)),
            ]

            try:
                popt, pcov = curve_fit(
                    double_gaussian, x_fit, y_fit, p0=p_0_clipped, bounds=(lower, upper), maxfev=20000
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


def cal_spec(spec, est_peaks, wavelen):
    # need to add a lot of robustness checks here

    other_peaks = []
    for peak in est_peaks:
        a, b = peak
        other_peaks.append(a + np.nanargmax(spec[a:b]))
    other_new_centers, p = refine_peak_centers(spec, other_peaks, window=3)
    if settings.plot:
        # TODO: move the plotting code out
        print(p)
        plt.plot(spec)
        plt.vlines(other_new_centers, 0, 10e5, color="r", alpha=0.5)
        plt.ylim(10, np.max(spec) * 1.1)
        plt.yscale("log")
        plt.show()
    x_points = np.array(range(len(spec)))
    other_lbda = np.array(wavelen)
    p_3 = np.polyfit(other_new_centers, other_lbda, 3)
    wavelengths_3 = p_3[0] * x_points**3 + p_3[1] * x_points**2 + p_3[2] * x_points + p_3[3]

    fitted_centers_lbda = np.polyval(p_3, other_new_centers)
    residuals = fitted_centers_lbda - other_lbda
    return wavelengths_3, p_3, residuals**2


def find_closest_index(array, value):
    idx = np.argmin(np.abs(array - value))
    return idx


def recalc_spec(spec, peaks, lbda):
    # this is the part that takes the longest time
    other_new_centers, p = refine_peak_centers(spec, peaks, window=3)
    x_points = np.array(range(len(spec)))
    other_lbda = np.array(lbda)
    p_3 = np.polyfit(other_new_centers, other_lbda, 3)
    wavelengths_3 = p_3[0] * x_points**3 + p_3[1] * x_points**2 + p_3[2] * x_points + p_3[3]

    fitted_centers_lbda = np.polyval(p_3, other_new_centers)
    residuals = fitted_centers_lbda - other_lbda
    return wavelengths_3, p_3, residuals**2


def plot_params(params, name):
    fig, ax = plt.subplots(2, 2)

    params = np.array(params)

    coeff_names = ["x³ coefficient", "x² coefficient", "x¹ coefficient", "constant term"]
    labels = np.arange(1, 226).reshape(15, 15)

    x, y = np.meshgrid(np.arange(15), np.arange(15))
    for i in range(4):
        grid = params[:, i].reshape(15, 15)
        row, col = divmod(i, 2)
        ax_i = ax[row, col]
        sc = ax_i.scatter(x, y, c=grid, s=50)
        # Add labels
        for xi in range(15):
            for yi in range(15):
                ax_i.text(x[xi, yi] + 0.05, y[xi, yi] + 0.05, str(labels[xi, yi]), fontsize=8, rotation=35)
        fig.colorbar(sc, ax=ax_i, label=coeff_names[i])
        ax_i.set_title(coeff_names[i])
        ax_i.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"{name}fitcoefficients_grid.png", dpi=300, bbox_inches="tight")
    plt.show()


# TODO: move plotting into a separate file
# TODO: only call the plotting based on the global settings
def plot_spec(wavelengths, fluxes, save=False, name="sample"):
    # Flatten the data for plotting
    X = wavelengths.flatten()  # Wavelengths
    Y = np.repeat(np.arange(225), 1499)  # Object indices
    C = fluxes.flatten()  # Flux values

    # Avoid log(0) issues — filter out or replace nonpositive values
    mask = C > 1
    X, Y, C = X[mask], Y[mask], C[mask]

    plt.figure(figsize=(12, 6))
    sc = plt.scatter(X, Y, c=C, cmap="viridis", s=2, norm=LogNorm())
    plt.colorbar(sc, label="Flux (log scale)")
    plt.xlabel(r"Wavelength ($\mathrm{\AA}$)")
    plt.ylabel("Spaxel")
    plt.vlines(
        [
            5769.6,
            5460.735,
            5085.822,
            4916,
            4358.328,
            4198.317,
            4158.59,
            4077.837,
            4046.563,
            3906.371,
            3663.279,
            3650.153,
            3610.5077,
            3466.1996,
            3261.0548,
            3131.7,
        ],
        -2,
        230,
        color="k",
        linestyle="--",
        alpha=0.5,
    )

    # plt.xlim(3000,6000)
    plt.title(f"{name} Arc: Flux vs Wavelength")
    if save:
        plt.savefig(f"{name}fitwavelengths.png", dpi=300, bbox_inches="tight")


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
                big_arr = make_array(rootdir + name + "_crossSumFitArc", rootdir + name + "_fit_arc_vector.npy")
                big_wave = []
                params = []
                residuals = []
                # TODO: what is the first dim representing? ah - each spaxel
                for i in range(big_arr.shape[0]):
                    spec = big_arr[i]
                    waves, ps, res = cal_spec(spec, PEAK_ESTIMATES, WAVELENGTH_VALUES)
                    big_wave.append(waves)
                    params.append(ps)
                    residuals.extend(res)
                print("early RMS: ", np.sqrt(np.mean(residuals)))
                big_wave = np.array(big_wave)
                print("refitting")
                residuals = []
                for i in range(big_arr.shape[0]):
                    spec = big_arr[i]
                    closest_indices = [find_closest_index(big_wave[i], p) for p in PEAKS]
                    waves, ps, res = recalc_spec(spec, closest_indices, PEAKS)
                    big_wave[i] = waves
                    params[i] = ps
                    residuals.extend(res)
                plot_params(params, name)
                print("late RMS: ", np.sqrt(np.mean(residuals)))
                plot_spec(big_wave, big_arr, save=True, name=name)
                np.save(f"{name}fitWavelengthCal", big_wave)
                print(f"done with {name}")
