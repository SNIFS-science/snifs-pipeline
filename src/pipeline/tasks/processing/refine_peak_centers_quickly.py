import numpy as np


def log(x):
    with open("refine_peaks_logs.txt", "a") as file:
        file.write(x + ",")


def gaussian(x: np.ndarray, amp: float, mu: float, sigma: float, offset: float) -> np.ndarray:
    """Args:
        x : The input data points.
        amp : Amplitude of the Gaussian.
        mu : Mean of the Gaussian.
        sigma : Standard deviation of the Gaussian.
        offset : Offset of the Gaussian.

    Returns:
        np.ndarray: The Gaussian function evaluated at the input data points.
    """
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset


def gaussian_jac(x, amp, mu, sigma, offset):
    """Jacobian for the gaussian with respect to amp, mu, sigma, offset"""
    # sigma1 = np.clip(sigma1, 1e-8, None)

    exp = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    # print(f"here and sigma is {sigma}")
    # log(sigma)
    # der_amp = exp
    # der_mu = amp * exp * (x-mu) / (sigma ** 2)
    # der_sigma = amp * exp * ((x - mu) ** 2 ) / (sigma ** 3)
    # der_offset = 1
    # sigma = np.clip(sigma, 1e-8, None)

    jac = np.vstack(
        (exp, amp * exp * (x - mu) / (sigma**2), amp * exp * ((x - mu) ** 2) / (sigma**3), np.ones(len(x)))
    ).T
    return jac


def double_gaussian(
    x: np.ndarray, amp1: float, center: float, sigma1: float, amp2: float, delta: float, sigma2: float, offset: float
) -> np.ndarray:
    """Defined so we can control the distance between the two peaks of the double Gaussian.

    Args:
        x : The input data points.
        amp1 : Amplitude of the first Gaussian.
        center : Center of the double Gaussian.
        sigma1 : Standard deviation of the first Gaussian.
        amp2 : Amplitude of the second Gaussian.
        delta : Distance between the centers of the two Gaussians.
        sigma2 : Standard deviation of the second Gaussian.
        offset : Offset of the double Gaussian.

    Returns:
        np.ndarray: The double Gaussian function evaluated at the input data points.
    """
    mu1 = center - delta / 2
    mu2 = center + delta / 2
    return amp1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2) + amp2 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2) + offset


def double_gaussian_jac(x, amp1, center, sigma1, amp2, delta, sigma2, offset):
    mu1 = center - delta / 2
    mu2 = center + delta / 2

    # sigma1 = np.clip(sigma1, 1e-8, None)
    # sigma2 = np.clip(sigma2, 1e-8, None)

    exp1 = np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
    exp2 = np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)

    der_amp1 = exp1
    der_center = amp1 * exp1 * (x - mu1) / (sigma1**2) + amp2 * exp2 * (x - mu2) / (sigma2**2)
    der_sigma1 = amp1 * exp1 * ((x - mu1) ** 2) / (sigma1**3)
    der_amp2 = exp2
    der_delta = (amp2 * exp2 * (x - mu2) / (sigma2**2) - amp1 * exp1 * (x - mu1) / (sigma1**2)) / 2
    der_sigma2 = amp2 * exp2 * ((x - mu2) ** 2) / (sigma2**3)
    der_offset = 1

    jac = np.vstack((der_amp1, der_center, der_sigma1, der_amp2, der_delta, der_sigma2, der_offset)).T
    return jac


def refine_peak_centers_quickly(
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
                    gaussian,
                    x_fit,
                    y_fit,
                    p0=p_0_single,
                    bounds=(lower_single, upper_single),
                    maxfev=20000,  # IN THE FUTURE GIVE WEIGHT TO CONFIDENCE BY FINDING SIGMA OF BETWEEN FROM FITS TO EXPONENTIAL
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
