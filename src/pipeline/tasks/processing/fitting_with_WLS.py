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

    error = 1

    # CHANGE IS HERE
    p_3 = np.polyfit(other_new_centers, other_lbda, 3, w=1 / error)

    wavelengths_3 = p_3[0] * x_points**3 + p_3[1] * x_points**2 + p_3[2] * x_points + p_3[3]

    fitted_centers_lbda = np.polyval(p_3, other_new_centers)
    residuals = fitted_centers_lbda - other_lbda
    return wavelengths_3, p_3, residuals**2


# comments on refine_peak_centers
# why start double peak guess like c + window/4??

# why /4

# split fitting gaussians onto multiple cpus
# 3 things to speed up
# parallelize
# guess P0 using smoothness
# give jacobian to the curve fit


# like partition cpus into chuncks of proximitiy so we can use the smoothness of peak variations to supply P0 guesses. surrounding tiles give information
# we can use the iterative paths to build good initial guesses into parallelizing

# popt, pcov = opt.curve_fit(func, xdata, ydata, jac=jacobian) give jacobian

# can MSE with gaussian jacbobian be calculated analytically for curve fitting with gaussian and given to scipy optimize curve fit to speed up

# utilize returned fit params or search range for wavelengths actually I wanna ask about that. We can encorporate average info comparing ot neighoors.


# new topic TODOs
# robustness checks it returns residuals

# tqdm for tracking iteration speed
