import numpy as np


def gaussian(x: np.ndarray, amp: float, mu: float, sigma: float, offset: float) -> np.ndarray:
    """
    Args:
        x : The input data points.
        amp : Amplitude of the Gaussian.
        mu : Mean of the Gaussian.
        sigma : Standard deviation of the Gaussian.
        offset : Offset of the Gaussian.
    Returns:
        np.ndarray: The Gaussian function evaluated at the input data points.
    """
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset


def double_gaussian(
    x: np.ndarray, amp1: float, center: float, sigma1: float, amp2: float, delta: float, sigma2: float, offset: float
) -> np.ndarray:
    """
    Defined so we can control the distance between the two peaks of the double Gaussian.
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


def gauss_2d(xy: tuple[np.ndarray, np.ndarray], x0: float, y0: float, a: float, w1: float, w2: float) -> np.ndarray:
    """
    Args:
        xy: Tuple of (x, y) coordinates.
        x0: Center position in x.
        y0: Center position in y.
        a: Amplitude of the Gaussian.
        w1: Width in x direction.
        w2: Width in y direction.
    Returns:
        np.ndarray: The value of the 2D Gaussian at the given (x, y) coordinates.
    """
    x, y = xy
    return a * np.exp(-((x - x0) ** 2 / w1**2 + (y - y0) ** 2 / w2**2))


def func_2nd(x: np.ndarray, y: np.ndarray, a: float, b: float, c: float, d: float, e: float, f: float) -> np.ndarray:
    """
    Args:
        x: x coordinates.
        y: y coordinates.
        a, b, c, d, e, f: Coefficients for the 2nd order polynomial.
    Returns:
        The value of the 2nd order polynomial at the given (x, y) coordinates
    """
    return a + b * x + c * y + d * x**2 + e * x * y + f * y**2


def func_3rd(
    x: np.ndarray,
    y: np.ndarray,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    g: float,
    h: float,
    k: float,
    m: float,
) -> np.ndarray:
    """
    Args:
        x: x coordinates.
        y: y coordinates.
        a, b, c, d, e, f, g, h, k, m: Coefficients for the 3rd order polynomial.
    Returns:
        The value of the 3rd order polynomial at the given (x, y) coordinates
    """
    return a + b * x + c * y + d * x**2 + e * x * y + f * y**2 + g * x**3 + h * x**2 * y + k * x * y**2 + m * y**3


def func_4th(
    x: np.ndarray,
    y: np.ndarray,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    g: float,
    h: float,
    k: float,
    m: float,
    n: float,
    o: float,
    p: float,
    q: float,
    r: float,
) -> np.ndarray:
    """
    Args:
        x: x coordinates.
        a, b, c, d, e, f, g, h, k, m, n, o, p, q,r: Coefficients for the 4th order polynomial.
    Returns:
        The value of the 4th order polynomial at the given (x, y) coordinates
    """
    return (
        a
        + b * x
        + c * y
        + d * x**2
        + e * x * y
        + f * y**2
        + g * x**3
        + h * x**2 * y
        + k * x * y**2
        + m * y**3
        + n * x**4
        + o * x**3 * y
        + p * x**2 * y**2
        + q * x * y**3
        + r * y**4
    )


def pseudo_voigt(
    x: np.ndarray, xo: float, wg: float, wl: float, n: float, eta: float, beta: float | None = None, l_off: float = 1
):
    """
    Args:
        x : The input data points.
        xo : Center position.
        wg : Gaussian width.
        wl : Lorentzian width.
        n : Power for the Lorentzian component.
        eta : Mixing parameter for Lorentzian component.
        beta : Mixing parameter for Gaussian component. If None, it is set to (1 - eta).
        l_off : Offset for the Lorentzian component.
    Returns:
        np.ndarray: The pseudo-Voigt function evaluated at the input data points.
    """
    if beta is None:
        beta = 1 - eta
    G = np.exp(-np.log(2) * (x - xo) ** 2 / wg**2)
    L = 1 / (l_off + (x - xo) ** n / wl**n)
    PV = eta * L + (beta) * G
    return PV


def core_2d(x: np.ndarray, y: np.ndarray, x0: float, y0: float, a: float, w1: float, w2: float) -> np.ndarray:
    """
    Args:
        x: x coordinates.
        y: y coordinates.
        x0: Center position in x.
        y0: Center position in y.
        a: Amplitude of the pseudo-Voigt.
        w1: Width in x direction.
        w2: Width in y direction.
    Returns:
        np.ndarray: The value of the 2D pseudo-Voigt at the given (x, y) coordinates.
    """
    cross = pseudo_voigt(np.abs(x - x0), 0, 1.0 * w1, 1.6 * w1, 4.5, 0.1)  # + PV(x, 0, 1.2, 0.1, -n, 0.1,beta=0)
    spec = pseudo_voigt(np.abs(y - y0), 0, 1.0 * w2, 1.6 * w2, 4.5, 0.1)  # + PV(y, 0, 1.2, 0.2, -n, 0.1,beta=0)
    return a * cross * spec
