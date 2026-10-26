"""Fit the trace shift, width and profile shape of a group of 15 spaxels by Gauss-Newton.

The model is the one make_parameter_matrix builds: image = A(p).T @ s, with the spectra s solved linearly and the
shape parameters p fitted by damped Gauss-Newton (Levenberg-Marquardt) steps on all 15 spaxels at once. Derivatives
are central differences of the profile as built, and the spectra are re-solved for every parameter set (variable
projection). The fit runs in two stages: shift and width curves first, then also the cross-dispersion core power, the
wing amplitude and, optionally, a profile asymmetry; fitting the profile from uncorrected curves overshoots.

Curves are cubic B-splines along the spectrum, held flat where a spaxel's signal is low. Results are written to
gn_loop_{shifts,widths}_editable.json and gn_profile_editable.json; make_parameter_matrix still reads only np.poly1d
coefficients and hard-codes the profile, so it cannot use them yet.
"""

import json
import socket
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from astropy.io import fits
from numpy.polynomial import Polynomial
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.ndimage import uniform_filter1d
from scipy.sparse import eye as speye
from scipy.sparse.linalg import splu

from pipeline.common.fitting_math__utils import pseudo_voigt
from pipeline.common.log import get_logger
from pipeline.common.model_params import (
    A0_PARAMS,
    A1_PARAMS,
    B0_PARAMS,
    B1_PARAMS,
    ELL_A,
    ELL_B,
    ELL_C0,
    ELL_C1,
    LIN_CROSS,
    LIN_SPEC,
    QUAD_SPEC,
    QUARTIC_LINEAR_PARAMS,
    Z_1ST,
    Z_2ND,
    Z_4TH,
)
from pipeline.common.prefect_utils import pipeline_flow
from pipeline.tasks.plotting.wavelength_arc_calibration_plots import (
    noise_model,
    plot_fit_difference,
    residual_stats,
    trace_centres,
)
from pipeline.tasks.processing.make_parameter_matrix import (
    _git_commit,
    _load_science_image,
    _peak_memory_mb,
    columnindex,
    n_cross,
    n_spec,
    rowindex,
    xoff,
    yoff,
)

logger = get_logger()

N_ELEMENTS = 1400
N_PIXELS = 2048 * 4096
MASKED_ELEMENTS = (220, 540)  # default; matches the polyfit mask in make_parameter_matrix.l1_calculations
X0 = np.arange(N_ELEMENTS)
T = (X0 - (N_ELEMENTS - 1) / 2) / ((N_ELEMENTS - 1) / 2)  # spectral element mapped onto [-1, 1]
CORE_POWER = 5.4  # power of the core pseudo-Voigt of `spectral` (the cross-dispersion profile) in make_parameter_matrix
WING_AMP = 0.1  # amplitude of its power-law wing term


def _binned_profile(
    xv_sub_mono: np.ndarray,
    yv_sub_mono: np.ndarray,
    spec_element: int,
    centre: float,
    width: float,
    spaxel_ID: int,
    oversample_factor: int,
    gauss_width_scale: float,
    lorentz_width_scale: float,
    core_power: float = CORE_POWER,
    wing_amp: float = WING_AMP,
    asymmetry: float = 0.0,
    asymmetry_mode: str = "shoulder",
) -> np.ndarray:
    """One spectral element's profile on its window, as in make_parameter_matrix._build_spaxel_rows.

    The defaults reproduce make_parameter_matrix exactly.

    Args:
        xv_sub_mono: Column coordinates of the element's sub-sampled window.
        yv_sub_mono: Row coordinates of the element's sub-sampled window.
        spec_element: Spectral element index.
        centre: Trace centre column (in window coordinates).
        width: Profile width.
        spaxel_ID: Spaxel index, for its trace and footprint parameters.
        oversample_factor: Sub-samples per pixel in each direction.
        gauss_width_scale: Gaussian width of the cross-dispersion core, per unit width.
        lorentz_width_scale: Lorentzian width of the cross-dispersion core, per unit width.
        core_power: Power of the cross-dispersion core pseudo-Voigt.
        wing_amp: Amplitude of the cross-dispersion power-law wing.
        asymmetry: Widths are scaled by (1 + asymmetry) on the +column side of the trace and (1 - asymmetry) on the
            other.
        asymmetry_mode: "shoulder" scales only the Lorentzian width, "core" also the Gaussian width.

    Returns:
        np.ndarray: The pixel-binned profile, normalised to a peak of 1 before binning.
    """
    y = yv_sub_mono.T[0] - spec_element
    x = xv_sub_mono[0] - centre

    spec_trace = QUAD_SPEC[spaxel_ID] * y**2 + LIN_SPEC[spaxel_ID] * y
    cross_trace = LIN_CROSS[spaxel_ID] * x

    xiii = (xv_sub_mono - centre).T - spec_trace
    yiii = (yv_sub_mono - spec_element) - cross_trace

    mask_footprint = (
        np.sqrt(
            (yv_sub_mono - spec_element - ELL_C1[spaxel_ID]) ** 2 / ELL_B[spaxel_ID] ** 2
            + (xv_sub_mono - centre - ELL_C0[spaxel_ID]) ** 2 / ELL_A[spaxel_ID] ** 2
        )
        < 1
    )

    side_width = width * (1 + asymmetry * np.sign(xiii)) if asymmetry else width
    gauss_width = side_width if asymmetry_mode == "core" else width
    spectral = 0.8 * pseudo_voigt(
        np.abs(xiii), 0, gauss_width_scale * gauss_width, lorentz_width_scale * side_width, core_power, 0.6
    ) + pseudo_voigt(np.abs(xiii), 0, 1.2, 0.2, -n_spec, wing_amp, beta=0)
    crossdis = 0.99 * pseudo_voigt(np.abs(yiii), 0, 0.6 * width, 1.4 * width, 5.2, 0.6) + pseudo_voigt(
        np.abs(yiii), 0, 1.2, 0.1, -n_cross, 0.1, beta=0, l_off=10
    )

    model = spectral * crossdis.T * (mask_footprint.T * 1.0)
    model = model / np.max(model)

    H = model.shape[0] // oversample_factor
    W = model.shape[1] // oversample_factor
    return (
        model[: H * oversample_factor, : W * oversample_factor]
        .reshape(H, oversample_factor, W, oversample_factor)
        .sum(axis=(1, 3))
    )


def _build_spaxel_with_derivatives(
    spaxel_ID: int,
    local_spaxel_idx: int,
    shift_vals: np.ndarray,
    width_vals: np.ndarray,
    oversample_factor: int,
    eps_shift: float,
    eps_width: float,
    derivatives: bool = True,
    core_power: float = CORE_POWER,
    wing_amp: np.ndarray | None = None,
    profile_derivatives: bool = False,
    asymmetry: np.ndarray | None = None,
    asymmetry_derivatives: bool = False,
    asymmetry_mode: str = "shoulder",
    eps_power: float = 0.01,
    eps_wing: float = 0.002,
    eps_asymmetry: float = 0.005,
    gauss_width_scale: float = 0.6,
    lorentz_width_scale: float = 1.1,
) -> tuple:
    """Build one spaxel's model rows and, by central differences, their derivatives.

    The derivatives use the same window and pixels as the model, so the cutoff and normalisation are included.

    Args:
        spaxel_ID: Spaxel index.
        local_spaxel_idx: Position of the spaxel among the modelled spaxels (sets its rows).
        shift_vals: Trace shift per spectral element, added to the nominal trace.
        width_vals: Profile width per spectral element.
        oversample_factor: Sub-samples per pixel in each direction.
        eps_shift: Central-difference step for the shift.
        eps_width: Central-difference step for the width.
        derivatives: Whether to compute the shift and width derivatives.
        core_power: Cross-dispersion core power.
        wing_amp: Wing amplitude per spectral element (default make_parameter_matrix's).
        profile_derivatives: Whether to compute the core-power and wing-amplitude derivatives.
        asymmetry: Asymmetry per spectral element (default 0).
        asymmetry_derivatives: Whether to compute the asymmetry derivative.
        asymmetry_mode: See _binned_profile.
        eps_power: Central-difference step for the core power.
        eps_wing: Central-difference step for the wing amplitude.
        eps_asymmetry: Central-difference step for the asymmetry.
        gauss_width_scale: See _binned_profile.
        lorentz_width_scale: See _binned_profile.

    Returns:
        tuple: (data, d_shift, d_width, d_power, d_wing, d_asymmetry, rows, cols) as COO triplet arrays; derivatives
        that were not computed are None.
    """
    if wing_amp is None:
        wing_amp = np.full(N_ELEMENTS, WING_AMP)
    if asymmetry is None:
        asymmetry = np.zeros(N_ELEMENTS)
    all_data: list = []
    all_dshift: list = []
    all_dwidth: list = []
    all_dpower: list = []
    all_dwing: list = []
    all_dasym: list = []
    all_rows: list = []
    all_cols: list = []

    a0 = int(A0_PARAMS[spaxel_ID] + yoff)
    a1 = int(A1_PARAMS[spaxel_ID] + yoff) + 1
    b0 = int(B0_PARAMS[spaxel_ID] + xoff - 50)
    off = 50
    if b0 < 0:
        off += b0
        b0 = 0
    b1 = int(B1_PARAMS[spaxel_ID] + xoff + 50) + 1

    xsub = (np.arange((a1 - a0) * oversample_factor) + 0.5) / oversample_factor - 0.5
    ysub = (np.arange((b1 - b0) * oversample_factor) + 0.5) / oversample_factor - 0.5
    xv_sub, yv_sub = np.meshgrid(ysub, xsub)

    rowind_slab = rowindex[a0:a1, b0:b1]
    colind_slab = columnindex[a0:a1, b0:b1]

    p = Polynomial([QUARTIC_LINEAR_PARAMS[spaxel_ID], Z_1ST[spaxel_ID], Z_2ND[spaxel_ID], 0, Z_4TH[spaxel_ID]])
    curve = p(X0) + off + shift_vals

    # (profile argument, step, output list) of each central difference
    perturbations = []
    if derivatives:
        perturbations += [("centre", eps_shift, all_dshift), ("width", eps_width, all_dwidth)]
    if profile_derivatives:
        perturbations += [("core_power", eps_power, all_dpower), ("wing_amp", eps_wing, all_dwing)]
    if asymmetry_derivatives:
        perturbations += [("asymmetry", eps_asymmetry, all_dasym)]

    for spec_element in range(N_ELEMENTS):
        c0 = max(int(spec_element - 50) * oversample_factor, 0)
        c1 = min(int(spec_element + 50) * oversample_factor, N_ELEMENTS * oversample_factor)
        d0 = max(int(round(curve[spec_element] - 50)) * oversample_factor, 0)
        d1 = int(round(curve[spec_element] + 50)) * oversample_factor
        if d1 >= N_ELEMENTS * oversample_factor:
            d1 = N_ELEMENTS * oversample_factor - 1

        xv_sub_mono = xv_sub[c0:c1, d0:d1]
        yv_sub_mono = yv_sub[c0:c1, d0:d1]

        profile = partial(
            _binned_profile,
            xv_sub_mono,
            yv_sub_mono,
            spec_element,
            spaxel_ID=spaxel_ID,
            oversample_factor=oversample_factor,
            gauss_width_scale=gauss_width_scale,
            lorentz_width_scale=lorentz_width_scale,
            asymmetry_mode=asymmetry_mode,
        )
        point = {
            "centre": curve[spec_element],
            "width": width_vals[spec_element],
            "core_power": core_power,
            "wing_amp": wing_amp[spec_element],
            "asymmetry": asymmetry[spec_element],
        }
        base = profile(**point)

        c0s, c1s = c0 // oversample_factor, c1 // oversample_factor
        d0s, d1s = d0 // oversample_factor, d1 // oversample_factor

        rowind = rowind_slab[c0s:c1s, d0s:d1s]
        colind = colind_slab[c0s:c1s, d0s:d1s]
        n_r, n_c = rowind.shape
        mv = (base.T > 1e-4)[:n_r, :n_c]

        all_data.append(base.T[:n_r, :n_c][mv])
        for name, eps, derivative_list in perturbations:
            plus = profile(**{**point, name: point[name] + eps})
            minus = profile(**{**point, name: point[name] - eps})
            derivative_list.append(((plus - minus).T[:n_r, :n_c][mv]) / (2 * eps))
        n = int(mv.sum())
        all_rows.append(np.full(n, local_spaxel_idx * N_ELEMENTS + spec_element, dtype=np.int32))
        all_cols.append((colind[mv] * 2048 + rowind[mv]).astype(np.int32))

    return (
        np.concatenate(all_data),
        np.concatenate(all_dshift) if derivatives else None,
        np.concatenate(all_dwidth) if derivatives else None,
        np.concatenate(all_dpower) if profile_derivatives else None,
        np.concatenate(all_dwing) if profile_derivatives else None,
        np.concatenate(all_dasym) if asymmetry_derivatives else None,
        np.concatenate(all_rows),
        np.concatenate(all_cols),
    )


def _build_job(job: dict) -> tuple:
    return _build_spaxel_with_derivatives(**job)


def _spline_knots(segments: int) -> np.ndarray:
    """Clamped cubic B-spline knots over t in [-1, 1].

    Args:
        segments: Number of equal knot intervals.

    Returns:
        np.ndarray: The knot vector.
    """
    return np.concatenate([[-1.0] * 3, np.linspace(-1, 1, segments + 1), [1.0] * 3])


def _slope_basis(knots: np.ndarray) -> np.ndarray:
    """d/dt of each cubic B-spline basis function, evaluated per spectral element.

    Args:
        knots: Spline knot vector (see _spline_knots).

    Returns:
        np.ndarray: (1400, n_basis) array.
    """
    n_basis = len(knots) - 4
    return np.stack([BSpline(knots, np.eye(n_basis)[k], 3).derivative()(T) for k in range(n_basis)], axis=1)


def _spline_entry(knots: np.ndarray, coeffs: np.ndarray) -> dict:
    """JSON form of a cubic B-spline curve in t = (element - 699.5) / 699.5.

    Args:
        knots: Spline knot vector.
        coeffs: Spline coefficients.

    Returns:
        dict: {"degree", "knots", "coeffs"}.
    """
    return {"degree": 3, "knots": knots.tolist(), "coeffs": coeffs.tolist()}


def _latest_curve(history: dict, default: float) -> np.ndarray:
    """Per-element curve of the latest iteration in a history.

    A curve is np.poly1d coefficients in the element index (tester_loop_* JSONs) or a _spline_entry (gn_loop_*).

    Args:
        history: {iteration: curve}.
        default: Constant value used if the history is empty.

    Returns:
        np.ndarray: The curve at the 1400 spectral elements.
    """
    if not history:
        return np.full(N_ELEMENTS, default)
    entry = history[str(max(int(k) for k in history))]
    if isinstance(entry, dict):
        return BSpline(np.asarray(entry["knots"]), np.asarray(entry["coeffs"]), entry["degree"])(T)
    return np.poly1d(entry)(X0)


def curves_from_jsons(spaxels: list[int], shift_coeffs: dict, width_coeffs: dict) -> dict[int, tuple]:
    """Shift and width curves of each spaxel from loaded JSONs.

    Args:
        spaxels: Spaxel indices.
        shift_coeffs: Shift JSON contents ({spaxel: {iteration: curve}}).
        width_coeffs: Width JSON contents.

    Returns:
        dict[int, tuple]: {spaxel: (shift curve, width curve)}; spaxels missing from the JSONs get no correction.
    """
    return {
        sid: (_latest_curve(shift_coeffs.get(str(sid), {}), 0.0), _latest_curve(width_coeffs.get(str(sid), {}), 1.0))
        for sid in spaxels
    }


class GaussNewtonGroupFit:
    """Model, Jacobian and normal equations for one stage of the fit of one group of 15 spaxels.

    The model contains the group plus `pad` spaxels on each side, whose spectra are solved but whose curves stay
    fixed. The parameters are one flat vector: each group spaxel's shift then width spline coefficients; then, with
    fit_profile, the core power and the log-wing-amplitude spline; then, with fit_asymmetry, the asymmetry spline.
    A penalty on the slope of each curve holds it flat where the signal is below prior_threshold of its peak.

    Args:
        science_image: The (4096, 2048) image to fit.
        group: Group index (spaxels 15 * group to 15 * group + 14).
        segments: Spline intervals of the shift and width curves.
        oversample_factor: Sub-samples per pixel.
        eps_shift: Central-difference step for the shift.
        eps_width: Central-difference step for the width.
        pool: Worker processes for building spaxels, or None.
        pad: Spaxels modelled on each side of the group.
        fixed_curves: {spaxel: (shift, width)} for the padding spaxels (default no correction).
        prior_threshold: Signal fraction below which curves are held flat.
        prior_strength: Strength of that hold (0 = off).
        fit_profile: Whether to fit the core power and wing amplitude.
        wing_segments: Spline intervals of the wing amplitude.
        fit_asymmetry: Whether to fit the profile asymmetry.
        asymmetry_segments: Spline intervals of the asymmetry.
        asymmetry_mode: See _binned_profile.
        masked_elements: (start, stop) spectral elements left out of the misfit, or None.
    """

    def __init__(
        self,
        science_image: np.ndarray,
        group: int,
        segments: int = 8,
        oversample_factor: int = 1,
        eps_shift: float = 0.01,
        eps_width: float = 0.01,
        pool: ProcessPoolExecutor | None = None,
        pad: int = 7,
        fixed_curves: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
        prior_threshold: float = 0.03,
        prior_strength: float = 1.0,
        fit_profile: bool = True,
        wing_segments: int = 3,
        fit_asymmetry: bool = False,
        asymmetry_segments: int = 3,
        asymmetry_mode: str = "shoulder",
        masked_elements: tuple[int, int] | None = MASKED_ELEMENTS,
    ):
        self.image_flat = np.asarray(science_image, dtype=float).ravel()
        self.finite = np.isfinite(self.image_flat)
        self.data = np.where(self.finite, self.image_flat, 0.0)
        self.group_start = 15 * group
        self.spaxels = list(range(self.group_start, self.group_start + 15))
        self.model_spaxels = list(range(max(0, self.group_start - pad), min(225, self.group_start + 15 + pad)))
        self.n_model = len(self.model_spaxels)
        self.fit_offset = self.group_start - self.model_spaxels[0]  # local index of the group's first spaxel
        self.padding = [s for s in self.model_spaxels if s not in self.spaxels]
        fixed_curves = fixed_curves or {}
        self.fixed_curves = {s: fixed_curves.get(s, (np.zeros(N_ELEMENTS), np.ones(N_ELEMENTS))) for s in self.padding}
        self.padding_parts: list | None = None  # built once and reused (also by the next stage)
        self.knots = _spline_knots(segments)
        self.n_coeff = segments + 3  # cubic B-spline basis functions per curve
        self.n_curve_params = 15 * 2 * self.n_coeff
        self.fit_profile = fit_profile
        self.wing_knots = _spline_knots(wing_segments)
        self.n_wing = wing_segments + 3
        self.wing_basis = BSpline.design_matrix(T, self.wing_knots, 3).toarray()  # (1400, n_wing)
        self.wing_slope_basis = _slope_basis(self.wing_knots)
        self.wing_offset = self.n_curve_params + 1  # first log-wing coefficient (the core power is just before it)
        self.fit_asymmetry = fit_asymmetry
        self.asymmetry_mode = asymmetry_mode
        self.masked_elements = masked_elements
        self.asymmetry_knots = _spline_knots(asymmetry_segments)
        self.n_asymmetry = asymmetry_segments + 3
        self.asymmetry_basis = BSpline.design_matrix(T, self.asymmetry_knots, 3).toarray()  # (1400, n_asymmetry)
        self.asymmetry_slope_basis = _slope_basis(self.asymmetry_knots)
        self.asymmetry_offset = self.n_curve_params + (1 + self.n_wing if fit_profile else 0)
        self.n_params = self.asymmetry_offset + (self.n_asymmetry if fit_asymmetry else 0)
        self.oversample_factor = oversample_factor
        self.eps_shift = eps_shift
        self.eps_width = eps_width
        self.pool = pool
        self.basis = BSpline.design_matrix(T, self.knots, 3).toarray()  # (1400, n_coeff)
        self.slope_basis = _slope_basis(self.knots)
        self.prior_threshold = prior_threshold
        self.prior_strength = prior_strength
        # Set on the first evaluate() (the starting parameters), then kept fixed
        self.weights: np.ndarray | None = None
        self.prior_matrix: np.ndarray | None = None
        self.pinned: np.ndarray | None = None  # (15, 1400), 1 = held flat, 0 = free

    # --- parameters ---

    def shift_and_width(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Shift and width curves of the group spaxels.

        Args:
            params: Parameter vector.

        Returns:
            tuple[np.ndarray, np.ndarray]: (15, 1400) shift and width curves.
        """
        curve_params = params[: self.n_curve_params].reshape(15, 2 * self.n_coeff)
        return curve_params[:, : self.n_coeff] @ self.basis.T, curve_params[:, self.n_coeff :] @ self.basis.T

    def profile_shape(self, params: np.ndarray) -> tuple[float, np.ndarray]:
        """Core power and wing amplitude of the group (make_parameter_matrix's if the profile is not fitted).

        The wing amplitude is exp of a spline, so it stays positive.

        Args:
            params: Parameter vector.

        Returns:
            tuple[float, np.ndarray]: The core power and the per-element wing amplitude.
        """
        if not self.fit_profile:
            return CORE_POWER, np.full(N_ELEMENTS, WING_AMP)
        log_wing_coeffs = params[self.wing_offset : self.wing_offset + self.n_wing]
        return float(params[self.n_curve_params]), np.exp(self.wing_basis @ log_wing_coeffs)

    def asymmetry(self, params: np.ndarray) -> np.ndarray:
        """Asymmetry of the group's profile.

        Args:
            params: Parameter vector.

        Returns:
            np.ndarray: Per-element asymmetry (0 if it is not fitted).
        """
        if not self.fit_asymmetry:
            return np.zeros(N_ELEMENTS)
        return self.asymmetry_basis @ params[self.asymmetry_offset : self.asymmetry_offset + self.n_asymmetry]

    def initial_params(
        self,
        curves: dict[int, tuple[np.ndarray, np.ndarray]],
        core_power: float = CORE_POWER,
        wing_amp: float | np.ndarray = WING_AMP,
        asymmetry: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        """Parameter vector least-squares fitted to given curves.

        Args:
            curves: {spaxel: (shift, width)} for the group spaxels.
            core_power: Starting core power.
            wing_amp: Starting wing amplitude, constant or per element.
            asymmetry: Starting asymmetry, constant or per element.

        Returns:
            np.ndarray: The parameter vector.
        """
        params = np.zeros(self.n_params)
        curve_params = params[: self.n_curve_params].reshape(15, 2 * self.n_coeff)
        for i, sid in enumerate(self.spaxels):
            shift, width = curves[sid]
            curve_params[i, : self.n_coeff] = np.linalg.lstsq(self.basis, shift, rcond=None)[0]
            curve_params[i, self.n_coeff :] = np.linalg.lstsq(self.basis, width, rcond=None)[0]
        if self.fit_profile:
            params[self.n_curve_params] = core_power
            log_wing = np.log(np.broadcast_to(np.asarray(wing_amp, dtype=float), (N_ELEMENTS,)))
            params[self.wing_offset : self.wing_offset + self.n_wing] = np.linalg.lstsq(
                self.wing_basis, log_wing, rcond=None
            )[0]
        if self.fit_asymmetry:
            asym_curve = np.broadcast_to(np.asarray(asymmetry, dtype=float), (N_ELEMENTS,))
            params[self.asymmetry_offset :] = np.linalg.lstsq(self.asymmetry_basis, asym_curve, rcond=None)[0]
        return params

    def curves(self, params: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Shift and width curves of the group spaxels, for starting the next stage.

        Args:
            params: Parameter vector.

        Returns:
            dict[int, tuple[np.ndarray, np.ndarray]]: {spaxel: (shift, width)}.
        """
        shifts, widths = self.shift_and_width(params)
        return {sid: (shifts[i], widths[i]) for i, sid in enumerate(self.spaxels)}

    def penalty(self, params: np.ndarray) -> float:
        """Flat-hold penalty of a parameter vector.

        Args:
            params: Parameter vector.

        Returns:
            float: The penalty.
        """
        return float(params @ self.prior_matrix @ params)

    def spline_entries(self, params: np.ndarray) -> tuple[list[dict], list[dict]]:
        """JSON entries of every group spaxel's shift and width.

        Args:
            params: Parameter vector.

        Returns:
            tuple[list[dict], list[dict]]: Shift and width entries (see _spline_entry), in group order.
        """
        curve_params = params[: self.n_curve_params].reshape(15, 2 * self.n_coeff)
        return (
            [_spline_entry(self.knots, p[: self.n_coeff]) for p in curve_params],
            [_spline_entry(self.knots, p[self.n_coeff :]) for p in curve_params],
        )

    def profile_entry(self, params: np.ndarray) -> dict:
        """JSON entry of the group's profile shape.

        Args:
            params: Parameter vector.

        Returns:
            dict: {"core_power", "log_wing_amp"} and, if fitted, {"asymmetry", "asymmetry_mode"}.
        """
        core_power, _ = self.profile_shape(params)
        if self.fit_profile:
            log_wing_coeffs = params[self.wing_offset : self.wing_offset + self.n_wing]
        else:
            log_wing_coeffs = np.full(self.n_wing, np.log(WING_AMP))
        entry = {"core_power": core_power, "log_wing_amp": _spline_entry(self.wing_knots, log_wing_coeffs)}
        if self.fit_asymmetry:
            entry["asymmetry"] = _spline_entry(self.asymmetry_knots, params[self.asymmetry_offset :])
            entry["asymmetry_mode"] = self.asymmetry_mode
        return entry

    # --- model ---

    def build(self, params: np.ndarray) -> tuple[sparse.csr_matrix, ...]:
        """Model matrix and the derivative matrices of the group's rows.

        Args:
            params: Parameter vector.

        Returns:
            tuple[sparse.csr_matrix, ...]: (A, D_shift, D_width, D_power, D_wing, D_asymmetry), all (model elements x
            pixels); derivatives that are not fitted are None.
        """
        shifts, widths = self.shift_and_width(params)
        core_power, wing_amp = self.profile_shape(params)
        asymmetry = self.asymmetry(params)
        common = {"oversample_factor": self.oversample_factor, "eps_shift": self.eps_shift, "eps_width": self.eps_width}
        jobs = [
            {
                "spaxel_ID": sid,
                "local_spaxel_idx": self.fit_offset + i,
                "shift_vals": shifts[i],
                "width_vals": widths[i],
                "core_power": core_power,
                "wing_amp": wing_amp,
                "profile_derivatives": self.fit_profile,
                "asymmetry": asymmetry,
                "asymmetry_derivatives": self.fit_asymmetry,
                "asymmetry_mode": self.asymmetry_mode,
                **common,
            }
            for i, sid in enumerate(self.spaxels)
        ]
        if self.padding_parts is None:
            jobs += [
                {
                    "spaxel_ID": sid,
                    "local_spaxel_idx": self.model_spaxels.index(sid),
                    "shift_vals": self.fixed_curves[sid][0],
                    "width_vals": self.fixed_curves[sid][1],
                    "derivatives": False,
                    **common,
                }
                for sid in self.padding
            ]
        parts = list(self.pool.map(_build_job, jobs)) if self.pool is not None else [_build_job(j) for j in jobs]
        fit_parts = parts[:15]
        if self.padding_parts is None:
            self.padding_parts = [(p[0], p[-2], p[-1]) for p in parts[15:]]
        del parts

        shape = (self.n_model * N_ELEMENTS, N_PIXELS)
        model_parts = [(p[0], p[-2], p[-1]) for p in fit_parts] + self.padding_parts
        data, rows, cols = (np.concatenate([part[k] for part in model_parts]) for k in range(3))
        A = sparse.csr_matrix((data, (rows, cols)), shape=shape)
        rows, cols = np.concatenate([p[-2] for p in fit_parts]), np.concatenate([p[-1] for p in fit_parts])
        derivative_matrices = [
            sparse.csr_matrix((np.concatenate([p[k] for p in fit_parts]), (rows, cols)), shape=shape)
            if fit_parts[0][k] is not None
            else None
            for k in (1, 2, 3, 4, 5)
        ]
        return (A, *derivative_matrices)

    def evaluate(self, params: np.ndarray) -> dict:
        """Solve the spectra and form the Gauss-Newton normal equations.

        The parameter step uses the Jacobian with the part the spectra can absorb projected out.

        Args:
            params: Parameter vector.

        Returns:
            dict: "params", "misfit", "penalty", "objective" (misfit + penalty), "model" (4096, 2048), "spectra",
            "gram" and "gradient" (both including the penalty) and "nnz" (model entries).
        """
        A, D_shift, D_width, D_power, D_wing, D_asymmetry = self.build(params)

        # Spectra: unweighted linear least squares over the finite pixels, as in make_parameter_matrix.fit()
        A_f = A.copy()
        A_f.data[~self.finite[A_f.indices]] = 0.0
        A_f.eliminate_zeros()
        AtA = (A_f @ A_f.T).tocsc()
        AtA += 1e-10 * speye(AtA.shape[0], format="csc")
        lu = splu(AtA)
        del AtA
        spectra = lu.solve(A_f @ self.data)
        model = A.T @ spectra
        residual = self.data - A_f.T @ spectra

        wing_amp = self.profile_shape(params)[1]
        if self.weights is None:
            self.weights = self._misfit_weights(A, spectra, model)
            self._set_prior(spectra, D_shift, D_width, D_wing, wing_amp, D_asymmetry)
        w = self.weights
        misfit = float(np.sum(w * residual**2))

        # Jacobian of the model w.r.t. all parameters at fixed spectra: (pixels x n_params)
        J = D_shift.T @ self._coefficient_map(spectra, 0) + D_width.T @ self._coefficient_map(spectra, self.n_coeff)
        if self.fit_profile:
            power_column = np.ones((N_ELEMENTS, 1))
            J = J + D_power.T @ self._shared_map(spectra, self.n_curve_params, power_column)
            # d wing_amp / d (log-wing coefficient k) = wing_amp * wing_basis_k, as wing_amp = exp(spline)
            J = J + D_wing.T @ self._shared_map(spectra, self.wing_offset, wing_amp[:, None] * self.wing_basis)
        if self.fit_asymmetry:
            J = J + D_asymmetry.T @ self._shared_map(spectra, self.asymmetry_offset, self.asymmetry_basis)
        J = J.tocsr()  # non-finite pixels drop out below: A_f has no entries there and their weight is 0
        del D_shift, D_width, D_power, D_wing, D_asymmetry

        # Project out what the spectra can absorb: J_perp = J - A_f.T X with X = (A_f A_f.T)^-1 A_f J, then form
        # J_perp.T W J_perp and J_perp.T W r without ever building the dense (pixels x n_params) J_perp.
        X = lu.solve((A_f @ J).toarray())
        del lu
        W = sparse.diags(w)
        J_w = (W @ J).tocsr()
        A_w = A_f @ W
        AwJw = (A_w @ J_w).toarray()
        r_w = w * residual
        gram = (J_w.T @ J_w).toarray() - AwJw.T @ X - X.T @ AwJw + X.T @ ((A_w @ A_w.T) @ X)
        gradient = J_w.T @ r_w - X.T @ (A_w @ r_w)

        penalty = self.penalty(params)
        return {
            "params": params,
            "misfit": misfit,
            "penalty": penalty,
            "objective": misfit + penalty,
            "model": model.reshape(4096, 2048),
            "spectra": spectra,
            "gram": gram + self.prior_matrix,
            "gradient": np.asarray(gradient).ravel() - self.prior_matrix @ params,
            "nnz": A.nnz,
        }

    def _group_rows(self, i: int) -> np.ndarray:
        return (self.fit_offset + i) * N_ELEMENTS + X0

    def _coefficient_map(self, spectra: np.ndarray, col_offset: int) -> sparse.csr_matrix:
        """(elements x params) map putting s[element] * basis_k(element) in each spaxel's coefficient columns.

        Args:
            spectra: Solved spectra.
            col_offset: 0 for the shift coefficients, n_coeff for the width coefficients.

        Returns:
            sparse.csr_matrix: The map.
        """
        rows, cols, vals = [], [], []
        for i in range(15):
            element_rows = self._group_rows(i)
            s = spectra[element_rows]
            for k in range(self.n_coeff):
                rows.append(element_rows)
                cols.append(np.full(N_ELEMENTS, i * 2 * self.n_coeff + col_offset + k))
                vals.append(s * self.basis[:, k])
        return sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_model * N_ELEMENTS, self.n_params),
        )

    def _shared_map(self, spectra: np.ndarray, first_col: int, basis: np.ndarray) -> sparse.csr_matrix:
        """(elements x params) map for parameters shared by the group.

        Args:
            spectra: Solved spectra.
            first_col: Column of the first of the shared parameters.
            basis: (1400, m) derivative of the per-element profile parameter with respect to each shared parameter.

        Returns:
            sparse.csr_matrix: The map, s[element] * basis[element, k] over all group spaxels' rows.
        """
        rows, cols, vals = [], [], []
        for i in range(15):
            element_rows = self._group_rows(i)
            s = spectra[element_rows]
            for k in range(basis.shape[1]):
                rows.append(element_rows)
                cols.append(np.full(N_ELEMENTS, first_col + k))
                vals.append(s * basis[:, k])
        return sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_model * N_ELEMENTS, self.n_params),
        )

    def _misfit_weights(self, A: sparse.csr_matrix, spectra: np.ndarray, model: np.ndarray) -> np.ndarray:
        """Pixels in the misfit: those whose starting model is mostly the group's own unmasked elements.

        Args:
            A: Model matrix.
            spectra: Solved spectra.
            model: Flattened model image.

        Returns:
            np.ndarray: 1 for pixels in the misfit, 0 otherwise.
        """
        group_unmasked = np.zeros_like(spectra)
        for i in range(15):
            rows = self._group_rows(i)
            group_unmasked[rows] = spectra[rows]
            if self.masked_elements is not None:
                group_unmasked[rows[slice(*self.masked_elements)]] = 0.0
        group_model = A.T @ group_unmasked
        keep = self.finite & (model > 0) & (group_model > 0.5 * model)
        return keep.astype(float)

    def _set_prior(
        self,
        spectra: np.ndarray,
        D_shift: sparse.csr_matrix,
        D_width: sparse.csr_matrix,
        D_wing: sparse.csr_matrix | None,
        wing_amp: np.ndarray,
        D_asymmetry: sparse.csr_matrix | None,
    ) -> None:
        """Build the flat-hold penalty matrix.

        Each curve's slope is penalised in proportion to the data's curvature and to how faint the element is:
        f = 1 / (1 + (signal / prior_threshold)**4), signal being the smoothed spectrum over its maximum. Shared
        curves use the group mean of f; the core power is not penalised.

        Args:
            spectra: Solved starting spectra.
            D_shift: Shift derivative matrix.
            D_width: Width derivative matrix.
            D_wing: Wing-amplitude derivative matrix, or None.
            wing_amp: Starting wing amplitude per element.
            D_asymmetry: Asymmetry derivative matrix, or None.
        """
        row_norm_shift = np.asarray(D_shift.multiply(D_shift).sum(axis=1)).ravel()
        row_norm_width = np.asarray(D_width.multiply(D_width).sum(axis=1)).ravel()
        self.prior_matrix = np.zeros((self.n_params, self.n_params))
        self.pinned = np.zeros((15, N_ELEMENTS))
        for i in range(15):
            rows = self._group_rows(i)
            s = spectra[rows]
            smooth = np.clip(uniform_filter1d(s, 25, mode="nearest"), 0, None)
            signal = smooth / max(smooth.max(), 1e-30)
            self.pinned[i] = 1 / (1 + (signal / self.prior_threshold) ** 4)
            block = slice(i * 2 * self.n_coeff, (i + 1) * 2 * self.n_coeff)
            blocks = []
            for row_norm in (row_norm_shift, row_norm_width):
                # Data curvature of each element's curve value: s^2 * |d model / d curve|^2
                stiffness = s**2 * row_norm[rows]
                alpha = self.prior_strength * np.percentile(stiffness, 90) * self.pinned[i]
                blocks.append(self.slope_basis.T @ (alpha[:, None] * self.slope_basis))
            self.prior_matrix[block, block] = np.block(
                [[blocks[0], np.zeros_like(blocks[0])], [np.zeros_like(blocks[1]), blocks[1]]]
            )
        if self.fit_profile:
            # log wing amplitude: d model / d(log wing) = wing * d model / d wing
            self._hold_shared_curve(spectra, D_wing, wing_amp, self.wing_offset, self.wing_slope_basis)
        if self.fit_asymmetry:
            self._hold_shared_curve(
                spectra, D_asymmetry, np.ones(N_ELEMENTS), self.asymmetry_offset, self.asymmetry_slope_basis
            )

    def _hold_shared_curve(
        self,
        spectra: np.ndarray,
        D: sparse.csr_matrix,
        chain: np.ndarray,
        first_col: int,
        slope_basis: np.ndarray,
    ) -> None:
        """Add the flat-hold penalty block of a curve shared by the group.

        Args:
            spectra: Solved starting spectra.
            D: Derivative matrix of the per-element profile parameter.
            chain: Per-element factor from that parameter to the curve value.
            first_col: Column of the curve's first coefficient.
            slope_basis: Slope basis of the curve's spline.
        """
        row_norm = np.asarray(D.multiply(D).sum(axis=1)).ravel()
        stiffness = sum(spectra[self._group_rows(i)] ** 2 * row_norm[self._group_rows(i)] for i in range(15)) * chain**2
        alpha = self.prior_strength * np.percentile(stiffness, 90) * self.pinned.mean(axis=0)
        block = slice(first_col, first_col + slope_basis.shape[1])
        self.prior_matrix[block, block] = slope_basis.T @ (alpha[:, None] * slope_basis)

    def pinned_ranges(self, i: int) -> str:
        """Element ranges where a group spaxel's curves are mostly held flat.

        Args:
            i: Position of the spaxel in the group.

        Returns:
            str: Ranges such as "0-372, 1340-1399", or "none".
        """
        held = self.pinned[i] > 0.5
        edges = np.flatnonzero(np.diff(np.concatenate([[0], held.astype(int), [0]])))
        return ", ".join(f"{a}-{b - 1}" for a, b in zip(edges[::2], edges[1::2], strict=False)) or "none"


def _write_jsons(out: Path, gn: GaussNewtonGroupFit, history: list[np.ndarray]) -> None:
    """Write the shift, width and profile histories of the final stage.

    Args:
        out: Output directory.
        gn: The stage's fit.
        history: Parameter vectors of the stage's start and each accepted iteration.
    """
    shifts: dict = {str(s): {} for s in gn.spaxels}
    widths: dict = {str(s): {} for s in gn.spaxels}
    profile = {}
    for iteration, params in enumerate(history):
        shift_entries, width_entries = gn.spline_entries(params)
        for i, s in enumerate(gn.spaxels):
            shifts[str(s)][str(iteration)] = shift_entries[i]
            widths[str(s)][str(iteration)] = width_entries[i]
        profile[str(iteration)] = gn.profile_entry(params)
    for name, contents in (
        ("gn_loop_shifts_editable.json", shifts),
        ("gn_loop_widths_editable.json", widths),
        ("gn_profile_editable.json", profile),
    ):
        with (out / name).open("w") as f:
            json.dump(contents, f, indent=2)
    logger.info(f"Wrote shift, width and profile JSONs to {out}")


def _start_profile(init_from: Path | None, asymmetry_mode: str) -> dict:
    """Starting profile shape from an earlier run.

    Args:
        init_from: Directory of an earlier run, or None.
        asymmetry_mode: Only a fitted asymmetry of this mode is reused.

    Returns:
        dict: initial_params keyword arguments; empty if there is no earlier profile.
    """
    path = Path(init_from) / "gn_profile_editable.json" if init_from is not None else None
    if path is None or not path.exists():
        return {}
    with path.open("r") as f:
        history = json.load(f)
    profile = history[str(max(int(k) for k in history))]
    log_wing = _latest_curve({"0": profile["log_wing_amp"]}, np.log(WING_AMP))
    kwargs = {"core_power": profile["core_power"], "wing_amp": np.exp(log_wing)}
    if "asymmetry" in profile and profile.get("asymmetry_mode", "core") == asymmetry_mode:
        kwargs["asymmetry"] = _latest_curve({"0": profile["asymmetry"]}, 0.0)
    return kwargs


def _load_combined_jsons(directory: Path) -> tuple[dict, dict]:
    """Shift and width JSONs from a Gauss-Newton (gn_loop_*) or scan (tester_loop_*) run.

    Args:
        directory: Directory holding them.

    Returns:
        tuple[dict, dict]: The shift and width JSON contents.
    """
    for prefix in ("gn_loop", "tester_loop"):
        shifts_path = directory / f"{prefix}_shifts_editable.json"
        widths_path = directory / f"{prefix}_widths_editable.json"
        if shifts_path.exists() and widths_path.exists():
            with shifts_path.open("r") as f:
                shifts = json.load(f)
            with widths_path.open("r") as f:
                widths = json.load(f)
            logger.info(f"Loaded {shifts_path} / {widths_path}")
            return shifts, widths
    raise FileNotFoundError(f"No gn_loop_* or tester_loop_* combined JSONs in {directory}")


def _step_size(gn: GaussNewtonGroupFit, old: np.ndarray, new: np.ndarray, tolerances: dict) -> tuple[str, bool]:
    """Size of a step.

    Args:
        gn: The stage's fit.
        old: Parameters before the step.
        new: Parameters after the step.
        tolerances: Convergence tolerances: "shift", "width", "power", "wing", "asymmetry".

    Returns:
        tuple[str, bool]: Log text, and whether every change is below its tolerance.
    """
    d_shift, d_width = gn.shift_and_width(new - old)
    (old_power, old_wing), (new_power, new_wing) = gn.profile_shape(old), gn.profile_shape(new)
    new_asymmetry = gn.asymmetry(new)
    sizes = {
        "shift": float(np.max(np.abs(d_shift))),
        "width": float(np.max(np.abs(d_width))),
        "power": abs(new_power - old_power),
        "wing": float(np.max(np.abs(new_wing - old_wing))),
        "asymmetry": float(np.max(np.abs(new_asymmetry - gn.asymmetry(old)))),
    }
    samples = (0, 350, 700, 1050, 1399)
    text = f"max |d shift| {sizes['shift']:.4f} px, max |d width| {sizes['width']:.4f}"
    if gn.fit_profile:
        wing_samples = "/".join(f"{new_wing[e]:.3f}" for e in samples)
        text += f", core power {new_power:.3f}, wing amp at elements 0/350/700/1050/1399 {wing_samples}"
    if gn.fit_asymmetry:
        text += ", asymmetry at the same elements " + "/".join(f"{new_asymmetry[e]:+.3f}" for e in samples)
    return text, all(sizes[k] < tolerances[k] for k in sizes)


def _damped_step(
    gn: GaussNewtonGroupFit,
    state: dict,
    damping: float,
    limits: dict,
    tolerances: dict,
    rel_tol: float,
    log_prefix: str,
) -> tuple[dict | None, float]:
    """One Levenberg-Marquardt step, raising the damping until a step lowers the objective.

    Steps outside the limits are not tried. A rejected step that is already below the tolerances, or changes the
    objective by less than rel_tol of it, means the fit has converged.

    Args:
        gn: The stage's fit.
        state: Current evaluate() result.
        damping: Starting damping.
        limits: "min_width", "max_shift_step", "max_power_step" and "max_asymmetry".
        tolerances: See _step_size.
        rel_tol: Relative objective change treated as no change.
        log_prefix: Prefix of log messages.

    Returns:
        tuple[dict | None, float]: The accepted step's evaluate() result (None if converged or no step helps) and
        the damping for the next iteration.
    """
    params, gram, gradient = state["params"], state["gram"], state["gradient"]
    diag = np.maximum(np.diag(gram), 1e-12 * np.max(np.diag(gram)))
    while damping < 1e8:
        step = np.linalg.solve(gram + damping * np.diag(diag), gradient)
        d_shift = gn.shift_and_width(step)[0]
        new_width = gn.shift_and_width(params + step)[1]
        old_power, new_power = gn.profile_shape(params)[0], gn.profile_shape(params + step)[0]
        new_asymmetry = float(np.max(np.abs(gn.asymmetry(params + step))))
        # A negative width gives NaN in pseudo_voigt; the caps keep single steps from running away
        if (
            not np.all(np.isfinite(step))
            or np.min(new_width) < limits["min_width"]
            or np.max(np.abs(d_shift)) > limits["max_shift_step"]
            or not 1.5 <= new_power <= 12
            or abs(new_power - old_power) > limits["max_power_step"]
            or new_asymmetry > limits["max_asymmetry"]
        ):
            logger.info(
                f"{log_prefix}: step skipped (min width {np.min(new_width):.3f}, "
                f"max |d shift| {np.max(np.abs(d_shift)):.3f} px, core power {new_power:.3f}, "
                f"max |asymmetry| {new_asymmetry:.3f}), damping -> {damping * 10:.0e}"
            )
            damping *= 10
            continue
        trial = gn.evaluate(params + step)
        if trial["objective"] < state["objective"]:
            return trial, max(damping / 10, 1e-7)
        tiny = _step_size(gn, params, params + step, tolerances)[1]
        if tiny or trial["objective"] - state["objective"] < rel_tol * state["objective"]:
            logger.info(f"{log_prefix}: step rejected but already below tolerance; converged")
            return None, damping
        logger.info(
            f"{log_prefix}: step rejected (objective {trial['objective']:.6g} >= {state['objective']:.6g}), "
            f"damping -> {damping * 10:.0e}"
        )
        damping *= 10
    return None, damping


def _run_stage(
    gn: GaussNewtonGroupFit,
    params: np.ndarray,
    iteration_max: int,
    limits: dict,
    tolerances: dict,
    rel_tol: float,
    label: str,
) -> tuple[dict, list[np.ndarray], np.ndarray]:
    """Fit one stage to convergence.

    Args:
        gn: The stage's fit.
        params: Starting parameters.
        iteration_max: Maximum accepted steps.
        limits: See _damped_step.
        tolerances: See _step_size.
        rel_tol: Relative objective improvement treated as converged.
        label: Log label, e.g. "[group 9 stage 1]".

    Returns:
        tuple[dict, list[np.ndarray], np.ndarray]: The final evaluate() result, the parameters of the start and of
        each accepted step, and the starting model image.
    """
    t0 = time.time()
    state = gn.evaluate(params)
    logger.info(
        f"{label} start: misfit {state['misfit']:.6g}, {state['nnz']:,} model entries, {time.time() - t0:.0f}s | "
        f"peak RSS {_peak_memory_mb():.0f} MB"
    )
    history = [params]
    start_model = state["model"]
    damping = 1e-3
    for iteration in range(1, iteration_max + 1):
        t0 = time.time()
        trial, damping = _damped_step(gn, state, damping, limits, tolerances, rel_tol, f"{label} iter {iteration}")
        if trial is None:
            break
        step_text, below_tolerances = _step_size(gn, state["params"], trial["params"], tolerances)
        improvement = (state["objective"] - trial["objective"]) / state["objective"]
        state = trial
        history.append(state["params"])
        logger.info(
            f"{label} iter {iteration}: misfit {state['misfit']:.6g}, penalty {state['penalty']:.4g}, {step_text}, "
            f"damping {damping:.0e}, {time.time() - t0:.0f}s | peak RSS {_peak_memory_mb():.0f} MB"
        )
        if below_tolerances or improvement < rel_tol:
            logger.info(f"{label} converged after {iteration} iterations")
            break
    return state, history, start_model


def _report(
    gn: GaussNewtonGroupFit,
    state: dict,
    science_image: np.ndarray,
    initial_model: np.ndarray,
    out: Path,
    group: int,
    save_model: bool,
) -> None:
    """Save the final model, plot every group spaxel and log its residual metrics.

    Args:
        gn: The final stage's fit.
        state: Its final evaluate() result.
        science_image: The fitted image.
        initial_model: Model before the fit, for the plots.
        out: Output directory.
        group: Group index.
        save_model: Whether to write the model as FITS.
    """
    final_model = state["model"]
    if save_model:
        fits.PrimaryHDU(final_model).writeto(out / f"gn_model_group_{group}.fits", overwrite=True)

    # Trace centres from the fitted shifts (the padding spaxels' fixed ones, nominal for the rest)
    shift_curves = {s: curve for s, (curve, _) in gn.fixed_curves.items()}
    shift_curves.update(zip(gn.spaxels, gn.shift_and_width(state["params"])[0], strict=True))
    centres = trace_centres(shift_curves)
    noise = noise_model(science_image, final_model, region=gn.weights.reshape(final_model.shape) > 0)
    logger.info(f"[group {group}] pixel noise sigma^2 = {noise[0]:.4g} + {noise[1]:.4g} flux + {noise[2]:.4g} flux^2")
    for spaxel in gn.spaxels:
        plot_fit_difference(
            science_image,
            initial_model,
            final_model,
            spaxel,
            out / f"gn_fit_difference_spaxel_{spaxel}.png",
            centres=centres,
            noise=noise,
        )
        old = residual_stats(science_image, final_model, spaxel)
        new = residual_stats(science_image, final_model, spaxel, centres=centres, noise=noise)
        logger.info(
            f"[spaxel {spaxel}] own pixels, |resid|/own row peak: median {new['median']:.2%}, p90 {new['p90']:.1%}, "
            f"<1% {new['under_1pct']:.0%} (noise ceiling {new['noise_ceiling']:.0%}, "
            f"noise-limited pixels {new['noise_limited']:.0%}) | old +-50 column crop metric: "
            f"p90 {old['p90']:.1%}, <1% {old['under_1pct']:.0%}"
        )


def _parse_element_range(text: str) -> tuple[int, int] | None:
    """Parse a spectral element range.

    Args:
        text: "start:stop", or "none".

    Returns:
        tuple[int, int] | None: (start, stop), or None for "none".
    """
    if text.strip().lower() == "none":
        return None
    start, stop = (int(v) for v in text.split(":"))
    return start, stop


@pipeline_flow()
def gauss_newton_make_parameter_matrix(
    group: int = 9,
    iteration_max: int = 10,
    output_dir: Path | None = None,
    science_image_path: Path | str = "/global/homes/a/anousha/deep_skyflat_coadd.fits",
    oversample_factor: int = 1,
    segments: int = 8,
    workers: int = 1,
    init_from: Path | None = None,
    neighbors_from: Path | None = None,
    pad: int = 7,
    prior_threshold: float = 0.03,
    prior_strength: float = 1.0,
    shift_tol: float = 0.005,
    width_tol: float = 0.005,
    rel_tol: float = 1e-6,
    min_width: float = 0.2,
    max_shift_step: float = 2.0,
    max_power_step: float = 0.5,
    wing_segments: int = 3,
    power_tol: float = 0.01,
    wing_tol: float = 0.002,
    fit_asymmetry: bool = False,
    asymmetry_segments: int = 3,
    asymmetry_mode: str = "shoulder",
    asymmetry_tol: float = 0.002,
    max_asymmetry: float = 0.5,
    masked_elements: tuple[int, int] | None = MASKED_ELEMENTS,
    save_model: bool = False,
    hostname: str | None = None,
    git_commit: str | None = None,
) -> None:
    """Fit a group in two stages and write the second stage's results.

    Stage 1 fits the shift and width curves; stage 2 starts from them and also fits the core power, the wing
    amplitude and, with fit_asymmetry, the asymmetry. Only stage 2 writes JSONs, plots and the model.

    Args:
        group: Group index (spaxels 15 * group to 15 * group + 14).
        iteration_max: Maximum accepted steps per stage.
        output_dir: Output directory (default the working directory).
        science_image_path: FITS image to fit.
        oversample_factor: Sub-samples per pixel.
        segments: Spline intervals of the shift and width curves.
        workers: Processes building spaxels in parallel.
        init_from: Earlier run to start from (curves, and the profile for stage 2), or None.
        neighbors_from: Run holding the padding spaxels' curves (default init_from).
        pad: Spaxels modelled on each side of the group.
        prior_threshold: Signal fraction below which curves are held flat.
        prior_strength: Strength of that hold (0 = off).
        shift_tol: Convergence tolerance of the shift (px).
        width_tol: Convergence tolerance of the width.
        rel_tol: Relative objective improvement treated as converged.
        min_width: Smallest width a step may leave.
        max_shift_step: Largest shift change of one step (px).
        max_power_step: Largest core-power change of one step.
        wing_segments: Spline intervals of the wing amplitude.
        power_tol: Convergence tolerance of the core power.
        wing_tol: Convergence tolerance of the wing amplitude.
        fit_asymmetry: Whether stage 2 also fits the profile asymmetry.
        asymmetry_segments: Spline intervals of the asymmetry.
        asymmetry_mode: See _binned_profile.
        asymmetry_tol: Convergence tolerance of the asymmetry.
        max_asymmetry: Largest asymmetry a step may leave.
        masked_elements: (start, stop) spectral elements left out of the misfit, or None.
        save_model: Whether to write the final model as FITS.
        hostname: Host name, for the log.
        git_commit: Code version, for the log.
    """
    logger.info(f"[group {group}] Gauss-Newton fit | host={hostname} git_commit={git_commit}")
    logger.info(f"[group {group}] spectral elements left out of the misfit: {masked_elements or 'none'}")
    science_image = _load_science_image(science_image_path)
    out = output_dir if output_dir is not None else Path.cwd()
    out.mkdir(parents=True, exist_ok=True)

    start_jsons = _load_combined_jsons(Path(init_from)) if init_from is not None else ({}, {})
    neighbor_jsons = _load_combined_jsons(Path(neighbors_from)) if neighbors_from is not None else start_jsons
    group_spaxels = list(range(15 * group, 15 * group + 15))
    model_spaxels = list(range(max(0, 15 * group - pad), min(225, 15 * group + 15 + pad)))
    tolerances = {
        "shift": shift_tol,
        "width": width_tol,
        "power": power_tol,
        "wing": wing_tol,
        "asymmetry": asymmetry_tol,
    }
    limits = {
        "min_width": min_width,
        "max_shift_step": max_shift_step,
        "max_power_step": max_power_step,
        "max_asymmetry": max_asymmetry,
    }

    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        common = {
            "segments": segments,
            "oversample_factor": oversample_factor,
            "pool": pool,
            "pad": pad,
            "fixed_curves": curves_from_jsons(model_spaxels, *neighbor_jsons),
            "prior_threshold": prior_threshold,
            "prior_strength": prior_strength,
            "wing_segments": wing_segments,
            "asymmetry_segments": asymmetry_segments,
            "asymmetry_mode": asymmetry_mode,
            "masked_elements": masked_elements,
        }
        stage1 = GaussNewtonGroupFit(science_image, group, fit_profile=False, fit_asymmetry=False, **common)
        fitted_neighbors = [s for s in stage1.padding if str(s) in neighbor_jsons[0]]
        logger.info(
            f"[group {group}] model spaxels {stage1.model_spaxels[0]}-{stage1.model_spaxels[-1]}; padding with fitted "
            f"curves: {fitted_neighbors or 'none'} (others uncorrected)"
        )
        params = stage1.initial_params(curves_from_jsons(group_spaxels, *start_jsons))
        state, _, initial_model = _run_stage(
            stage1, params, iteration_max, limits, tolerances, rel_tol, f"[group {group} stage 1]"
        )
        for i, sid in enumerate(stage1.spaxels):
            logger.info(f"[spaxel {sid}] curves held flat (low signal) over elements: {stage1.pinned_ranges(i)}")

        stage2 = GaussNewtonGroupFit(science_image, group, fit_profile=True, fit_asymmetry=fit_asymmetry, **common)
        stage2.padding_parts = stage1.padding_parts
        params = stage2.initial_params(stage1.curves(state["params"]), **_start_profile(init_from, asymmetry_mode))
        state, history, _ = _run_stage(
            stage2, params, iteration_max, limits, tolerances, rel_tol, f"[group {group} stage 2]"
        )
    finally:
        if pool is not None:
            pool.shutdown()

    _write_jsons(out, stage2, history)
    _report(stage2, state, science_image, initial_model, out, group, save_model)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", type=int, default=9, help="Spaxel group to fit (e.g. 9 for spaxels 135-149).")
    parser.add_argument("--iteration-max", type=int, default=10, help="Maximum accepted steps per stage.")
    parser.add_argument("--science-image", type=str, default="/home/anousha/snifs-pipeline/deep_skyflat_coadd.fits")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to write JSONs and plots (default: cwd)")
    parser.add_argument("--oversample", type=int, default=1)
    parser.add_argument("--segments", type=int, default=8, help="Spline intervals of the shift and width curves.")
    parser.add_argument("--workers", type=int, default=1, help="Processes building spaxels in parallel.")
    parser.add_argument("--init-from", type=str, default=None, help="Earlier run to start from.")
    parser.add_argument(
        "--neighbors-from",
        type=str,
        default=None,
        help="Run holding the padding spaxels' curves (default: --init-from).",
    )
    parser.add_argument("--pad", type=int, default=7, help="Spaxels modelled on each side of the group.")
    parser.add_argument(
        "--prior-threshold", type=float, default=0.03, help="Signal fraction below which curves are held flat."
    )
    parser.add_argument("--prior-strength", type=float, default=1.0, help="Strength of the flat hold (0 = off).")
    parser.add_argument("--wing-segments", type=int, default=3, help="Spline intervals of the wing amplitude.")
    parser.add_argument(
        "--fit-asymmetry", action=argparse.BooleanOptionalAction, default=False, help="Also fit the profile asymmetry."
    )
    parser.add_argument("--asymmetry-segments", type=int, default=3, help="Spline intervals of the asymmetry.")
    parser.add_argument(
        "--asymmetry-mode",
        choices=["shoulder", "core"],
        default="shoulder",
        help="shoulder: only the Lorentzian width differs between the sides; core: the Gaussian width too.",
    )
    parser.add_argument(
        "--mask-elements",
        default=f"{MASKED_ELEMENTS[0]}:{MASKED_ELEMENTS[1]}",
        help="Spectral elements start:stop left out of the misfit, or 'none'.",
    )
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    gauss_newton_make_parameter_matrix(
        group=args.group,
        iteration_max=args.iteration_max,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        science_image_path=args.science_image,
        oversample_factor=args.oversample,
        segments=args.segments,
        workers=args.workers,
        init_from=Path(args.init_from) if args.init_from else None,
        neighbors_from=Path(args.neighbors_from) if args.neighbors_from else None,
        pad=args.pad,
        prior_threshold=args.prior_threshold,
        prior_strength=args.prior_strength,
        wing_segments=args.wing_segments,
        fit_asymmetry=args.fit_asymmetry,
        asymmetry_segments=args.asymmetry_segments,
        asymmetry_mode=args.asymmetry_mode,
        masked_elements=_parse_element_range(args.mask_elements),
        save_model=args.save_model,
        hostname=socket.gethostname(),
        git_commit=_git_commit(),
    )
