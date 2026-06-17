import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from astropy.io import fits
from numpy.polynomial import Polynomial
from PIL import Image as PILImage
from scipy import sparse
from scipy.interpolate import interp1d  # noqa: F401
from scipy.sparse import eye as speye
from scipy.sparse.linalg import spsolve

from pipeline.common.fitting_math__utils import pseudo_voigt
from pipeline.common.image import Image
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
from pipeline.common.prefect_utils import create_image_artifact, create_markdown_artifact, pipeline_flow, pipeline_task
from pipeline.tasks.plotting.wavelength_arc_calibration_plots import animate_poly_convergence

logger = get_logger()
superStart = time.time()
# global offsets to the spectrum taken from the cross-correlation of the arc with a reference arc.
xoff = 0
yoff = 0

total_coeffs = {str(spax): {"0": [0.0, 0.0, 0.0, 0.0, 0.0]} for spax in range(225)}
total_widths = {str(spax): {"0": [0.0, 0.0, 0.0, 0.0, 1.0]} for spax in range(225)}

if os.path.exists("loop_shifts_editable.json") and os.path.exists("loop_widths_editable.json"):
    with open("loop_shifts_editable.json", "r") as f:
        total_coeffs = json.load(f)
    with open("loop_widths_editable.json", "r") as f:
        total_widths = json.load(f)
    logger.info("Adding to existing loop_shifts_editable.json / loop_widths_editable.json")

offsets_cross_disp = np.zeros(225)
offsets_cross_disp[~np.isfinite(offsets_cross_disp)] = 0.0  # fix for nans in the array

n_cross = (np.log(0.0032) - np.log(6.90e-4)) / (np.log(4.917) - np.log(26.939))
n_spec = (np.log(0.0040) - np.log(8.447e-4)) / (np.log(4.52) - np.log(21.026))

start_time = time.time()

rowindex, columnindex = np.meshgrid(np.arange(0, 2048, 1), np.arange(0, 4096, 1))

spec: np.ndarray = np.ones((225, 1400))
heights: np.ndarray = np.linspace(0, 4095, 256)
n_bins: int = len(heights) - 1
x_sparse: np.ndarray = np.linspace(0, 4095, 256) + 8
science_image: np.ndarray | None = None
params: list[float] = []


def combine_spaxel_jsons(output_dir: Path) -> None:
    """Merge all per-spaxel shift/width JSONs in output_dir into combined files.

    Each job writes loop_shifts_spaxel_N.json and loop_widths_spaxel_N.json.
    Call this after all jobs finish to produce loop_shifts_editable.json and
    loop_widths_editable.json containing every spaxel.
    """
    combined_shifts: dict = {}
    combined_widths: dict = {}
    for path in sorted(output_dir.glob("loop_shifts_spaxel_*.json")):
        with path.open("r") as f:
            combined_shifts.update(json.load(f))
    for path in sorted(output_dir.glob("loop_widths_spaxel_*.json")):
        with path.open("r") as f:
            combined_widths.update(json.load(f))
    with (output_dir / "loop_shifts_editable.json").open("w") as f:
        json.dump(combined_shifts, f, indent=2)
    with (output_dir / "loop_widths_editable.json").open("w") as f:
        json.dump(combined_widths, f, indent=2)


def stat_l1(sci: np.ndarray, mod: np.ndarray, sl: tuple) -> float:
    return float(np.nansum(np.abs(sci[sl] - mod[sl])))


def _build_spaxel_rows(
    spaxel_IDs: list[int],
    row_start: int,
    offset_per_id: dict[int, float],
    width_per_id: dict[int, float],
    oversample_factor: int,
    use_total_coeffs_for: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build COO triplets for a subset of spaxel rows.

    offset_per_id / width_per_id: scalar offset/width adjustment per spaxel_ID.
    use_total_coeffs_for: if not None, apply total_coeffs/total_widths lookup for that spaxel_ID.
    Returns (data, rows, cols) ready for sparse.csr_matrix construction.
    """
    all_data: list = []
    all_rows: list = []
    all_cols: list = []

    for spaxel_ID in spaxel_IDs:
        a0 = int(A0_PARAMS[spaxel_ID] + yoff)
        a1 = int(A1_PARAMS[spaxel_ID] + yoff) + 1
        b0 = int(B0_PARAMS[spaxel_ID] + xoff - 50)
        off = 50
        if b0 < 0:
            off += b0
            b0 = 0
        b1 = int(B1_PARAMS[spaxel_ID] + xoff + 50) + 1

        xsub = np.linspace(0, a1 - a0 - 1, (a1 - a0) * oversample_factor)
        ysub = np.linspace(0, b1 - b0 - 1, (b1 - b0) * oversample_factor)
        xv_sub, yv_sub = np.meshgrid(ysub, xsub)

        rowind_slab = rowindex[a0:a1, b0:b1]
        colind_slab = columnindex[a0:a1, b0:b1]

        x0 = np.arange(0, 1400, 1)

        p = Polynomial([QUARTIC_LINEAR_PARAMS[spaxel_ID], Z_1ST[spaxel_ID], Z_2ND[spaxel_ID], 0, Z_4TH[spaxel_ID]])
        adjustmentP = Polynomial([offset_per_id.get(spaxel_ID, 0.0)])
        curve = p(x0) + adjustmentP(x0) + off

        if use_total_coeffs_for == spaxel_ID:
            latest_s = str(max(int(k) for k in total_coeffs[str(spaxel_ID)].keys()))
            curve = curve + np.poly1d(total_coeffs[str(spaxel_ID)][latest_s])(x0)

        widthAdjustmentP = Polynomial([width_per_id.get(spaxel_ID, 0.0)])
        widthVals = widthAdjustmentP(x0)

        if use_total_coeffs_for == spaxel_ID:
            latest_w = str(max(int(k) for k in total_widths[str(spaxel_ID)].keys()))
            widthVals += np.poly1d(total_widths[str(spaxel_ID)][latest_w])(x0)

        local_spaxel_idx = spaxel_ID - row_start

        for spec_element in range(0, 1400):
            c0 = int(spec_element - 50) * oversample_factor
            if c0 < 0:
                c0 = 0
            c1 = int(spec_element + 50) * oversample_factor
            if c1 >= 1400 * oversample_factor:
                c1 = int(1400 * oversample_factor)

            d0 = int(round((curve[spec_element] - 50))) * oversample_factor
            if d0 < 0:
                d0 = 0
            d1 = int(round((curve[spec_element] + 50))) * oversample_factor
            if d1 >= 1400 * oversample_factor:
                d1 = int(1400 * oversample_factor) - 1

            xv_sub_mono = xv_sub[c0:c1, d0:d1]
            yv_sub_mono = yv_sub[c0:c1, d0:d1]

            y = yv_sub_mono.T[0] - spec_element
            x = xv_sub_mono[0] - curve[spec_element]

            spec_trace = QUAD_SPEC[spaxel_ID] * y**2 + LIN_SPEC[spaxel_ID] * y
            cross_trace = LIN_CROSS[spaxel_ID] * x

            xiii = (xv_sub_mono - curve[spec_element]).T - spec_trace
            yiii = (yv_sub_mono - spec_element) - cross_trace

            mask_footprint = (
                np.sqrt(
                    (yv_sub_mono - spec_element - ELL_C1[spaxel_ID]) ** 2 / ELL_B[spaxel_ID] ** 2
                    + (xv_sub_mono - curve[spec_element] - ELL_C0[spaxel_ID]) ** 2 / ELL_A[spaxel_ID] ** 2
                )
                < 1
            )

            spectral = 0.8 * pseudo_voigt(
                np.abs(xiii), 0, 0.6 * widthVals[spec_element], 1.3 * widthVals[spec_element], 5.4, 0.6
            ) + pseudo_voigt(np.abs(xiii), 0, 1.2, 0.2, -n_spec, 0.1, beta=0)
            crossdis = 0.99 * pseudo_voigt(
                np.abs(yiii), 0, 0.6 * widthVals[spec_element], 1.4 * widthVals[spec_element], 5.2, 0.6
            ) + pseudo_voigt(np.abs(yiii), 0, 1.2, 0.1, -n_cross, 0.1, beta=0, l_off=10)

            model = spectral * crossdis.T * (mask_footprint.T * 1.0)
            model = model / np.max(model)

            H = model.shape[0] // oversample_factor
            W = model.shape[1] // oversample_factor
            testModel = (
                model[: H * oversample_factor, : W * oversample_factor]
                .reshape(H, oversample_factor, W, oversample_factor)
                .sum(axis=(1, 3))
            )

            c0s = c0 // oversample_factor
            c1s = c1 // oversample_factor
            d0s = d0 // oversample_factor
            d1s = d1 // oversample_factor

            mask_val = testModel.T > 1e-4
            rowind = rowind_slab[c0s:c1s, d0s:d1s]
            colind = colind_slab[c0s:c1s, d0s:d1s]
            mv = mask_val[: len(rowind), : len(rowind.T)]
            matrix_row = local_spaxel_idx * 1400 + spec_element
            data = testModel.T[: len(rowind), : len(rowind.T)][mv]

            all_data.append(data)
            all_rows.append(np.full(len(data), matrix_row, dtype=np.int32))
            all_cols.append(colind[mv] * 2048 + rowind[mv])

    return np.concatenate(all_data), np.concatenate(all_rows), np.concatenate(all_cols)


@pipeline_task()
def makeShiftedMat_neighbors(spaxel: int, oversample_factor: int = 1, iteration: int = 0):
    """Build matrix rows for the 14 neighbors of spaxel (no offset/width perturbation).

    This result is identical for every param value in an iteration, so compute it once.
    """
    row_start = 15 * (spaxel // 15)
    row_end = 15 * (spaxel // 15 + 1)
    neighbor_ids = [sid for sid in range(row_start, row_end) if sid != spaxel]

    data, rows, cols = _build_spaxel_rows(
        neighbor_ids, row_start,
        offset_per_id={}, width_per_id={},
        oversample_factor=oversample_factor,
        use_total_coeffs_for=None,
    )
    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(15 * 1400, 2048 * 4096),
    )


@pipeline_task()
def makeShiftedMat_target(
    spaxel: int, offset: float, width: float, oversample_factor: int = 1, iteration: int = 0
):
    """Build the single matrix row for the target spaxel with a given offset/width perturbation."""
    row_start = 15 * (spaxel // 15)

    data, rows, cols = _build_spaxel_rows(
        [spaxel], row_start,
        offset_per_id={spaxel: offset},
        width_per_id={spaxel: width},
        oversample_factor=oversample_factor,
        use_total_coeffs_for=spaxel,
    )
    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(15 * 1400, 2048 * 4096),
    )


@pipeline_task()
def combine_sparse_matrices(neighbor_mat, target_mat):
    """Add neighbor and target sparse matrices (they have non-overlapping rows)."""
    return neighbor_mat + target_mat


@pipeline_task()
def makeShiftedMat(
    spaxel: int, offsets: np.ndarray, widths: np.ndarray, oversample_factor: int = 1, iteration: int = 0
):
    """Original combined builder — kept for backward compatibility with final fit calls."""
    row_start = 15 * (spaxel // 15)
    row_end = 15 * (spaxel // 15 + 1)

    data, rows, cols = _build_spaxel_rows(
        list(range(row_start, row_end)), row_start,
        offset_per_id={sid: float(offsets[sid]) for sid in range(row_start, row_end)},
        width_per_id={sid: float(widths[sid]) for sid in range(row_start, row_end)},
        oversample_factor=oversample_factor,
        use_total_coeffs_for=spaxel,
    )
    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(15 * 1400, 2048 * 4096),
    )


@pipeline_task()
def fit(matrix, image, spectra, worker="main", spaxel=0, iteration=0, param=0.0):
    logger.info(f"[{worker}]   fitting model for spaxel={spaxel}, iteration={iteration}, param={param}...")
    matrix = matrix.transpose()
    spectra = sparse.csr_matrix(spectra).transpose()

    shifted_image = matrix.dot(spectra).reshape((4096, 2048)).todense()

    flag = (shifted_image > 0.0) & np.isfinite(image)
    imagea = np.where(flag, image, 0.0)

    fl = np.array(imagea.flatten().transpose().flat)
    assert np.all(np.isfinite(fl)), "b contains NaN or Inf!"

    AtA = matrix.T.dot(matrix).tocsc()
    Atb = matrix.T.dot(fl)
    # small regularization guards against singular columns (zero-contribution elements)
    AtA += 1e-10 * speye(AtA.shape[0], format="csc")
    x = spsolve(AtA, Atb)

    fitModel = matrix.dot(x).reshape((4096, 2048))

    if param == 0.0:
        hdu = fits.PrimaryHDU(fitModel)
        fits_filename = f"spaxel_{spaxel}_iteration_{iteration}.fits"
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(fits_filename, overwrite=True)
        hdulist.close()

        model_arr = np.asarray(fitModel)
        vmin, vmax = np.nanpercentile(model_arr, [1, 99])
        normalized = np.clip((model_arr - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        buf = BytesIO()
        PILImage.fromarray(normalized).save(buf, format="WEBP")
        b64 = base64.b64encode(buf.getvalue()).decode()
        create_image_artifact(
            image_url=f"data:image/webp;base64,{b64}",
            description=f"spaxel={spaxel}, iteration={iteration}",
            key=f"model-spaxel-{spaxel}-iteration-{iteration}",
        )

        zoom = np.s_[700:3000, 970:1100]
        frob = float(np.nansum(np.asarray(fitModel)[zoom] * np.asarray(image)[zoom]))
        create_markdown_artifact(
            markdown=f"**Frobenius product (zoom)** spaxel={spaxel}, iteration={iteration}\n\n`{frob:.6g}`",
            key=f"frob-spaxel-{spaxel}-iteration-{iteration}",
            description=f"Frobenius inner product of model vs science in zoom region, spaxel={spaxel} iter={iteration}",
        )

    return fitModel


# TODO: CHANGE OVERSAMPLE FACTOR BACK TO 4


def fit_best_shift(shifts, values, degree=4):
    """Fit a degree-N polynomial and return the shift at the minimum."""
    coeffs = np.polyfit(shifts, values, degree)
    poly = np.poly1d(coeffs)
    x_fine = np.linspace(shifts.min(), shifts.max(), 2000)
    return x_fine[np.argmin(poly(x_fine))]


@pipeline_task()
def compute_bin_stats(model: np.ndarray, spax: int) -> np.ndarray:
    """Compute per-bin L1 statistics for one (spax, param) result."""
    bin_stats = np.full(n_bins, np.nan)
    for bin_idx in range(n_bins):
        sl = (slice(int(heights[bin_idx]), int(heights[bin_idx + 1])), slice(None))
        try:
            bin_stats[bin_idx] = stat_l1(science_image, model, sl)  # type: ignore
        except Exception:
            pass
    return bin_stats


def psf_calculation(sum_along_cross_disp: np.ndarray) -> np.ndarray:
    """Calculate the psf.

    Args:
        sum_along_cross_disp: numpy array of summed cross-dispersion profile for one spec element

    Returns:
        line profile function: adjusted cross-dispersion profile of length 100
    """
    if len(sum_along_cross_disp) != 100:
        toAppend = np.zeros(100)
        toAppend[: len(sum_along_cross_disp)] = sum_along_cross_disp
        return toAppend
    return sum_along_cross_disp


@pipeline_task()
def l1_calculations(spaxels_to_process, flat_results, n_params, iteration, is_offset_iter):
    for spax_idx, spax in enumerate(spaxels_to_process):
        bin_stats_per_param = np.array(
            flat_results[spax_idx * n_params : (spax_idx + 1) * n_params]
        ).T  # shape: (n_bins, n_params)

        results_L1 = np.full(n_bins, np.nan)
        for bin_idx in range(n_bins):
            vals = bin_stats_per_param[bin_idx]
            if np.sum(np.isfinite(vals)) >= 4:
                results_L1[bin_idx] = fit_best_shift(np.array(params), vals)

        a0 = int(A0_PARAMS[spax] + yoff)
        a1 = int(A1_PARAMS[spax] + yoff) + 1

        f2 = interp1d(x_sparse[a0 // 16 - 2 : a1 // 16 + 2], results_L1[a0 // 16 - 2 : a1 // 16 + 2], kind="nearest")
        x_dense = np.arange(a0, a1 - 1)
        L1_dense = np.array(f2(x_dense))

        fittable = np.copy(L1_dense)
        fittable[220:540] = np.nan
        x_fittable = np.arange(len(fittable))
        mask_fin = np.isfinite(fittable)
        if mask_fin.sum() < 5:
            print(f"Spaxel {spax}: insufficient finite values for polyfit, skipping", flush=True)
            continue
        coeffs = np.polyfit(x_fittable[mask_fin], fittable[mask_fin], 4)

        print(f"Spaxel {spax}, L1 Coefficients: {coeffs}")

        new_coeffs = coeffs.tolist()
        prev_key = str(max(0, iteration - 1))
        if is_offset_iter:
            cumulative = np.polyadd(total_coeffs[str(spax)][prev_key], new_coeffs).tolist()
            total_coeffs[str(spax)][str(iteration + 1)] = cumulative
            print(f"Updated total_coeffs for spaxel {spax}: {total_coeffs[str(spax)]}", flush=True)
        else:
            cumulative = np.polyadd(total_widths[str(spax)][prev_key], new_coeffs).tolist()
            total_widths[str(spax)][str(iteration + 1)] = cumulative
            print(f"Updated total_widths for spaxel {spax}: {total_widths[str(spax)]}", flush=True)


def _load_spaxel_jsons(spaxels: list[int], out: Path) -> None:
    for spax in spaxels:
        shifts_path = out / f"loop_shifts_spaxel_{spax}.json"
        widths_path = out / f"loop_widths_spaxel_{spax}.json"
        if shifts_path.exists():
            with shifts_path.open("r") as f:
                total_coeffs[str(spax)] = json.load(f)[str(spax)]
        if widths_path.exists():
            with widths_path.open("r") as f:
                total_widths[str(spax)] = json.load(f)[str(spax)]


def _save_spaxel_jsons(spaxels: list[int], out: Path, iteration: int) -> None:
    for spax in spaxels:
        with (out / f"loop_shifts_spaxel_{spax}.json").open("w") as f:
            json.dump({str(spax): total_coeffs[str(spax)]}, f)
        with (out / f"loop_widths_spaxel_{spax}.json").open("w") as f:
            json.dump({str(spax): total_widths[str(spax)]}, f)
    logger.info(f"Saved per-spaxel JSONs after iteration {iteration}")


def _run_one_iteration(
    spaxels: list[int], iteration: int, is_offset_iter: bool, iter_params: list[float]
) -> list:
    flat_results = []
    for spax in spaxels:
        spectra = np.concatenate(spec[15 * (spax // 15) : 15 * (spax // 15 + 1)])
        spax_futures = []
        for param in iter_params:
            offsets = np.zeros(225)
            widths = np.zeros(225)
            if is_offset_iter:
                offsets[spax] = param
            else:
                widths[spax] = param
            mat_future = makeShiftedMat.submit(spax, offsets, widths, oversample_factor=4, iteration=iteration)
            fit_future = fit.submit(
                mat_future, science_image, spectra, spaxel=spax, iteration=iteration, param=param
            )
            spax_futures.append(compute_bin_stats.submit(fit_future, spax))
        flat_results.extend(f.result() for f in spax_futures)
    return flat_results




@pipeline_flow()
def make_parameter_matrix_old(
    spaxels_to_process: list[int] | None = None,
    iteration_max: int = 10,
    output_dir: Path | None = None,
):
    shift_offsets = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2]
    width_multipliers = [-0.2, -0.1, 0, 0.1, 0.2, 0.3]

    global science_image, params
    with fits.open("/home/anousha/SNIFS-model/refs/deep_skyflat_coadd.fits") as hdul:
        science_image = hdul[0].data  # type:ignore

    spaxels_to_process = spaxels_to_process if spaxels_to_process is not None else [8]
    out = output_dir if output_dir is not None else Path.cwd()
    out.mkdir(parents=True, exist_ok=True)

    _load_spaxel_jsons(spaxels_to_process, out)

    iteration = 0
    while iteration < iteration_max:
        is_offset_iter = iteration % 2 == 0
        params = shift_offsets if is_offset_iter else width_multipliers
        n_params = len(params)
        logger.info(
            f"iteration {iteration}, {'offsets' if is_offset_iter else 'widths'}, "
            f"{len(spaxels_to_process) * n_params} tasks → parallel"
        )
        flat_results = _run_one_iteration_parallel(spaxels_to_process, iteration, is_offset_iter, params)
        l1_calculations(spaxels_to_process, flat_results, n_params, iteration, is_offset_iter)
        logger.info("L1 calculations complete")
        iteration += 1
        _save_spaxel_jsons(spaxels_to_process, out, iteration)

    for spax in spaxels_to_process:
        create_markdown_artifact(
            markdown=(
                f"**Per-spaxel coefficients (spaxel {spax})**\n\n"
                f"shifts: `{out / f'loop_shifts_spaxel_{spax}.json'}`\n\n"
                f"widths: `{out / f'loop_widths_spaxel_{spax}.json'}`"
            ),
            key=f"wavelength-cal-spaxel-{spax}",
            description=f"Shift/width polynomial coefficients for spaxel {spax}",
        )

    for spaxel in spaxels_to_process:
        print(spaxel)
        _final_fit_and_animate(spaxel, iteration_max, out, cleanup_fits=False)


def _run_one_iteration_parallel(
    spaxels: list[int], iteration: int, is_offset_iter: bool, iter_params: list[float]
) -> list:
    """Submit all tasks for all spaxels in parallel before waiting.

    Neighbors (14 unchanged rows) are computed once per spaxel per iteration;
    only the target row is recomputed per param value.
    """
    bin_stats_futures = []
    for spax in spaxels:
        spectra = np.concatenate(spec[15 * (spax // 15) : 15 * (spax // 15 + 1)])
        neighbor_future = makeShiftedMat_neighbors.submit(spax, oversample_factor=4, iteration=iteration)
        for param in iter_params:
            offset = param if is_offset_iter else 0.0
            width = param if not is_offset_iter else 0.0
            target_future = makeShiftedMat_target.submit(spax, offset, width, oversample_factor=4, iteration=iteration)
            mat_future = combine_sparse_matrices.submit(neighbor_future, target_future)
            fit_future = fit.submit(
                mat_future, science_image, spectra, spaxel=spax, iteration=iteration, param=param
            )
            bin_stats_futures.append(compute_bin_stats.submit(fit_future, spax))
    return [f.result() for f in bin_stats_futures]


def _final_fit_and_animate(spaxel: int, iteration_max: int, output_dir: Path, cleanup_fits: bool) -> None:
    spectra = np.concatenate(spec[15 * (spaxel // 15) : 15 * (spaxel // 15 + 1)])
    mat_future = makeShiftedMat.submit(
        spaxel, np.zeros(225), np.zeros(225), oversample_factor=4, iteration=iteration_max
    )
    fit.submit(mat_future, science_image, spectra, spaxel=spaxel, iteration=iteration_max, param=0.0).result()

    models = []
    for it in [0] + list(range(1, iteration_max, 2)):
        with fits.open(f"spaxel_{spaxel}_iteration_{it}.fits") as hdul:
            models.append(hdul[0].data)  # type:ignore
    animate_poly_convergence(
        science_image, models, total_widths, str(spaxel),
        row_range=(700, 3000), col_range=(970, 1100), x_range=(0, 1400),
        fps=2, fade_factor=0.6, image_labels=None, colsum_yscale="symlog",
    )
    gif_path = output_dir / f"poly_convergence_{spaxel}.gif"
    create_markdown_artifact(
        markdown=f"**poly_convergence_{spaxel}.gif**\n\n`{gif_path}`",
        key=f"animation-spaxel-{spaxel}",
        description=f"Convergence animation for spaxel {spaxel}",
    )
    if cleanup_fits:
        for it in range(iteration_max + 1):
            fits_path = Path(f"spaxel_{spaxel}_iteration_{it}.fits")
            if fits_path.exists():
                fits_path.unlink()

# @pipeline_flow()
def run_make_parameter_matrix(
    science_exposure_path: Path,
    output_dir: Path,
    spaxels_to_process: list[int] | None = None,
    iteration_max: int = 10,
    cleanup_fits: bool = False,
):
    shift_offsets = [-0.2, -0.1, 0, 0.1, 0.2]
    width_multipliers = [-0.2, -0.1, 0, 0.1, 0.2, 0.3]

    global science_image, params
    image = Image.from_asdf(science_exposure_path)
    science_image = image.data.T  # transpose to match the orientation used in the original code

    output_dir.mkdir(parents=True, exist_ok=True)
    spaxels_to_process = spaxels_to_process if spaxels_to_process is not None else list(range(225))
    _load_spaxel_jsons(spaxels_to_process, output_dir)

    iteration = 0
    while iteration < iteration_max:
        is_offset_iter = iteration % 2 == 0
        params = shift_offsets if is_offset_iter else width_multipliers
        n_params = len(params)
        logger.info(
            f"iteration {iteration}, {'offsets' if is_offset_iter else 'widths'}, "
            f"{len(spaxels_to_process) * n_params} tasks → parallel"
        )
        flat_results = _run_one_iteration_parallel(spaxels_to_process, iteration, is_offset_iter, params)
        l1_calculations(spaxels_to_process, flat_results, n_params, iteration, is_offset_iter)
        iteration += 1
        _save_spaxel_jsons(spaxels_to_process, output_dir, iteration)

    shifts_path = output_dir / "loop_shifts_editable.json"
    widths_path = output_dir / "loop_widths_editable.json"
    image.header.set("shift_coeff_path", str(shifts_path))
    image.header.set("width_coeff_path", str(widths_path))
    image.to_asdf(science_exposure_path)
    logger.info(f"Updated science exposure header with coefficient paths at {science_exposure_path}")

    for spaxel in spaxels_to_process:
        _final_fit_and_animate(spaxel, iteration_max, output_dir, cleanup_fits)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--spaxels", type=str, default=None,
                        help="Comma-separated spaxel indices, e.g. 0,1,2 or a range like 0-14")
    parser.add_argument("--iteration-max", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to write per-spaxel JSON files (default: cwd)")
    args = parser.parse_args()

    if args.spaxels is None:
        spaxels = list(range(120, 135))
    elif "-" in args.spaxels and "," not in args.spaxels:
        start, end = args.spaxels.split("-")
        spaxels = list(range(int(start), int(end) + 1))
    else:
        spaxels = [int(s) for s in args.spaxels.split(",")]

    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    make_parameter_matrix_old(spaxels, iteration_max=args.iteration_max, output_dir=output_dir)

    combine_spaxel_jsons(output_dir)
    logger.info(f"Combined JSONs written to {output_dir}")

    for spaxel in spaxels:
        for it in range(args.iteration_max + 1):
            fits_path = Path(f"spaxel_{spaxel}_iteration_{it}.fits")
            if fits_path.exists():
                fits_path.unlink()
    logger.info("Cleaned up fits files")
