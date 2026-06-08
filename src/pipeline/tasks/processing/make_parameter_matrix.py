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


@pipeline_task()
def makeShiftedMat(
    spaxel: int, offsets: np.ndarray, widths: np.ndarray, oversample_factor: int = 1, iteration: int = 0
):
    assert oversample_factor > 0, "can't divide by 0"

    # offsests is a list of 225 offsets in the cross-dispersion direction
    #worker = multiprocessing.current_process().name
    row_start = 15 * (spaxel // 15)
    row_end = 15 * (spaxel // 15 + 1)

    list_huge_matrix = []

    for spaxel_ID in range(row_start, row_end):
        #logger.info(f"[{worker}]   spaxel_ID={spaxel_ID}, elapsed={time.time() - start_time:.1f}s")
        # create a new image
        # find the place in the image where to put the spectrum per spaxel
        a0 = int(A0_PARAMS[spaxel_ID] + yoff)
        a1 = int(A1_PARAMS[spaxel_ID] + yoff) + 1
        b0 = int(B0_PARAMS[spaxel_ID] + xoff - 50)
        off = 50
        if b0 < 0:
            off += b0
            b0 = 0

        b1 = int(B1_PARAMS[spaxel_ID] + xoff + 50) + 1

        # try:
        xsub = np.linspace(0, a1 - a0 - 1, (a1 - a0) * oversample_factor)
        ysub = np.linspace(0, b1 - b0 - 1, (b1 - b0) * oversample_factor)

        xv_sub, yv_sub = np.meshgrid(ysub, xsub)

        # we will model 1400 spectral elements for each spaxel in the blue cube
        x0 = np.arange(0, 1400, 1)

        p = Polynomial([QUARTIC_LINEAR_PARAMS[spaxel_ID], Z_1ST[spaxel_ID], Z_2ND[spaxel_ID], 0, Z_4TH[spaxel_ID]])
        adjustmentP = Polynomial(offsets[spaxel_ID])

        curve = p(x0) + adjustmentP(x0) + off

        if spaxel_ID == spaxel:
            latest_s = str(max(int(k) for k in total_coeffs[str(spaxel_ID)].keys()))
            curve = curve + np.poly1d(total_coeffs[str(spaxel_ID)][latest_s])(x0)

        widthAdjustmentP = Polynomial(widths[spaxel_ID])
        widthVals = widthAdjustmentP(x0)

        if spaxel_ID == spaxel:  # doing a spaxel specific width adjustment to refine
            latest_w = str(max(int(k) for k in total_widths[str(spaxel_ID)].keys()))
            widthVals += np.poly1d(total_widths[str(spaxel_ID)][latest_w])(x0)

        ######################################################
        for spec_element in range(0, 1400):
            # the per spaxel per wavelength model will be in this box:
            c0 = int(spec_element - 50) * oversample_factor
            if c0 < 0:
                c0 = 0
            c1 = int(spec_element + 50) * oversample_factor
            if c1 >= 1400 * oversample_factor:
                c1 = int(1400 * oversample_factor)

            ##########################################################################
            # the monochromatic image for spaxel with number spaxel_ID can be found in this box.
            # image[a0:a1,b0:b1][c0:c1,int(curve[spec_element]-50):int(curve[spec_element]+50)]

            d0 = int(round((curve[spec_element] - 50))) * oversample_factor
            if d0 < 0:
                d0 = 0
            d1 = int(round((curve[spec_element] + 50))) * oversample_factor
            if d1 >= 1400 * oversample_factor:
                d1 = int(1400 * oversample_factor) - 1

            xv_sub_mono = xv_sub[c0:c1, d0:d1]
            yv_sub_mono = yv_sub[c0:c1, d0:d1]

            # make the model ########################################
            popt = [0, 0]  # a way to allow for a shift.
            # redefine x and y
            y = yv_sub_mono.T[0] - spec_element
            x = xv_sub_mono[0] - curve[spec_element]

            spec_trace = QUAD_SPEC[spaxel_ID] * y**2 + LIN_SPEC[spaxel_ID] * y + popt[0]
            cross_trace = LIN_CROSS[spaxel_ID] * x + popt[1]

            xiii = (xv_sub_mono - curve[spec_element]).T - spec_trace
            yiii = (yv_sub_mono - spec_element) - cross_trace

            mask_footprint = (
                np.sqrt(
                    (yv_sub_mono - spec_element - ELL_C1[spaxel_ID] - popt[1]) ** 2 / ELL_B[spaxel_ID] ** 2
                    + (xv_sub_mono - curve[spec_element] - ELL_C0[spaxel_ID] - popt[0]) ** 2 / ELL_A[spaxel_ID] ** 2
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

            c0 = c0 // oversample_factor
            c1 = c1 // oversample_factor
            d0 = d0 // oversample_factor
            d1 = d1 // oversample_factor

            mask_val = testModel.T > 1e-4

            rowind = rowindex[a0:a1, b0:b1][c0:c1, d0:d1]

            row = rowind[mask_val[: len(rowind), : len(rowind.T)]]
            colind = columnindex[a0:a1, b0:b1][c0:c1, d0:d1]
            col = colind[mask_val[: len(rowind), : len(rowind.T)]]

            data = testModel.T[: len(rowind), : len(rowind.T)][mask_val[: len(rowind), : len(rowind.T)]]
            s_image = sparse.csr_matrix((data, (col, row)), shape=(4096, 2048))

            s_image = s_image.reshape((1, int(2048 * 4096)))
            list_huge_matrix.append(s_image)

    huge_matrix = sparse.vstack(list_huge_matrix)

    return huge_matrix


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


@pipeline_flow()
def make_parameter_matrix_old(spaxels_to_process: list[int] | None = None, iteration_max: int = 14):
    shift_offsets = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2]
    width_multipliers = [-0.2, -0.1, 0, 0.1, 0.2, 0.3]

    global science_image, params
    with fits.open("/home/anousha/snifs_model/refs/deep_skyflat_coadd.fits") as hdul:
        science_image = hdul[0].data  # type:ignore

    spaxels_to_process = spaxels_to_process if spaxels_to_process is not None else [8]

    iteration = 0
    while iteration < iteration_max:
        is_offset_iter = iteration % 2 == 0
        params = shift_offsets if is_offset_iter else width_multipliers
        n_params = len(params)

        logger.info(
            f"iteration {iteration}, {'offsets' if is_offset_iter else 'widths'}, "
            f"{len(spaxels_to_process) * n_params} tasks → parallel"
        )

        # Each spaxel's params run in parallel; wait for all of one spaxel before moving to the next.
        flat_results = []
        for spax in spaxels_to_process:
            spectra = np.concatenate(spec[15 * (spax // 15) : 15 * (spax // 15 + 1)])
            spax_futures = []
            for param in params:
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

        l1_calculations(spaxels_to_process, flat_results, n_params, iteration, is_offset_iter)
        logger.info("L1 calculations complete")
        iteration += 1

    for spax in spaxels_to_process:
        spax_shifts_path = os.path.abspath(f"loop_shifts_spaxel_{spax}.json")
        with open(spax_shifts_path, "w") as f:
            json.dump({str(spax): total_coeffs[str(spax)]}, f)
        spax_widths_path = os.path.abspath(f"loop_widths_spaxel_{spax}.json")
        with open(spax_widths_path, "w") as f:
            json.dump({str(spax): total_widths[str(spax)]}, f)
        create_markdown_artifact(
            markdown=(
                f"**Per-spaxel coefficients (spaxel {spax})**\n\n"
                f"shifts: `{spax_shifts_path}`\n\nwidths: `{spax_widths_path}`"
            ),
            key=f"wavelength-cal-spaxel-{spax}",
            description=f"Shift/width polynomial coefficients for spaxel {spax}",
        )

    for spaxel in spaxels_to_process:
        spectra = np.concatenate(spec[15 * (spaxel // 15) : 15 * (spaxel // 15 + 1)])
        mat_future = makeShiftedMat.submit(spaxel, np.zeros(225), np.zeros(225), oversample_factor=4, iteration=14)
        fit.submit(mat_future, science_image, spectra, spaxel=spaxel, iteration=14, param=0.0).result()


    iteration = iteration_max
    for spaxel in spaxels_to_process:
        print(spaxel)
        models = []
        model_iterations = [0]
        model_iterations.extend(list(range(2, iteration+1, 2)))
        for it in model_iterations:
            fname = f"spaxel_{spaxel}_iteration_{it}.fits"
            with fits.open(fname) as hdul:
                models.append(hdul[0].data)  # type:ignore
            print(fname)
            print(total_widths[str(spaxel)][str(it)])
        animate_poly_convergence(
            science_image,
            models,
            total_widths,
            str(spaxel),
            row_range=(700, 3000),
            col_range=(970, 1100),
            x_range=(0, 1400),
            fps=2,
            fade_factor=0.6,
            image_labels=None,
            colsum_yscale="symlog",
        )
        gif_path = os.path.abspath(f"poly_convergence_{spaxel}.gif")
        create_markdown_artifact(
            markdown=f"**poly_convergence_{spaxel}.gif**\n\n`{gif_path}`",
            key=f"animation-spaxel-{spaxel}",
            description=f"Convergence animation for spaxel {spaxel}",
        )

#@pipeline_flow()
def run_make_parameter_matrix(
    science_exposure_path: Path,
    output_dir: Path,
    spaxels_to_process: list[int] | None = None,
    iteration_max: int = 14,
    cleanup_fits: bool = False,
):
    shift_offsets = [-0.2, -0.1, 0, 0.1, 0.2]
    width_multipliers = [-0.2, -0.1, 0, 0.1, 0.2, 0.3]

    global science_image, params
    image = Image.from_asdf(science_exposure_path)
    science_image = image.data
    science_image = science_image.T # transpose to match the orientation used in the original code

    output_dir.mkdir(parents=True, exist_ok=True)
    shifts_path = output_dir / "loop_shifts_editable.json"
    widths_path = output_dir / "loop_widths_editable.json"

    if spaxels_to_process is None:
        spaxels_to_process = list(range(225))

    iteration = 0
    while iteration < iteration_max:
        is_offset_iter = iteration % 2 == 0
        params = shift_offsets if is_offset_iter else width_multipliers
        n_params = len(params)

        logger.info(
            f"iteration {iteration}, {'offsets' if is_offset_iter else 'widths'}, "
            f"{len(spaxels_to_process) * n_params} tasks → parallel"
        )

        bin_stats_futures = []
        for spax in spaxels_to_process:
            spectra = np.concatenate(spec[15 * (spax // 15) : 15 * (spax // 15 + 1)])
            for param in params:
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
                bin_stats_futures.append(compute_bin_stats.submit(fit_future, spax))
        flat_results = [f.result() for f in bin_stats_futures]

        l1_calculations(spaxels_to_process, flat_results, n_params, iteration, is_offset_iter)

        iteration += 1

    for spax in spaxels_to_process:
        spax_shifts_path = output_dir / f"loop_shifts_spaxel_{spax}.json"
        with spax_shifts_path.open("w") as f:
            json.dump({str(spax): total_coeffs[str(spax)]}, f)
        spax_widths_path = output_dir / f"loop_widths_spaxel_{spax}.json"
        with spax_widths_path.open("w") as f:
            json.dump({str(spax): total_widths[str(spax)]}, f)
    logger.info(f"Saved per-spaxel shift/width JSONs to {output_dir}")

    image.header.set("shift_coeff_path", str(shifts_path))
    image.header.set("width_coeff_path", str(widths_path))
    image.to_asdf(science_exposure_path)
    logger.info(f"Updated science exposure header with coefficient paths at {science_exposure_path}")

    for spaxel in spaxels_to_process:
        spectra = np.concatenate(spec[15 * (spaxel // 15) : 15 * (spaxel // 15 + 1)])
        mat_future = makeShiftedMat.submit(spaxel, np.zeros(225), np.zeros(225),
                                           oversample_factor=4, iteration=iteration_max)
        fit.submit(mat_future, science_image, spectra, spaxel=spaxel, iteration=iteration_max, param=0.0).result()

        models = []
        model_iterations = [0]
        model_iterations.extend(list(range(1, iteration_max, 2)))
        for it in model_iterations:
            fname = f"spaxel_{spaxel}_iteration_{it}.fits"
            with fits.open(fname) as hdul:
                models.append(hdul[0].data)  # type:ignore
        animate_poly_convergence(
            science_image,
            models,
            total_widths,
            str(spaxel),
            row_range=(700, 3000),
            col_range=(970, 1100),
            x_range=(0, 1400),
            fps=2,
            fade_factor=0.6,
            image_labels=None,
            colsum_yscale="symlog",
        )
        gif_path = output_dir / f"poly_convergence_{spaxel}.gif"
        create_markdown_artifact(
            markdown=f"**poly_convergence_{spaxel}.gif**\n\n`{gif_path}`",
            key=f"animation-spaxel-{spaxel}",
            description=f"Convergence animation for spaxel {spaxel}",
        )
        if cleanup_fits:
            for it in range(15):
                fits_path = Path(f"spaxel_{spaxel}_iteration_{it}.fits")
                if fits_path.exists():
                    fits_path.unlink()


if __name__ == "__main__":
    make_parameter_matrix_old(list(range(120,135)), iteration_max=10)
