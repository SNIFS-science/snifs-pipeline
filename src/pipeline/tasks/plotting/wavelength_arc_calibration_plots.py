from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from numpy.polynomial import Polynomial
from scipy import sparse
from scipy.optimize import nnls
from scipy.special import erf

from pipeline import settings
from pipeline.common import Image, get_logger, pipeline_task
from pipeline.common.model_params import (
    A0_PARAMS,
    A1_PARAMS,
    B0_PARAMS,
    B1_PARAMS,
    QUARTIC_LINEAR_PARAMS,
    Z_1ST,
    Z_2ND,
    Z_4TH,
)
from pipeline.common.plotting_utils import get_all_peaks
from pipeline.resolver.resolver import PUBLIC_PATH_MAP, get_run_id
from pipeline.tasks.plotting.plots import CMAP_DATA, CMAP_DIFF, add_colorbar, plot_standalone

ALL_PEAKS = get_all_peaks()


def animate_poly_convergence(
    science_image,
    models,
    coeffs_dict,
    outer_key,
    row_range,
    col_range=(900, 1100),
    x_range=(0, 1400),
    fps=2,
    fade_factor=0.6,
    image_labels=None,
    colsum_yscale="symlog",
    output_file=None,
):
    """Animate polynomial convergence with the same left-panel layout as animate_bin_metrics.

    Args :
    science_image  : 2-D array
    models         : list of 2-D arrays, one per iteration
    coeffs_dict    : dict  {outer_key: {iter_label: [c4,c3,c2,c1,c0], ...}}
    outer_key      : str   which outer key to animate
    row_range      : (row_lo, row_hi)  rows shown in the cutout
    col_range      : (col_lo, col_hi)  columns shown in the cutout
    x_range        : (float, float)    x span for polynomial evaluation
    fps            : int
    fade_factor    : float  alpha multiplier per step of age
    image_labels   : list of str or None  (used as upper-left title each frame)
    colsum_yscale  : str  y-axis scale for the column-sum panel ('symlog',
                    'log', or 'linear').  Use 'symlog' when data may include
                    zeros or negatives — 'log' will hide those values entirely.
    """
    inner = coeffs_dict[str(outer_key)]
    sorted_items = sorted(inner.items(), key=lambda kv: int(kv[0]))

    a0 = int(A0_PARAMS[int(outer_key)])
    a1 = int(A1_PARAMS[int(outer_key)]) + 1
    b0 = int(B0_PARAMS[int(outer_key)]) - 50
    off = 50
    if b0 < 0:
        off += b0
        b0 = 0
    b1 = int(B1_PARAMS[int(outer_key)]) + 50 + 1

    # if the dict has more entries than models, drop from the front
    n_models = len(models)
    if len(sorted_items) > n_models:
        sorted_items = sorted_items[len(sorted_items) - n_models :]
    elif len(sorted_items) < n_models:
        raise ValueError(
            f"coeffs_dict['{outer_key}'] has only {len(sorted_items)} entries but {n_models} models were provided."
        )

    print(f"number of models: {n_models}")
    print(f"number of coeffs_dict entries: {len(sorted_items)}, sorted items are {sorted_items}")
    iter_labels = [kv[0] for kv in sorted_items]
    n_lines = len(sorted_items)

    row_lo, row_hi = a0, a1  # row_range
    col_lo, col_hi = (b0 + b1) // 2 - 6, (b0 + b1) // 2 + 6  # col_range
    col_indices = np.arange(col_lo, col_hi)

    # ── precompute diff cutouts and vertical sums ────────────────────────
    diff_cutouts = [science_image[row_lo:row_hi, col_lo:col_hi] - m[row_lo:row_hi, col_lo:col_hi] for m in models]
    all_diffs = np.stack(diff_cutouts)
    vlo, vhi = np.nanpercentile(all_diffs, [1, 99])

    diff_vsums = [np.nansum(dc, axis=0) for dc in diff_cutouts]
    vsum_min = min(v.min() for v in diff_vsums)
    vsum_max = max(v.max() for v in diff_vsums)
    vsum_margin = max((vsum_max - vsum_min) * 0.1, 1e-8)

    # ── precompute column sums of each model in the cutout ───────────────

    model_colsums = [np.nansum(m[b0:b1, a0:a1], axis=1) for m in models]
    print("model colsums:", len(model_colsums))

    science_colsum = np.nansum(science_image[b0:b1, a0:a1], axis=1)

    # ── precompute polynomials ───────────────────────────────────────────
    x = np.linspace(x_range[0], x_range[1], 1000)
    poly_ys = [np.poly1d(coeffs)(x) for _, coeffs in sorted_items]

    all_y = np.concatenate(poly_ys)
    y_lo, y_hi = np.nanmin(all_y), np.nanmax(all_y)
    y_margin = max((y_hi - y_lo) * 0.08, 1e-8)

    # color each line light→dark: oldest=faintest, newest=darkest
    colors = plt.cm.Blues(np.linspace(0.3, 0.95, n_lines))

    # ── layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 6))
    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.35, figure=fig)

    # left side: two rows
    left_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[0], hspace=0.4)

    # upper-left: diff image
    ax_img = fig.add_subplot(left_gs[0])
    im_handle = ax_img.imshow(diff_cutouts[0], aspect="auto", origin="lower", vmin=-vhi, vmax=vhi, cmap="RdBu_r")
    fig.colorbar(im_handle, ax=ax_img, fraction=0.046, pad=0.04)
    img_title = ax_img.set_title(
        image_labels[0] if image_labels else f"iter {iter_labels[0]}\nscience − model",
        fontsize=9,
    )
    ax_img.set_xlabel("Column offset")
    ax_img.set_ylabel("Row offset")

    # lower-left: vertical sum
    ax_vsum = fig.add_subplot(left_gs[1])
    (vsum_line,) = ax_vsum.plot(col_indices, diff_vsums[0], color="steelblue", lw=1.2)
    ax_vsum.set_xlim(col_lo, col_hi)
    ax_vsum.set_ylim(vsum_min - vsum_margin, vsum_max + vsum_margin)
    ax_vsum.set_xlabel("Column")
    ax_vsum.set_ylabel("Summed counts")
    ax_vsum.set_title("Vertical sum (science − model)", fontsize=9)

    # right side: two rows with height ratios 4:1
    right_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[1], height_ratios=[4, 1], hspace=0.15)

    # right-top: polynomial accumulation
    ax_poly = fig.add_subplot(right_gs[0])
    ax_poly.set_xlim(x_range[0], x_range[1])
    ax_poly.set_ylim(y_lo - y_margin, y_hi + y_margin)
    ax_poly.set_ylabel("y")
    ax_poly.set_title(f"Polynomial convergence  (key={outer_key})", fontsize=9)
    ax_poly.axhline(0, color="k", lw=0.4, ls="--", alpha=0.3)
    plt.setp(ax_poly.get_xticklabels(), visible=False)

    line_handles = []
    for i, (y_vals, label) in enumerate(zip(poly_ys, iter_labels, strict=False)):
        (ln,) = ax_poly.plot(x, y_vals, color=colors[i], alpha=0, lw=2, label=f"iter {label}")
        line_handles.append(ln)

    ax_poly.legend(loc="upper right", fontsize=7)

    # right-bottom: column sum of model in cutout (sharex with poly plot)
    ax_col = fig.add_subplot(right_gs[1], sharex=ax_poly)
    (colsum_line,) = ax_col.plot(model_colsums[i], lw=1.2, label="model")
    ax_col.plot(science_colsum, lw=1.2, color="k", ls="--", label="science")
    ax_col.set_ylim(science_colsum.min(), science_colsum.max())
    ax_col.set_xlabel("x (pixels)")
    # ax_col.set_ylabel('Col sum', fontsize=8)
    ax_col.legend()
    ax_col.set_yscale("symlog")

    fig.suptitle(f"Polynomial evolution — key {outer_key}", fontsize=11)

    # ── animation ───────────────────────────────────────────────────────
    def update(frame):
        im_handle.set_data(diff_cutouts[frame])
        vsum_line.set_ydata(diff_vsums[frame])
        colsum_line.set_ydata(model_colsums[frame])
        img_title.set_text(image_labels[frame] if image_labels else f"iter {iter_labels[frame]}\nscience − model")
        for i, ln in enumerate(line_handles):
            if i <= frame:
                ln.set_alpha(fade_factor ** (frame - i))

    anim = FuncAnimation(fig, update, frames=n_lines, blit=False, interval=int(1000 / fps))

    if output_file is None:
        output_file = f"poly_convergence_{outer_key}.gif"
    anim.save(output_file, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    print(f"Saved → {output_file}")


def animate_gl_convergence(
    science_image,
    models,
    coeffs_dict,
    outer_key,
    frame_iterations,
    gauss_vals,
    lorentz_vals,
    row_range,
    col_range=(900, 1100),
    x_range=(0, 1400),
    fps=2,
    fade_factor=0.6,
    image_labels=None,
    output_file=None,
):
    """Like animate_poly_convergence, but the bottom-right panel tracks the gauss/lorentz scales.

    Intended for the gaussian/lorentzian width-scan stage: one frame per model, with the bottom-right
    panel showing iteration on the x-axis and a scatter point per frame for the gaussian and
    lorentzian width multipliers as they evolve.

    Args:
        science_image   : 2-D array
        models          : list of 2-D arrays, one per frame
        coeffs_dict     : dict {outer_key: {frame_label: [c4,c3,c2,c1,c0], ...}}  (width polynomial
                          in effect at each frame, exactly one entry per model)
        outer_key       : str  which outer key to animate
        frame_iterations: list of int  iteration number for each frame (x-axis of the scatter)
        gauss_vals      : list of float  gaussian width scale at each frame
        lorentz_vals    : list of float  lorentzian width scale at each frame
        row_range/col_range/x_range/fps/fade_factor/image_labels/output_file: as animate_poly_convergence.
    """
    inner = coeffs_dict[str(outer_key)]
    sorted_items = sorted(inner.items(), key=lambda kv: int(kv[0]))

    a0 = int(A0_PARAMS[int(outer_key)])
    a1 = int(A1_PARAMS[int(outer_key)]) + 1
    b0 = int(B0_PARAMS[int(outer_key)]) - 50
    off = 50
    if b0 < 0:
        off += b0
        b0 = 0
    b1 = int(B1_PARAMS[int(outer_key)]) + 50 + 1

    n_models = len(models)
    if len(sorted_items) > n_models:
        sorted_items = sorted_items[len(sorted_items) - n_models :]
    elif len(sorted_items) < n_models:
        raise ValueError(
            f"coeffs_dict['{outer_key}'] has only {len(sorted_items)} entries but {n_models} models were provided."
        )
    if not (len(frame_iterations) == len(gauss_vals) == len(lorentz_vals) == n_models):
        raise ValueError(
            f"frame_iterations ({len(frame_iterations)}), gauss_vals ({len(gauss_vals)}), "
            f"lorentz_vals ({len(lorentz_vals)}) must all match n_models ({n_models})."
        )

    iter_labels = [kv[0] for kv in sorted_items]
    n_lines = len(sorted_items)

    row_lo, row_hi = a0, a1
    col_lo, col_hi = (b0 + b1) // 2 - 6, (b0 + b1) // 2 + 6
    col_indices = np.arange(col_lo, col_hi)

    # ── precompute diff cutouts and vertical sums ────────────────────────
    diff_cutouts = [science_image[row_lo:row_hi, col_lo:col_hi] - m[row_lo:row_hi, col_lo:col_hi] for m in models]
    all_diffs = np.stack(diff_cutouts)
    vlo, vhi = np.nanpercentile(all_diffs, [1, 99])

    diff_vsums = [np.nansum(dc, axis=0) for dc in diff_cutouts]
    vsum_min = min(v.min() for v in diff_vsums)
    vsum_max = max(v.max() for v in diff_vsums)
    vsum_margin = max((vsum_max - vsum_min) * 0.1, 1e-8)

    # ── precompute polynomials ───────────────────────────────────────────
    x = np.linspace(x_range[0], x_range[1], 1000)
    poly_ys = [np.poly1d(coeffs)(x) for _, coeffs in sorted_items]
    all_y = np.concatenate(poly_ys)
    y_lo, y_hi = np.nanmin(all_y), np.nanmax(all_y)
    y_margin = max((y_hi - y_lo) * 0.08, 1e-8)

    colors = plt.cm.Blues(np.linspace(0.3, 0.95, n_lines))

    # ── layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 6))
    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.35, figure=fig)
    left_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[0], hspace=0.4)

    # upper-left: diff image
    ax_img = fig.add_subplot(left_gs[0])
    im_handle = ax_img.imshow(diff_cutouts[0], aspect="auto", origin="lower", vmin=-vhi, vmax=vhi, cmap="RdBu_r")
    fig.colorbar(im_handle, ax=ax_img, fraction=0.046, pad=0.04)
    img_title = ax_img.set_title(
        image_labels[0] if image_labels else f"iter {frame_iterations[0]}\nscience − model",
        fontsize=9,
    )
    ax_img.set_xlabel("Column offset")
    ax_img.set_ylabel("Row offset")

    # lower-left: vertical sum
    ax_vsum = fig.add_subplot(left_gs[1])
    (vsum_line,) = ax_vsum.plot(col_indices, diff_vsums[0], color="steelblue", lw=1.2)
    ax_vsum.set_xlim(col_lo, col_hi)
    ax_vsum.set_ylim(vsum_min - vsum_margin, vsum_max + vsum_margin)
    ax_vsum.set_xlabel("Column")
    ax_vsum.set_ylabel("Summed counts")
    ax_vsum.set_title("Vertical sum (science − model)", fontsize=9)

    right_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[1], height_ratios=[4, 1], hspace=0.3)

    # right-top: polynomial accumulation
    ax_poly = fig.add_subplot(right_gs[0])
    ax_poly.set_xlim(x_range[0], x_range[1])
    ax_poly.set_ylim(y_lo - y_margin, y_hi + y_margin)
    ax_poly.set_ylabel("y")
    ax_poly.set_title(f"Polynomial convergence  (key={outer_key})", fontsize=9)
    ax_poly.axhline(0, color="k", lw=0.4, ls="--", alpha=0.3)

    line_handles = []
    for i, (y_vals, label) in enumerate(zip(poly_ys, iter_labels, strict=False)):
        (ln,) = ax_poly.plot(x, y_vals, color=colors[i], alpha=0, lw=2, label=f"iter {label}")
        line_handles.append(ln)
    ax_poly.legend(loc="upper right", fontsize=7)

    # right-bottom: gauss/lorentz scales vs iteration (twin y-axes, different ranges)
    ax_g = fig.add_subplot(right_gs[1])
    ax_l = ax_g.twinx()
    it_arr = np.asarray(frame_iterations, dtype=float)
    g_arr = np.asarray(gauss_vals, dtype=float)
    l_arr = np.asarray(lorentz_vals, dtype=float)

    x_pad = max((it_arr.max() - it_arr.min()) * 0.05, 0.5)
    ax_g.set_xlim(it_arr.min() - x_pad, it_arr.max() + x_pad)
    g_pad = max((g_arr.max() - g_arr.min()) * 0.2, 0.05)
    l_pad = max((l_arr.max() - l_arr.min()) * 0.2, 0.05)
    ax_g.set_ylim(g_arr.min() - g_pad, g_arr.max() + g_pad)
    ax_l.set_ylim(l_arr.min() - l_pad, l_arr.max() + l_pad)

    ax_g.set_xlabel("Iteration")
    ax_g.set_ylabel("Gaussian scale", color="tab:blue", fontsize=8)
    ax_l.set_ylabel("Lorentzian scale", color="tab:red", fontsize=8)
    ax_g.tick_params(axis="y", labelcolor="tab:blue")
    ax_l.tick_params(axis="y", labelcolor="tab:red")

    g_scatter = ax_g.scatter([], [], color="tab:blue", marker="o", s=30, label="gaussian")
    l_scatter = ax_l.scatter([], [], color="tab:red", marker="s", s=30, label="lorentzian")
    ax_g.legend(handles=[g_scatter, l_scatter], labels=["gaussian", "lorentzian"], loc="upper left", fontsize=7)

    fig.suptitle(f"Gauss/Lorentz evolution — key {outer_key}", fontsize=11)

    # ── animation ───────────────────────────────────────────────────────
    def update(frame):
        im_handle.set_data(diff_cutouts[frame])
        vsum_line.set_ydata(diff_vsums[frame])
        img_title.set_text(image_labels[frame] if image_labels else f"iter {frame_iterations[frame]}\nscience − model")
        for i, ln in enumerate(line_handles):
            if i <= frame:
                ln.set_alpha(fade_factor ** (frame - i))
        # reveal scatter points up to and including the current frame
        g_scatter.set_offsets(np.column_stack([it_arr[: frame + 1], g_arr[: frame + 1]]))
        l_scatter.set_offsets(np.column_stack([it_arr[: frame + 1], l_arr[: frame + 1]]))

    anim = FuncAnimation(fig, update, frames=n_lines, blit=False, interval=int(1000 / fps))

    if output_file is None:
        output_file = f"gl_convergence_{outer_key}.gif"
    anim.save(output_file, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    print(f"Saved → {output_file}")


@pipeline_task()
def plot_refined_spectrum(spec: np.ndarray, other_new_centers: np.ndarray) -> None:
    """Plot the refined spectrum with the new center points overlaid as vertical lines.

    Args:
        spec: numpy array of spectrum data
        other_new_centers: numpy array of new center points for the spectrum.
    """
    logger = get_logger()

    if not settings.plot:
        logger.info("Plotting is disabled. Skipping plot generation.")
        return
    plt.plot(spec)
    plt.vlines(other_new_centers, 0, 10e5, color="r", alpha=0.5)
    plt.ylim(10, np.max(spec) * 1.1)
    plt.yscale("log")

    flow_run_id = get_run_id()
    try:
        output_location = (PUBLIC_PATH_MAP[flow_run_id] / f"wavelength_arc_fit_{flow_run_id}.webp").resolve()
        output_location.parent.mkdir(parents=True, exist_ok=True)
    except:
        output_location = Path(f"output/level=processed/wavelength_arc_fit_{flow_run_id}.webp").resolve()
    logger.info(f"Saving plot to {output_location}")
    plt.savefig(output_location, dpi=600, bbox_inches="tight")
    plt.close()
    output_location.chmod(0o644)  # Make the file readable by everyone
    return


@pipeline_task()
def plot_params(params: np.ndarray) -> None:
    """Plot the fit parameters for each spaxel as a grid of scatter plots to check for bad spaxels visually.

    Args:
        params: numpy array of parameters for the cubic wavelength fit for each spaxel
    Returns:
        None
    """
    logger = get_logger()

    if not settings.plot:
        logger.info("Plotting is disabled. Skipping plot generation.")
        return
    fig, ax = plt.subplots(2, 2)
    coeff_names = ["x³ coefficient", "x² coefficient", "x¹ coefficient", "constant term"]
    labels = np.arange(1, 226).reshape(15, 15)

    x, y = np.meshgrid(np.arange(15), np.arange(15))
    for i in range(4):
        grid = params[:, i].reshape(15, 15)
        row, col = divmod(i, 2)
        ax_i = ax[row, col]
        sc = ax_i.scatter(x, y, c=grid, s=50)
        # Add labels like in the figure for Yannick's habilitation
        for xi in range(15):
            for yi in range(15):
                ax_i.text(x[xi, yi] + 0.05, y[xi, yi] + 0.05, str(labels[xi, yi]), fontsize=8, rotation=35)
        fig.colorbar(sc, ax=ax_i, label=coeff_names[i])
        ax_i.set_title(coeff_names[i])
        ax_i.set_aspect("equal")

    flow_run_id = get_run_id()
    output_location = (PUBLIC_PATH_MAP[flow_run_id] / f"fit_coefficients_grid_{flow_run_id}.webp").resolve()
    output_location.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving plot to {output_location}")
    plt.tight_layout()
    plt.savefig(output_location, dpi=600, bbox_inches="tight")
    plt.close()
    output_location.chmod(0o644)  # Make the file readable by everyone

    return


@pipeline_task()
@plot_standalone("rectified spectrum")
def plot_spectrum(wavelengths: np.ndarray, fluxes: np.ndarray) -> Image:
    """Plot the spectrum as a scatter plot of flux vs wavelength, with color representing flux intensity.

    Args:
        wavelengths: numpy array of wavelengths
        fluxes: numpy array of flux values
    Returns:
        Image wrapping the flux data, for use with the plot_standalone decorator.
    """
    wave_grid = np.linspace(wavelengths.min(), wavelengths.max(), wavelengths.shape[1])
    regularized = np.array([np.interp(wave_grid, wavelengths[i], fluxes[i]) for i in range(wavelengths.shape[0])])

    # Flatten the data for plotting
    """ X = wavelengths.flatten()  # Wavelengths
    Y = np.repeat(np.arange(225), 1499)  # Object indices
    C = fluxes.flatten()  # Flux values

    # Avoid log(0) issues — filter out or replace nonpositive values
    mask = C > 1
    X, Y, C = X[mask], Y[mask], C[mask]

    plt.figure(figsize=(12, 6))
    sc = plt.scatter(X, Y, c=C, cmap="viridis", s=2, norm=LogNorm())
    plt.colorbar(sc, label="Flux (log scale)")
    plt.xlabel(r"Wavelength")
    plt.ylabel("Spaxel")
    plt.vlines(ALL_PEAKS, -2, 230, color="k", linestyle="--", alpha=0.5)

    plt.title(f"{flow_run_id} Arc: Flux vs Wavelength")
    output_location = (PUBLIC_PATH_MAP[flow_run_id] / f"fit_wavelengths_{flow_run_id}.webp").resolve()
    output_location.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving plot to {output_location}")
    plt.tight_layout()
    plt.savefig(output_location, dpi=600, bbox_inches="tight")
    plt.close()
    output_location.chmod(0o644)  # Make the file readable by everyone """  # noqa: W605

    return Image.from_array_and_dict({}, regularized, np.zeros_like(regularized))


@pipeline_task()
def plot_fitting_check(fitModel: sparse.csr_matrix, imagea: np.ndarray) -> None:
    """Plot the fitted model, actual data, and their difference to visually check the quality of the fit.

    Args:
        fitModel: sparse matrix of fitted model data
        imagea: numpy array of actual image data
    Returns:
        None
    """
    logger = get_logger()

    if not settings.plot:
        logger.info("Plotting is disabled. Skipping plot generation.")
        return

    flow_run_id = get_run_id()

    notbadmodel = fitModel + 9  # to account for readout noise
    difference = imagea - fitModel

    chi2 = np.square(difference) / notbadmodel
    logger.info("Chi^2 = ", np.sum(chi2))

    for i in range(1000, 3000, 600):
        plt.plot(imagea[i, :], label="data")
        plt.plot(fitModel[i, :], label="model")
        plt.plot(difference[i, :], label="data-model")
        plt.fill_between(
            np.arange(imagea[i, :].shape[0]),
            np.sqrt(imagea[i, :] + 9),
            -np.sqrt(imagea[i, :] + 9),
            alpha=0.3,
            color="C2",
        )
        plt.title(f"row {i} Both in first PV")
        plt.legend()
        output_location = (PUBLIC_PATH_MAP[flow_run_id] / f"spaxel_{i}_cut_through_{flow_run_id}.webp").resolve()
        output_location.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving plot to {output_location}")
        plt.tight_layout()
        plt.savefig(output_location, dpi=600, bbox_inches="tight")
        plt.close()
        output_location.chmod(0o644)

    plt.clf()

    fig, ax = plt.subplots(2, 2)
    im1 = ax[0][0].imshow(fitModel, cmap="plasma", aspect="auto")
    fig.colorbar(im1, ax=ax[0][0])
    ax[0][0].set_xlim(900, 1200)
    ax[0][0].set_ylim(3500, 0)
    ax[0][0].set_title("Model with 0-ed spaxel but including the read noise")
    # plt.show()

    im2 = ax[0][1].imshow(imagea, cmap="plasma", aspect="auto")
    fig.colorbar(im2, ax=ax[0][1])
    ax[0][1].set_xlim(900, 1200)
    ax[0][1].set_ylim(3500, 0)
    ax[0][1].set_title("Data")

    im3 = ax[1][0].imshow(difference, cmap="plasma", aspect="auto", norm="symlog")
    fig.colorbar(im3, ax=ax[1][0])

    ax[1][0].set_xlim(900, 1200)
    ax[1][0].set_ylim(3500, 0)
    ax[1][0].set_title("Data -  Model")

    im4 = ax[1][1].imshow(chi2, cmap="plasma", aspect="auto", norm="symlog")
    fig.colorbar(im4, ax=ax[1][1])
    ax[1][1].set_xlim(900, 1200)
    ax[1][1].set_ylim(3500, 0)
    ax[1][1].set_title("Chi2")

    output_location = (PUBLIC_PATH_MAP[flow_run_id] / f"fitter_checking_{flow_run_id}.webp").resolve()
    output_location.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving plot to {output_location}")
    plt.tight_layout()
    plt.savefig(output_location, dpi=600, bbox_inches="tight")
    plt.close()
    output_location.chmod(0o644)  # Make the file readable by everyone
    return


if __name__ == "__main__":
    import cmasher as cmr
    from astropy.io import fits

    # directory = "/Users/anousha/Desktop/preprocessed"
    directory = "/Users/anousha/Desktop/"

    image_dict = {}
    for file in [
        "guess_101.fits",
    ]:
        try:
            with fits.open(directory + "Homework/model_generated_images/" + file) as hdul:
                data = hdul[0].data  # type: ignore
            with fits.open(directory + "SNIFS/model/refs/deep_skyflat_coadd.fits") as hdul:
                data1 = hdul[0].data  # type: ignore
        except FileNotFoundError:
            continue

        ZOOM_START = (2048 - 25, 1024)  # (1000, 509 - 25)
        ZOOM_SIZE = (50, 50)

        print(ZOOM_START[0], ZOOM_START[1])
        print(type(ZOOM_START[0]), type(ZOOM_START[0] + ZOOM_SIZE[0]))

        CMAP_ZOOM = cmr.rainforest
        model_zoom = data[ZOOM_START[0] : ZOOM_START[0] + ZOOM_SIZE[0], ZOOM_START[1] : ZOOM_START[1] + ZOOM_SIZE[1]]
        # plt.imshow(model_zoom)
        # plt.show()
        data_zoom = data1[ZOOM_START[0] : ZOOM_START[0] + ZOOM_SIZE[0], ZOOM_START[1] : ZOOM_START[1] + ZOOM_SIZE[1]]
        # plt.imshow(data_zoom)
        # plt.show()
        difference_zoom = data_zoom - model_zoom
        plt.imshow(difference_zoom)
        plt.title("Data - Model")
        plt.colorbar()
        plt.plot([24, 24], [0, 50], color="r", linestyle="--")

        plt.plot([26, 26], [0, 50], color="k", linestyle="--")
        plt.show()

        numerator = np.multiply(model_zoom, data_zoom)
        denominator = np.multiply(model_zoom, model_zoom)
        residual = np.sum(numerator) / np.sum(denominator)

        print(residual)
        print("bright", np.sum(model_zoom[:, 26] * data_zoom[:, 26]) / np.sum(model_zoom[:, 26] * model_zoom[:, 26]))
        print("central", np.sum(model_zoom[:, 25] * data_zoom[:, 25]) / np.sum(model_zoom[:, 25] * model_zoom[:, 25]))

        plt.plot(100 * np.abs(data_zoom[:, 26] / model_zoom[:, 26]), label="bright col")
        plt.plot(100 * np.abs(data_zoom[:, 25] / model_zoom[:, 25]), label="central col")
        # plt.plot(difference_zoom[:,26],label="Difference")
        # plt.plot(data_zoom[:,26], label="Data")

        plt.title("Data/Model Percentage")
        plt.legend()
        plt.ylabel("Percentage")
        plt.show()


N_ELEMENTS = 1400
NOISE_SIGMAS = 2  # a pixel is noise-limited when 1% of its row's own peak is under this many noise sigmas


def trace_centres(shift_curves: dict[int, np.ndarray] | None = None) -> dict[int, tuple[int, np.ndarray]]:
    """Trace centre of every spaxel: the nominal trace of make_parameter_matrix plus a fitted shift.

    Args:
        shift_curves: {spaxel: shift per spectral element}; spaxels left out get no shift.

    Returns:
        dict[int, tuple[int, np.ndarray]]: {spaxel: (first row, centre column of each of its 1400 elements)}.
    """
    elements = np.arange(N_ELEMENTS)
    centres = {}
    for sid in range(len(A0_PARAMS)):
        b0 = int(B0_PARAMS[sid] - 50)
        off = 50
        if b0 < 0:
            off += b0
            b0 = 0
        nominal = Polynomial([QUARTIC_LINEAR_PARAMS[sid], Z_1ST[sid], Z_2ND[sid], 0, Z_4TH[sid]])(elements) + off
        shift = shift_curves.get(sid, 0.0) if shift_curves else 0.0
        centres[sid] = (int(A0_PARAMS[sid]), b0 + nominal + shift)
    return centres


def own_columns(
    spaxel: int, centres: dict[int, tuple[int, np.ndarray]], max_half_width: float = 5.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Columns belonging to a spaxel in each of its rows.

    A spaxel owns the pixels up to halfway to the nearest other trace on each side, at most max_half_width away.
    Pixel x is owned if left <= x < right.

    Args:
        spaxel: Spaxel index.
        centres: Output of trace_centres.
        max_half_width: Largest distance from the trace centre (px).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (rows, left edges, right edges), one entry per spectral element.
    """
    a0, centre = centres[spaxel]
    rows = a0 + np.arange(N_ELEMENTS)
    left_gap = np.full(N_ELEMENTS, np.inf)
    right_gap = np.full(N_ELEMENTS, np.inf)
    for sid, (other_a0, other_centre) in centres.items():
        if sid == spaxel:
            continue
        elements = rows - other_a0
        present = (elements >= 0) & (elements < N_ELEMENTS)
        offset = np.full(N_ELEMENTS, np.nan)
        offset[present] = other_centre[elements[present]] - centre[present]
        right_gap = np.where(offset > 0, np.minimum(right_gap, offset), right_gap)
        left_gap = np.where(offset < 0, np.minimum(left_gap, -offset), left_gap)
    left_half = np.minimum(left_gap / 2, max_half_width)
    right_half = np.minimum(right_gap / 2, max_half_width)
    return rows, centre - left_half, centre + right_half


def noise_model(
    science_image: np.ndarray, model: np.ndarray, region: np.ndarray | None = None
) -> tuple[float, float, float]:
    """Pixel noise model sigma**2 = a + b * flux + c * flux**2, measured from the data.

    Uses differences between adjacent rows of science - model, where slow model errors cancel; c captures
    pixel-to-pixel response structure (about 0.35% of the flux in the blue deep sky flat).

    Args:
        science_image: The image.
        model: Model of the image.
        region: Pixels to use (default model > 0).

    Returns:
        tuple[float, float, float]: (a, b, c).
    """
    science, model = np.asarray(science_image, dtype=float), np.asarray(model, dtype=float)
    use = (model > 0) if region is None else np.asarray(region, dtype=bool) & (model > 0)
    pair = use[1:] & use[:-1]
    diff = ((science[1:] - model[1:]) - (science[:-1] - model[:-1]))[pair] / np.sqrt(2)
    flux = 0.5 * (model[1:] + model[:-1])[pair]
    ok = np.isfinite(diff) & np.isfinite(flux)
    diff, flux = diff[ok], flux[ok]
    edges = np.unique(np.quantile(flux, np.linspace(0, 1, 21)))
    which = np.clip(np.digitize(flux, edges) - 1, 0, len(edges) - 2)
    centres_, variances = [], []
    for k in range(len(edges) - 1):
        d = diff[which == k]
        if len(d) > 50:
            centres_.append(np.median(flux[which == k]))
            variances.append((1.4826 * np.median(np.abs(d - np.median(d)))) ** 2)
    # Non-negative fit, each bin weighted by its own variance so faint and bright bins count alike
    flux_bins, variances = np.asarray(centres_), np.asarray(variances)
    design = np.stack([np.ones_like(flux_bins), flux_bins, flux_bins**2], axis=1) / variances[:, None]
    a, b, c = nnls(design, np.ones_like(variances))[0]
    return float(a), float(b), float(c)


def noise_sigma(noise: tuple[float, float, float], flux: np.ndarray) -> np.ndarray:
    """Pixel noise sigma at a given flux.

    Args:
        noise: Output of noise_model.
        flux: Model flux.

    Returns:
        np.ndarray: Sigma for each flux value.
    """
    a, b, c = noise
    flux = np.clip(flux, 0, None)
    return np.sqrt(a + b * flux + c * flux**2)


def _own_pixel_crop(spaxel: int, centres: dict[int, tuple[int, np.ndarray]], rows: slice, cols: slice) -> np.ndarray:
    """Mask of a spaxel's own pixels (see own_columns) over a crop.

    Args:
        spaxel: Spaxel index.
        centres: Output of trace_centres.
        rows: Row range of the crop.
        cols: Column range of the crop.

    Returns:
        np.ndarray: Boolean mask of the crop's shape.
    """
    own_rows, left, right = own_columns(spaxel, centres)
    mask = np.zeros((rows.stop - rows.start, cols.stop - cols.start), dtype=bool)
    x = np.arange(cols.start, cols.stop)
    inside = (own_rows >= rows.start) & (own_rows < rows.stop)
    for r, lo, hi in zip(own_rows[inside], left[inside], right[inside], strict=True):
        mask[r - rows.start] = (x >= lo) & (x < hi)
    return mask


def _own_fraction(science: np.ndarray, model: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Residual relative to each row's own peak.

    Args:
        science: Science crop.
        model: Model crop.
        mask: Own pixels of the crop.

    Returns:
        tuple[np.ndarray, np.ndarray]: |science - model| / own peak (NaN outside the mask), and the own peak per row.
    """
    own_science = np.where(mask, science, np.nan)
    with np.errstate(all="ignore"):
        peak = np.nanmax(own_science, axis=1, keepdims=True)
        frac = np.where(mask, np.abs(science - model) / peak, np.nan)
    return frac, peak[:, 0]


def residual_stats(
    science_image: np.ndarray,
    model: np.ndarray,
    spaxel: int,
    valid_mask: np.ndarray | None = None,
    centres: dict[int, tuple[int, np.ndarray]] | None = None,
    noise: tuple[float, float, float] | None = None,
) -> dict[str, float]:
    """Statistics of |science - model| relative to a row peak for one spaxel.

    Without centres: every pixel of the spaxel's +-50 column crop, normalised by the crop's row peak. With centres:
    only the spaxel's own pixels over all its rows, normalised by the row's own peak.

    Args:
        science_image: The image.
        model: Model of the image.
        spaxel: Spaxel index.
        valid_mask: Pixels to include (default all).
        centres: Output of trace_centres, for the own-pixel metric.
        noise: Output of noise_model; adds noise_ceiling (the <1% fraction a perfect model would get) and
            noise_limited (fraction of pixels where 1% of the own peak is under NOISE_SIGMAS sigma).

    Returns:
        dict[str, float]: "mean", "median", "p90", "under_1pct", plus the noise entries if noise is given.
    """
    a0, a1 = int(A0_PARAMS[spaxel]), int(A1_PARAMS[spaxel]) + 1
    b0, b1 = max(int(B0_PARAMS[spaxel]) - 50, 0), int(B1_PARAMS[spaxel]) + 51
    if centres is None:
        science = np.asarray(science_image)[a0:a1, b0:b1]
        frac = np.abs(science - np.asarray(model)[a0:a1, b0:b1]) / np.nanmax(science, axis=1, keepdims=True)
        if valid_mask is not None:
            frac = frac[np.asarray(valid_mask)[a0:a1, b0:b1]]
        return {
            "mean": float(np.nanmean(frac)),
            "median": float(np.nanmedian(frac)),
            "p90": float(np.nanpercentile(frac, 90)),
            "under_1pct": float(np.nanmean(frac < 0.01)),
        }

    height, width = np.shape(science_image)
    rows = slice(a0, min(a0 + N_ELEMENTS, height))
    cols = slice(b0, min(b1, width))  # crops of spaxels at the detector edge run past it
    mask = _own_pixel_crop(spaxel, centres, rows, cols)
    if valid_mask is not None:
        mask &= np.asarray(valid_mask)[rows, cols]
    model_crop = np.asarray(model)[rows, cols]
    frac, peak = _own_fraction(np.asarray(science_image)[rows, cols], model_crop, mask)
    used = mask & np.isfinite(frac)
    values = frac[used]
    stats = {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "under_1pct": float(np.mean(values < 0.01)),
    }
    if noise is not None:
        sigma = noise_sigma(noise, model_crop)[used]
        threshold = 0.01 * np.broadcast_to(peak[:, None], mask.shape)[used]
        stats["noise_ceiling"] = float(np.mean(erf(threshold / (np.sqrt(2) * sigma))))
        stats["noise_limited"] = float(np.mean(threshold < NOISE_SIGMAS * sigma))
    return stats


def plot_fit_difference(
    science_image: np.ndarray,
    initial_model: np.ndarray,
    final_model: np.ndarray,
    spaxel: int,
    output_path: Path,
    pad: int = 10,
    centres: dict[int, tuple[int, np.ndarray]] | None = None,
    noise: tuple[float, float, float] | None = None,
) -> Path:
    """Plot one spaxel's initial and best-fit models, their residuals and the best fit's percentage error.

    The bottom-right panel is 100 * (science - best fit) / row peak on a +-2% scale; with centres it uses each
    row's own peak and shows only the spaxel's own pixels.

    Args:
        science_image: The image.
        initial_model: Model before the fit.
        final_model: Model after the fit.
        spaxel: Spaxel index.
        output_path: PNG to write.
        pad: Rows shown beyond the spaxel's footprint.
        centres: Output of trace_centres, for the own-pixel metric.
        noise: Output of noise_model, for the noise ceiling in the title.

    Returns:
        Path: output_path.
    """
    a0, a1 = int(A0_PARAMS[spaxel]), int(A1_PARAMS[spaxel]) + 1
    b0, b1 = max(int(B0_PARAMS[spaxel]) - 50, 0), int(B1_PARAMS[spaxel]) + 51
    height, width = np.shape(science_image)
    rows, cols = slice(max(a0 - pad, 0), min(a1 + pad, height)), slice(b0, min(b1, width))
    extent = (cols.start, cols.stop, rows.start, rows.stop)

    science = np.asarray(science_image)[rows, cols]
    initial = np.asarray(initial_model)[rows, cols]
    final = np.asarray(final_model)[rows, cols]
    resid_initial, resid_final = science - initial, science - final
    model_change = final - initial
    if centres is None:
        percent_error = 100 * resid_final / np.nanmax(science, axis=1, keepdims=True)
        percent_label = "Best fit % error (of row peak)"
    else:
        own = _own_pixel_crop(spaxel, centres, rows, cols)
        frac, _ = _own_fraction(science, final, own)
        percent_error = 100 * np.sign(resid_final) * frac
        percent_label = "Best fit % error (of own row peak, own pixels)"
    percent_cmap = CMAP_DIFF.copy()
    percent_cmap.set_bad("#d9d9d6")

    data_lo, data_hi = np.nanpercentile(np.stack([initial, final]), [1, 99])
    resid_lim = float(max(np.nanpercentile(np.abs(resid_initial), 99), np.nanpercentile(np.abs(resid_final), 99)))
    change_lim = max(float(np.nanpercentile(np.abs(model_change), 99.5)), 1e-9)
    percent_lim = 2.0

    panels = [
        # (row, col, data, label, cmap, vmin, vmax)
        (0, 0, initial, "Model, initial", CMAP_DATA, data_lo, data_hi),
        (0, 1, final, "Model, best fit", CMAP_DATA, data_lo, data_hi),
        (0, 2, model_change, "Model diff (best fit - initial)", CMAP_DIFF, -change_lim, change_lim),
        (1, 0, resid_initial, "Science - initial", CMAP_DIFF, -resid_lim, resid_lim),
        (1, 1, resid_final, "Science - best fit", CMAP_DIFF, -resid_lim, resid_lim),
        (1, 2, percent_error, percent_label, percent_cmap, -percent_lim, percent_lim),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 12), gridspec_kw={"hspace": 0.2, "wspace": 0.2})
    stats = {
        name: residual_stats(science_image, model, spaxel, centres=centres, noise=noise)
        for name, model in (("initial", initial_model), ("best fit", final_model))
    }
    summary = "; ".join(
        f"{name}: mean {s['mean']:.1%}, median {s['median']:.1%}, <1% {s['under_1pct']:.0%}"
        for name, s in stats.items()
    )
    metric = "|resid|/row peak" if centres is None else "|resid|/own row peak, own pixels"
    if "noise_ceiling" in stats["best fit"]:
        s = stats["best fit"]
        summary += (
            f"\nnoise ceiling (perfect model) <1% {s['noise_ceiling']:.0%}; "
            f"noise-limited pixels {s['noise_limited']:.0%}"
        )
    axes[0, 0].annotate(
        f"Spaxel {spaxel} - initial vs best fit (columns {cols.start}-{cols.stop}, rows {rows.start}-{rows.stop})\n"
        f"{metric}, {summary}",
        xy=(0, 1.01),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=10,
    )
    for row, col, data, label, cmap, vmin, vmax in panels:
        ax = axes[row, col]
        im = ax.imshow(
            data, origin="lower", interpolation="none", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent
        )
        add_colorbar(f"{label} (mean={np.nanmean(data):.2f}, std{np.nanstd(data):.2f})", fig, ax, im)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
