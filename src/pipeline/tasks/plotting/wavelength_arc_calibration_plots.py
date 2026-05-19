from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy import sparse

from pipeline import settings
from pipeline.common import Image, get_logger, pipeline_task
from pipeline.common.model_params import A0_PARAMS, A1_PARAMS, B0_PARAMS, B1_PARAMS
from pipeline.common.plotting_utils import get_all_peaks
from pipeline.resolver.resolver import PUBLIC_PATH_MAP, get_run_id
from pipeline.tasks.plotting.plots import plot_standalone

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

    # if the dict has more entries than models, drop from the front
    n_models = len(models)
    if len(sorted_items) > n_models:
        sorted_items = sorted_items[len(sorted_items) - n_models :]
    elif len(sorted_items) < n_models:
        raise ValueError(
            f"coeffs_dict['{outer_key}'] has only {len(sorted_items)} entries but {n_models} models were provided."
        )

    iter_labels = [kv[0] for kv in sorted_items]
    n_lines = len(sorted_items)

    row_lo, row_hi = row_range
    col_lo, col_hi = col_range
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

    a0 = int(A0_PARAMS[int(outer_key)])
    a1 = int(A1_PARAMS[int(outer_key)]) + 1
    b0 = int(B0_PARAMS[int(outer_key)]) - 50
    off = 50
    if b0 < 0:
        off += b0
        b0 = 0
    b1 = int(B1_PARAMS[int(outer_key)]) + 50 + 1

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

    output_file = f"poly_convergence_{outer_key}.gif"
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
