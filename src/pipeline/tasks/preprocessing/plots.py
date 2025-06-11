import shutil
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar

import cmasher as cmr
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage
from matplotlib.ticker import MaxNLocator

from pipeline import settings
from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_task
from pipeline.resolver.common import FileStoreEntry
from pipeline.tasks.common import Image, Section, get_section_range

_DATA_STORE: dict[str, list[Image]] = {}

P = ParamSpec("P")
R = TypeVar("R")
ZOOM_START = (1024 + 32 - 100, 512)
ZOOM_SIZE = (100, 100)
ZOOM_END = (ZOOM_START[0] + ZOOM_SIZE[0], ZOOM_START[1] + ZOOM_SIZE[1])
MIDLINE_X_COORD = ZOOM_START[0] + ZOOM_SIZE[0] // 2
MIDLINE_Y_COORD = ZOOM_START[1] + ZOOM_SIZE[1] // 2
MIDLINE_HORIZONTAL_COLOUR = "#146727"
MIDLINE_VERTICAL_COLOUR = "#755028"

# I love the tailwind colours, and you can get them from the tailwind site
# or this fun tool: https://tailscan.com/colors
LINES_X: dict[int, str] = {
    # 256: "#9f1239",
    525: "#9f1239",
    1032: "#d97706",
}
LINES_Y: dict[int, str] = {
    # 10: "#86198f",
    # 512: "#86198f",
    # 1024: "#22c55e",
    # 1536: "#0e7490",
    # 2048: "#fb7185",
    4090: "#fb7185",
}


def determine_output_path(primary: FileStoreEntry) -> Path:
    run_id = "run_id=" + (primary.run_id or "unknown")
    channel = "channel=" + (primary.channel or "unknown")
    obstype = "obstype=" + (primary.type.value or "unknown")
    output_path = settings.output_path / "plots" / run_id / obstype / channel
    shutil.rmtree(output_path, ignore_errors=True)  # type: ignore
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def determine_figure_prefix(primary: FileStoreEntry) -> str:
    """Determine the figure title based on the primary file's metadata."""
    run_id = primary.run_id or "unknown"
    channel = primary.channel or "unknown"
    obstype = primary.type.value or "unknown"
    return f"{obstype=} - {run_id=} - {channel=}"


def ensure_list[T](x: T | list[T]) -> list[T]:
    """Ensure that the input is a list."""
    if isinstance(x, list):
        return x
    return [x]


def plot():
    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def inner(images: Image | list[Image], *args, **kwargs) -> Image | list[Image]:
            if not _DATA_STORE:
                _DATA_STORE["initial"] = ensure_list(images)
            result = func(images, *args, **kwargs)  # type: ignore
            _DATA_STORE[func.__name__] = ensure_list(result)
            return result

        return inner  # type: ignore

    return decorate


def log_image_data(name: str, images: Image | list[Image]) -> None:
    if isinstance(images, Image):
        images = [images]

    if not _DATA_STORE:
        _DATA_STORE["initial"] = images

    _DATA_STORE[name] = images


def extract_zoom(data: np.ndarray) -> np.ndarray:
    return data[ZOOM_START[0] : ZOOM_START[0] + ZOOM_SIZE[0], ZOOM_START[1] : ZOOM_START[1] + ZOOM_SIZE[1]]


def add_colorbar(label: str, fig: plt.Figure, ax: plt.Axes, im: AxesImage, height: float = 0.02) -> None:  # type: ignore
    cbax = ax.inset_axes([0, -height, 1.0, height], transform=ax.transAxes)  # type: ignore
    cbar = fig.colorbar(im, cax=cbax, orientation="horizontal", format="%1g")
    cbar.set_label(label, size=8)
    cbar.ax.tick_params(rotation=45, labelsize=6)


def add_callout_rectangle(ax: plt.Axes) -> None:  # type: ignore
    """Add a callout rectangle to the axes."""
    rect = patches.Rectangle(ZOOM_START, ZOOM_SIZE[0], ZOOM_SIZE[1], linewidth=0.5, edgecolor="r", facecolor="none")
    ax.add_patch(rect)


def add_section_rectangle(ax: plt.Axes, section: Section, **kwargs) -> None:  # type: ignore
    rect = patches.Rectangle(
        (section.x_min, section.y_min),
        section.x_max - section.x_min,
        section.y_max - section.y_min,
        facecolor="none",
        linewidth=0.5,
        **kwargs,
    )
    ax.add_patch(rect)


def add_ticks(ax: plt.Axes, locations: dict[int, str], axis: str = "y", reach: float = 0.02) -> None:  # type: ignore
    # Matplotlib won't let you do different coloured ticks, so we'll do it ourselves.
    for location, colour in locations.items():
        if axis == "y":
            ax.axhline(location, xmin=0, xmax=reach, color=colour)
            ax.axhline(location, xmin=(1 - reach), xmax=1, color=colour)
        elif axis == "x":
            ax.axvline(location, ymin=0, ymax=reach, color=colour)
            ax.axvline(location, ymin=(1 - reach), ymax=1, color=colour)


def add_midlines(ax: plt.Axes) -> None:  # type: ignore
    ax.hlines(
        MIDLINE_Y_COORD,
        xmin=ZOOM_START[0],
        xmax=ZOOM_END[0],
        color=MIDLINE_HORIZONTAL_COLOUR,
        lw=0.5,
    )
    ax.vlines(
        MIDLINE_X_COORD,
        ymin=ZOOM_START[1],
        ymax=ZOOM_END[1],
        color=MIDLINE_VERTICAL_COLOUR,
        lw=0.5,
    )
    ax.set_xmargin(0)
    ax.set_ymargin(0)


@pipeline_task()
def plot_images(primary: FileStoreEntry) -> None:  # noqa: C901
    # TODO: full image run id, type, channel on image itself
    # TODO: add a dotted fun box around the biassec if it exists
    """Plot the images in the data store."""
    logger = get_logger()
    if not settings.plot:
        logger.info("Plotting is disabled. Skipping plot generation.")
        return

    output_path = determine_output_path(primary)
    title_prefix = determine_figure_prefix(primary)

    cmap_data = cmr.torch
    cmap_zoom = cmr.rainforest
    cmap_diff = cmr.prinsenvlag
    # One column for data, one for variance

    all_data = np.concatenate(
        [im.data.astype(np.float64).flatten() for images in _DATA_STORE.values() for im in images]
    )
    min_c_data, max_c_data = np.percentile(all_data, [1, 99])

    prior_images = None
    for i, (key, images) in enumerate(_DATA_STORE.items()):
        title = f"{title_prefix} - {key}"
        num_cols = len(images) * 4
        aspect_ratio = images[0].data.shape[1] / images[0].data.shape[0]

        fig, axes = plt.subplots(
            5,
            num_cols,
            figsize=(num_cols * 1.5 + 7, 16),
            gridspec_kw={"hspace": 0.3, "wspace": 0.2},
            height_ratios=[aspect_ratio, 1, 1, 1, 1],
        )
        axes[0, 0].annotate(title, xy=(0, 1.01), xycoords="axes fraction", ha="left", va="bottom", fontsize=10)
        for k, image in enumerate(images):
            # Yeah this is ugly.
            axd, axv = axes[0, 4 * k], axes[0, 4 * k + 2]  # data and variance axes
            axdz, axvz = axes[1, 4 * k], axes[1, 4 * k + 2]  # zoomed data and variance axes
            axdd, axvd = axes[0, 4 * k + 1], axes[0, 4 * k + 3]  # data and variance difference axes
            axdzd, axvzd = axes[1, 4 * k + 1], axes[1, 4 * k + 3]  # zoomed data and variance difference axes
            axxdl, axxvl = axes[2, 4 * k], axes[2, 4 * k + 2]  # line plots for data and variance in x
            axydl, axyvl = axes[3, 4 * k], axes[3, 4 * k + 2]  # line plots for data and variance in y
            axxddl, axxvdl = axes[2, 4 * k + 1], axes[2, 4 * k + 3]  # line plots for data and variance difference for x
            axyddl, axyvdl = axes[3, 4 * k + 1], axes[3, 4 * k + 3]  # line plots for data and variance difference for y
            axxc, axyc = axes[4, 4 * k], axes[4, 4 * k + 2]  # axes for the callout midline line plots
            axxcd, axycd = (axes[4, 4 * k + 1], axes[4, 4 * k + 3])  # axes for the callout midline differences

            data, variance = image.data.astype(np.float64), image.variance.astype(np.float64)
            data[~np.isfinite(data)] = np.nan
            variance[~np.isfinite(variance)] = np.nan
            im_kw = {
                "origin": "lower",
                "interpolation": "none",
            }

            imd = axd.imshow(data.T, cmap=cmap_data, aspect="equal", vmin=min_c_data, vmax=max_c_data, **im_kw)
            add_colorbar(f"Data {k}", fig, axd, imd)
            imv = axv.imshow(variance.T, cmap=cmap_data, aspect="equal", vmin=min_c_data, vmax=max_c_data, **im_kw)
            add_colorbar(f"Variance {k}", fig, axv, imv)

            imdz = axdz.imshow(
                extract_zoom(data).T,
                cmap=cmap_zoom,
                aspect="auto",
                **im_kw,
            )
            add_colorbar(f"Zoomed Data {k}", fig, axdz, imdz, height=0.04)
            imvz = axvz.imshow(
                extract_zoom(variance).T,
                cmap=cmap_zoom,
                aspect="auto",
                **im_kw,
            )
            add_colorbar(f"Zoomed Variance {k}", fig, axvz, imvz, height=0.04)

            # Try to add the delta images if possible.
            if prior_images is not None and len(prior_images) == len(images):
                prior_data = prior_images[k].data.astype(np.float64)
                prior_data[~np.isfinite(prior_data)] = np.nan
                prior_variance = prior_images[k].variance.astype(np.float64)
                prior_variance[~np.isfinite(prior_variance)] = np.nan
                data_diff = np.round(data - prior_data, 6)
                variance_diff = np.round(variance - prior_variance, 6)
            else:
                data_diff = np.zeros_like(data)
                variance_diff = np.zeros_like(variance)

            vdmin, vdmax, vvmin, vvmax = None, None, None, None
            # if there is a constant difference (within rounding issues) we want the vmin and vmax to be set to the
            # maxabs and minabs so we don't get white images that people may confuse with zero-change deltas
            if np.std(data_diff) < 1e-5:
                vdmin = np.min(np.abs(data_diff))
                vdmax = np.max(np.abs(data_diff))
            if np.std(variance_diff) < 1e-5:
                vvmin = np.min(np.abs(variance_diff))
                vvmax = np.max(np.abs(variance_diff))

            imdd = axdd.imshow(data_diff.T, cmap=cmap_diff, aspect="equal", vmin=vdmin, vmax=vdmax, **im_kw)
            add_colorbar("ΔData", fig, axdd, imdd)
            imvd = axvd.imshow(variance_diff.T, cmap=cmap_diff, aspect="equal", vmin=vvmin, vmax=vvmax, **im_kw)
            add_colorbar("ΔVar", fig, axvd, imvd)

            zoomed_data_diff = extract_zoom(data_diff)
            imddz = axdzd.imshow(zoomed_data_diff.T, cmap=cmap_diff, aspect="auto", **im_kw)
            add_colorbar("Zoomed ΔData", fig, axdzd, imddz, height=0.04)

            zoomed_variance_diff = extract_zoom(variance_diff)
            imvdz = axvzd.imshow(zoomed_variance_diff.T, cmap=cmap_diff, aspect="auto", **im_kw)
            add_colorbar("Zoomed ΔVar", fig, axvzd, imvdz, height=0.04)

            for ax in (axd, axv, axdd, axvd):
                add_callout_rectangle(ax)
                if "BIASSEC" in image.header:
                    # TODO: Make this class constructor
                    bias_section = get_section_range(image.header.get_str("BIASSEC"))
                    add_section_rectangle(ax, bias_section, edgecolor="#38bdf8", linestyle=":")

            # Now we add some line plots for better readability
            kwargs = {"lw": 0.5}
            y_lim_percentages = [1, 99]
            for ax, sec in [(axxdl, data), (axxvl, variance), (axxddl, data_diff), (axxvdl, variance_diff)]:
                for location, colour in LINES_X.items():
                    ax.plot(sec[location, :], color=colour, **kwargs)

                dx = sec[list(LINES_X.keys()), :]
                ax.set_ylim(*np.nanpercentile(dx, y_lim_percentages))

            for ax, sec in [(axydl, data), (axyvl, variance), (axyddl, data_diff), (axyvdl, variance_diff)]:
                for location, colour in LINES_Y.items():
                    ax.plot(sec[:, location], color=colour, **kwargs)
                dy = sec[:, list(LINES_Y.keys())]
                ax.set_ylim(*np.nanpercentile(dy, y_lim_percentages))

            # We also want to add the midlines to the midline axes
            for ax, sec in [(axxc, data), (axyc, variance), (axxcd, data_diff), (axycd, variance_diff)]:
                column = sec[MIDLINE_X_COORD, ZOOM_START[1] : ZOOM_END[1]]
                ax.step(np.arange(column.size), column, where="post", color=MIDLINE_VERTICAL_COLOUR, **kwargs)
                row = sec[ZOOM_START[0] : ZOOM_END[0], MIDLINE_Y_COORD]
                ax.step(np.arange(row.size), row, where="post", color=MIDLINE_HORIZONTAL_COLOUR, **kwargs)
                combined = np.concatenate((column.flatten(), row.flatten()))
                ax.set_ylim(*np.nanpercentile(combined, y_lim_percentages))

            # We also want to mark this on the axes above as well. Not as a line,
            # as there would be too many, but as etra tick marks on the side of the axes.
            for ax in (axd, axv, axdd, axvd):
                add_ticks(ax, LINES_X, axis="x", reach=0.02)
                add_ticks(ax, LINES_Y, axis="y", reach=0.05)
                add_midlines(ax)

            for ax in (axd, axv, axdz, axvz, axdd, axvd, axdzd, axvzd):
                ax.set_xticks([])
                ax.set_yticks([])
            for ax in (axxdl, axxvl, axxddl, axxvdl, axydl, axyvl, axyddl, axyvdl, axxc, axyc, axxcd, axycd):
                ax.set_yticks([])
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
                ax.tick_params(axis="y", labelsize=6, labelrotation=90)
                ax.tick_params(axis="both", labelsize=6)
                ax.set_xmargin(0)

            # Add some more labels
            axxc.set_xlabel("Data callout midlines", fontsize=8)
            axyc.set_xlabel("Variance callout midlines", fontsize=8)
            axxcd.set_xlabel("ΔData midlines", fontsize=8)
            axycd.set_xlabel("ΔVariance midlines", fontsize=8)
            axxdl.set_xlabel("Data rows", fontsize=8)
            axxvl.set_xlabel("Variance rows", fontsize=8)
            axydl.set_xlabel("Data columns", fontsize=8)
            axyvl.set_xlabel("Variance columns", fontsize=8)
            axxddl.set_xlabel("ΔData rows", fontsize=8)
            axyddl.set_xlabel("ΔData columns", fontsize=8)
            axxvdl.set_xlabel("ΔVariance rows", fontsize=8)
            axyvdl.set_xlabel("ΔVariance columns", fontsize=8)
        output_location = output_path / f"{i}_{key}.png"
        logger.info(f"Saving plot to {output_location}")
        fig.savefig(output_location, dpi=450, bbox_inches="tight")
        plt.close(fig)

        prior_images = images
