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
ZOOM_START = (512, 512)
ZOOM_SIZE = (100, 100)

# I love the tailwind colours, and you can get them from the tailwind site
# or this fun tool: https://tailscan.com/colors
LINES_X: dict[int, str] = {
    256: "#9f1239",
    525: "#d97706",
}
LINES_Y: dict[int, str] = {
    512: "#86198f",
    1024: "#22c55e",
    1536: "#0e7490",
    2048: "#fb7185",
}


def determine_output_path(primary: FileStoreEntry) -> Path:
    run_id = "run_id=" + (primary.run_id or "unknown")
    channel = "channel=" + (primary.channel or "unknown")
    obstype = "obstype=" + (primary.type or "unknown")
    output_path = settings.output_path / "plots" / run_id / obstype / channel
    shutil.rmtree(output_path, ignore_errors=True)  # type: ignore
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


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
            ax.axvline(location, ymin=reach, ymax=2 * reach, color=colour)
            ax.axvline(location, ymin=(1 - reach), ymax=1, color=colour)


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

    cmap_data = cmr.torch
    cmap_zoom = cmr.rainforest
    cmap_diff = cmr.prinsenvlag
    # One column for data, one for variance

    all_data = np.concatenate(
        [im.data.astype(np.float64).flatten() for images in _DATA_STORE.values() for im in images]
    )
    all_data_zooms = np.concatenate(
        [extract_zoom(im.data).astype(np.float64).flatten() for images in _DATA_STORE.values() for im in images]
    )

    min_c_data, max_c_data = np.percentile(all_data, [1, 99])
    min_c_data_zoom, max_c_data_zoom = np.percentile(all_data_zooms, [1, 99])

    prior_images = None
    for i, (key, images) in enumerate(_DATA_STORE.items()):
        num_cols = len(images) * 4
        aspect_ratio = images[0].data.shape[1] / images[0].data.shape[0]

        fig, axes = plt.subplots(
            4,
            num_cols,
            figsize=(num_cols * 1.5 + 5, 14),
            gridspec_kw={"hspace": 0.2, "wspace": 0.2},
            height_ratios=[aspect_ratio, 1, 1, 1],
        )
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

            data, variance = image.data.astype(np.float64), image.variance.astype(np.float64)
            data[~np.isfinite(data)] = np.nan
            variance[~np.isfinite(variance)] = np.nan

            imd = axd.imshow(data.T, cmap=cmap_data, aspect="equal", origin="lower", vmin=min_c_data, vmax=max_c_data)
            add_colorbar(f"Data {k}", fig, axd, imd)
            imv = axv.imshow(
                variance.T, cmap=cmap_data, aspect="equal", origin="lower", vmin=min_c_data, vmax=max_c_data
            )
            add_colorbar(f"Variance {k}", fig, axv, imv)

            imdz = axdz.imshow(
                extract_zoom(data).T,
                cmap=cmap_zoom,
                aspect="auto",
                origin="lower",
                vmin=min_c_data_zoom,
                vmax=max_c_data_zoom,
            )
            add_colorbar(f"Zoomed Data {k}", fig, axdz, imdz, height=0.04)
            imvz = axvz.imshow(
                extract_zoom(variance).T,
                cmap=cmap_zoom,
                aspect="auto",
                origin="lower",
                vmin=min_c_data_zoom,
                vmax=max_c_data_zoom,
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

            imdd = axdd.imshow(data_diff.T, cmap=cmap_diff, aspect="equal", origin="lower", vmin=vdmin, vmax=vdmax)
            add_colorbar("ΔData", fig, axdd, imdd)
            imvd = axvd.imshow(variance_diff.T, cmap=cmap_diff, aspect="equal", origin="lower", vmin=vvmin, vmax=vvmax)
            add_colorbar("ΔVar", fig, axvd, imvd)

            zoomed_data_diff = extract_zoom(data_diff)
            imddz = axdzd.imshow(zoomed_data_diff.T, cmap=cmap_diff, aspect="auto", origin="lower")
            add_colorbar("Zoomed ΔData", fig, axdzd, imddz, height=0.04)

            zoomed_variance_diff = extract_zoom(variance_diff)
            imvdz = axvzd.imshow(zoomed_variance_diff.T, cmap=cmap_diff, aspect="auto", origin="lower")
            add_colorbar("Zoomed ΔVar", fig, axvzd, imvdz, height=0.04)

            for ax in (axd, axv, axdd, axvd):
                add_callout_rectangle(ax)
                if "BIASSEC" in image.header:
                    # TODO: Make this class constructor
                    bias_section = get_section_range(image.header.get_str("BIASSEC"))
                    add_section_rectangle(ax, bias_section, edgecolor="#38bdf8", linestyle=":")

            # Now we add some line plots for better readability
            kwargs = {"lw": 0.5}
            for location, colour in LINES_X.items():
                axxdl.plot(data[location, :], color=colour, **kwargs)
                axxvl.plot(variance[location, :], color=colour, **kwargs)
                axxddl.plot(data_diff[location, :], color=colour, **kwargs)
                axxvdl.plot(variance_diff[location, :], color=colour, **kwargs)

            for location, colour in LINES_Y.items():
                axydl.plot(data[:, location], color=colour, **kwargs)
                axyvl.plot(variance[:, location], color=colour, **kwargs)
                axyddl.plot(data_diff[:, location], color=colour, **kwargs)
                axyvdl.plot(variance_diff[:, location], color=colour, **kwargs)

            # We also want to mark this on the axes above as well. Not as a line,
            # as there would be too many, but as etra tick marks on the side of the axes.
            for ax in (axd, axv, axdd, axvd):
                add_ticks(ax, LINES_X, axis="x", reach=0.02)
                add_ticks(ax, LINES_Y, axis="y", reach=0.05)

            if k == 0:
                axd.set_title(key, size=8)

            for ax in (axd, axv, axdz, axvz, axdd, axvd, axdzd, axvzd):
                ax.set_xticks([])
                ax.set_yticks([])
            for ax in (axxdl, axxvl, axxddl, axxvdl, axydl, axyvl, axyddl, axyvdl):
                ax.set_yticks([])
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
                ax.tick_params(axis="y", labelsize=6, labelrotation=90)
                ax.tick_params(axis="both", labelsize=6)
                ax.set_xmargin(0)
        output_location = output_path / f"{i}_{key}.png"
        logger.info(f"Saving plot to {output_location}")
        fig.savefig(output_location, dpi=600, bbox_inches="tight")
        plt.close(fig)

        prior_images = images
