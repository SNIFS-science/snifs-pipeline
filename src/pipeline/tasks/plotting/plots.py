import contextlib
from collections import defaultdict
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, OrderedDict, ParamSpec, TypeVar

import cmasher as cmr
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import prefect
from matplotlib.image import AxesImage
from matplotlib.ticker import MaxNLocator

from pipeline import settings
from pipeline.common import Image, Section, get_logger, pipeline_task
from pipeline.resolver.common import FileStoreEntry

_IMAGE_STORE: dict[str, dict[str, list[Image]]] = defaultdict(OrderedDict)
_BIAS_STORE: dict[str, dict[str, list[Image]]] = defaultdict(OrderedDict)

P = ParamSpec("P")
R = TypeVar("R")
ZOOM_START = (1000, 509 - 25)
ZOOM_SIZE = (50, 50)
ZOOM_END = (ZOOM_START[0] + ZOOM_SIZE[0], ZOOM_START[1] + ZOOM_SIZE[1])
MIDLINE_X_COORD = ZOOM_START[0] + ZOOM_SIZE[0] // 2
MIDLINE_Y_COORD = ZOOM_START[1] + ZOOM_SIZE[1] // 2
MIDLINE_HORIZONTAL_COLOUR = "#146727"
MIDLINE_VERTICAL_COLOUR = "#755028"

# I love the tailwind colours, and you can get them from the tailwind site
# or this fun tool: https://tailscan.com/colors
LINES_X: dict[int, str] = {
    7: "#e4e23b",
    1022: "#9f1239",
    1032: "#d97706",
}
LINES_Y: dict[int, str] = {
    128: "#601d37",
    1536: "#0e7490",
    2048: "#fb7185",
    4090: "#ffafbb",
}
CMAP_DATA = cmr.torch
CMAP_ZOOM = cmr.rainforest
CMAP_DIFF = cmr.prinsenvlag


def determine_figure_prefix(primary: FileStoreEntry) -> str:
    """Determine the figure title based on the primary file's metadata."""
    run_id = primary.run_id or "unknown"
    channel = primary.channel or "unknown"
    obstype = primary.type.value or "unknown"
    return f"{obstype=} - {run_id=} - {channel=}"


def ensure_list[T](x: T | list[T] | tuple[T]) -> list[T]:
    """Ensure that the input is a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    return [x]


def flatten_and_filter[T](inputs: Any, filter_type: type[T]) -> list[T]:
    """Flatten a nested structure and filter out elements of a specific type."""
    if isinstance(inputs, (list, tuple)):
        return [
            item
            for sublist in inputs
            for item in flatten_and_filter(sublist, filter_type)
            if isinstance(item, filter_type)
        ]
    elif isinstance(inputs, filter_type):
        return [inputs]
    return []


def get_run_id() -> str:
    try:
        return prefect.context.FlowRunContext.get().flow_run.id
    except Exception:
        return "unknown"


def plot():
    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def inner(images: Image | list[Image], *args, **kwargs) -> Image | list[Image]:
            flow_run_id = get_run_id()
            if not _IMAGE_STORE[flow_run_id]:
                _IMAGE_STORE[flow_run_id]["initial"] = flatten_and_filter(images, Image)
            result = func(images, *args, **kwargs)  # type: ignore
            _IMAGE_STORE[flow_run_id][func.__name__] = flatten_and_filter(result, Image)
            return result

        return inner  # type: ignore

    return decorate


def plot_bias():
    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def inner(images: Image | list[Image], *args, **kwargs) -> Image | list[Image]:
            flow_run_id = get_run_id()
            if not _BIAS_STORE[flow_run_id]:
                _BIAS_STORE[flow_run_id]["initial"] = ensure_list(images)
            result = func(images, *args, **kwargs)  # type: ignore
            _BIAS_STORE[flow_run_id][func.__name__] = ensure_list(result)
            return result

        return inner  # type: ignore

    return decorate


def log_image_data(name: str, images: Image | list[Image]) -> None:
    if isinstance(images, Image):
        images = [images]
    flow_run_id = prefect.context.get_run_context().flow_run.id if prefect.context.get_run_context() else "unknown"
    if not _IMAGE_STORE[flow_run_id]:
        _IMAGE_STORE[flow_run_id]["initial"] = images

    _IMAGE_STORE[flow_run_id][name] = images


def extract_zoom(data: np.ndarray) -> np.ndarray:
    return data[ZOOM_START[0] : ZOOM_START[0] + ZOOM_SIZE[0], ZOOM_START[1] : ZOOM_START[1] + ZOOM_SIZE[1]]


def add_colorbar(label: str, fig: plt.Figure, ax: plt.Axes, im: AxesImage, height: float = 0.02) -> None:  # type: ignore
    cbax = ax.inset_axes([0, -height, 1.0, height], transform=ax.transAxes)  # type: ignore
    cbar = fig.colorbar(im, cax=cbax, orientation="horizontal", format="%1g")
    cbar.set_label(label, size=8)
    cbar.ax.tick_params(rotation=0, labelsize=4)


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
        with contextlib.suppress(Exception):
            if axis == "y":
                ax.axhline(location, xmin=0, xmax=reach, color=colour, lw=0.5)
                ax.axhline(location, xmin=(1 - reach), xmax=1, color=colour, lw=0.5)
            elif axis == "x":
                ax.axvline(location, ymin=0, ymax=reach, color=colour, lw=0.5)
                ax.axvline(location, ymin=(1 - reach), ymax=1, color=colour, lw=0.5)


def add_midlines(ax: plt.Axes) -> None:  # type: ignore
    with contextlib.suppress(Exception):
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


def get_vrange(data: np.ndarray) -> tuple[float, float]:
    if np.std(data) < 1e-6:
        return float(np.min(np.abs(data))), float(np.max(np.abs(data)))
    vmin, vmax = np.nanpercentile(data, [1, 99])
    if np.sign(vmin) != np.sign(vmax):
        # If the min and max are of different signs, we need to adjust the range so zero is white
        vmin = min(vmin, -np.abs(vmax))
        vmax = max(vmax, np.abs(vmin))
    return float(vmin), float(vmax)


@pipeline_task()
def plot_detailed_images(primary: FileStoreEntry, output_path: Path, start: str | None = None) -> None:  # noqa: C901
    # TODO: full image run id, type, channel on image itself
    # TODO: add a dotted fun box around the biassec if it exists
    """Plot the images in the data store."""
    logger = get_logger()
    if not settings.plot:
        logger.info("Plotting is disabled. Skipping plot generation.")
        return

    prior_images = None

    flow_run_id = get_run_id()
    if start is None:
        store = _IMAGE_STORE[flow_run_id]
    else:
        assert start in _IMAGE_STORE[flow_run_id], f"Start task '{start}' not found in _IMAGE_STORE."
        store = {}
        found = False
        for key, images in _IMAGE_STORE[flow_run_id].items():
            if key == start:
                found = True
            if found:
                store[key] = images
            else:
                prior_images = images

    title_prefix = determine_figure_prefix(primary)

    for i, (key, images) in enumerate(store.items()):
        title = f"{title_prefix} - {key}"
        num_cols = len(images) * 4
        aspect_ratio = images[0].data.shape[1] / images[0].data.shape[0]

        fig, axes = plt.subplots(
            5,
            num_cols,
            figsize=(num_cols * 1.5 + 7, 16),
            gridspec_kw={"hspace": 0.2, "wspace": 0.2},
            height_ratios=[1.5 * aspect_ratio, 1, 1, 1, 1],
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

            vmin, vmax = np.nanpercentile(data, [1, 99])
            imd = axd.imshow(data.T, cmap=CMAP_DATA, aspect="equal", vmin=vmin, vmax=vmax, **im_kw)
            add_colorbar(f"Data {k}", fig, axd, imd)
            vmin, vmax = np.nanpercentile(variance, [1, 99])
            imv = axv.imshow(variance.T, cmap=CMAP_DATA, aspect="equal", vmin=vmin, vmax=vmax, **im_kw)
            add_colorbar(f"Variance {k}", fig, axv, imv)

            imdz = axdz.imshow(
                extract_zoom(data).T,
                cmap=CMAP_ZOOM,
                aspect="auto",
                **im_kw,
            )
            add_colorbar(f"Zoomed Data {k}", fig, axdz, imdz, height=0.04)
            imvz = axvz.imshow(
                extract_zoom(variance).T,
                cmap=CMAP_ZOOM,
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
            vdmin, vdmax = get_vrange(data_diff)
            vvmin, vvmax = get_vrange(variance_diff)
            imdd = axdd.imshow(data_diff.T, cmap=CMAP_DIFF, aspect="equal", vmin=vdmin, vmax=vdmax, **im_kw)
            add_colorbar("ΔData", fig, axdd, imdd)
            imvd = axvd.imshow(variance_diff.T, cmap=CMAP_DIFF, aspect="equal", vmin=vvmin, vmax=vvmax, **im_kw)
            add_colorbar("ΔVar", fig, axvd, imvd)

            zoomed_data_diff = extract_zoom(data_diff)
            if zoomed_data_diff.size:
                vmin, vmax = np.nanpercentile(zoomed_data_diff, [1, 99])
                imddz = axdzd.imshow(zoomed_data_diff.T, cmap=CMAP_DIFF, aspect="auto", vmin=vmin, vmax=vmax, **im_kw)
                add_colorbar("Zoomed ΔData", fig, axdzd, imddz, height=0.04)

            zoomed_variance_diff = extract_zoom(variance_diff)
            if zoomed_variance_diff.size:
                vmin, vmax = np.nanpercentile(zoomed_variance_diff, [1, 99])
                imvdz = axvzd.imshow(
                    zoomed_variance_diff.T, cmap=CMAP_DIFF, aspect="auto", vmin=vmin, vmax=vmax, **im_kw
                )
                add_colorbar("Zoomed ΔVar", fig, axvzd, imvdz, height=0.04)

            for ax in (axd, axv, axdd, axvd):
                add_callout_rectangle(ax)
                if "BIASSEC" in image.header:
                    # TODO: Make this class constructor
                    bias_section = image.get_bias_section()[0]
                    add_section_rectangle(ax, bias_section, edgecolor="#38bdf8", linestyle=":")

            # Now we add some line plots for better readability
            kwargs = {"lw": 0.3, "alpha": 0.7}
            y_lim_percentages = [1, 99]
            for ax, sec in [(axxdl, data), (axxvl, variance), (axxddl, data_diff), (axxvdl, variance_diff)]:
                for location, colour in LINES_X.items():
                    ax.plot(sec[location, :], color=colour, **kwargs)

                dx = sec[list(LINES_X.keys()), :]
                ymin, ymax = np.nanpercentile(dx, y_lim_percentages)
                if abs(ymin) < 1e-4:
                    ymin -= 0.02 * (ymax - ymin)
                if abs(ymax) < 1e-4:
                    ymax += 0.02 * (ymax - ymin)
                ax.set_ylim(ymin, ymax)

            for ax, sec in [(axydl, data), (axyvl, variance), (axyddl, data_diff), (axyvdl, variance_diff)]:
                for location, colour in LINES_Y.items():
                    ax.plot(sec[:, location], color=colour, **kwargs)
                dy = sec[:, list(LINES_Y.keys())]
                ymin, ymax = np.nanpercentile(dy, y_lim_percentages)
                if abs(ymin) < 1e-4:
                    ymin -= 0.02 * (ymax - ymin)
                if abs(ymax) < 1e-4:
                    ymax += 0.02 * (ymax - ymin)
                ax.set_ylim(ymin, ymax)

            # We also want to add the midlines to the midline axes
            for ax, sec in [(axxc, data), (axyc, variance), (axxcd, data_diff), (axycd, variance_diff)]:
                with contextlib.suppress(Exception):
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
            axydl.set_xlabel("Data rows", fontsize=8)
            axyddl.set_xlabel("ΔData rows", fontsize=8)
            axyvl.set_xlabel("Variance rows", fontsize=8)
            axxvl.set_xlabel("Variance columns", fontsize=8)
            axxdl.set_xlabel("Data columns", fontsize=8)
            axxddl.set_xlabel("ΔData columns", fontsize=8)
            axyvdl.set_xlabel("ΔVariance rows", fontsize=8)
            axxvdl.set_xlabel("ΔVariance columns", fontsize=8)
        output_location = output_path / f"preprocessing_detailed_{i}_{key}.webp"
        logger.info(f"Saving plot to {output_location}")
        fig.savefig(output_location, dpi=600, bbox_inches="tight")
        plt.close(fig)

        prior_images = images


@pipeline_task()
def plot_bias_sections(primary_file: FileStoreEntry, output_folder: Path) -> None:
    """Plot the bias section of the images."""
    logger = get_logger()
    if not settings.plot:
        logger.info("Plotting is disabled. Skipping bias section plot generation.")
        return

    flow_run_id = get_run_id()
    prior_images = None
    for i, (key, images) in enumerate(_BIAS_STORE[flow_run_id].items()):
        fig, axes = plt.subplots(
            2,
            len(images),
            figsize=(len(images) * 3 + 3, 12),
            gridspec_kw={"hspace": 0.2, "wspace": 0.2},
        )
        title_prefix = determine_figure_prefix(primary_file)
        title = f"BIASSEC - {title_prefix} - {key}"
        axes[0, 0].annotate(title, xy=(0, 1.01), xycoords="axes fraction", ha="left", va="bottom", fontsize=10)

        for k, image in enumerate(images):
            ax = axes[0, k]
            axd = axes[1, k]

            _, bias_data, _ = image.get_bias_section()
            if (
                prior_images is not None
                and len(prior_images) == len(images)
                and prior_images[k].data.shape == image.data.shape
            ):
                _, prior_image_data, _ = prior_images[k].get_bias_section()
                bias_diff = bias_data - prior_image_data
            else:
                bias_diff = np.zeros_like(bias_data)

            cmin, cmax = np.percentile(bias_data, [1, 99])
            im = ax.imshow(
                bias_data.T,
                origin="lower",
                interpolation="none",
                cmap=CMAP_DATA,
                aspect="auto",
                vmin=cmin,
                vmax=cmax,
            )
            add_colorbar(f"Bias Section (mean={np.nanmean(bias_data):.2f}, std{np.nanstd(bias_data):.2f})", fig, ax, im)

            cmin, cmax = np.percentile(bias_diff, [1, 99])
            im = axd.imshow(
                bias_diff.T,
                origin="lower",
                interpolation="none",
                cmap=CMAP_DIFF,
                aspect="auto",
                vmin=np.nanmin(bias_diff),
                vmax=np.nanmax(bias_diff),
            )
            add_colorbar(
                f"Bias Section Diff (mean={np.nanmean(bias_diff):.2f}, std{np.nanstd(bias_diff):.2f})",
                fig,
                axd,
                im,
            )

            ax.set_xticks([])
            axd.set_xticks([])
            ax.tick_params(axis="y", labelsize=6)
            axd.tick_params(axis="y", labelsize=6)

        prior_images = images

        output_location = output_folder / f"bias_{i}_{key}.png"
        logger.info(f"Saving bias section plot to {output_location}")
        fig.savefig(output_location, dpi=900, bbox_inches="tight")
