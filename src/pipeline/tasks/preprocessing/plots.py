from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar

import cmasher as cmr
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage

from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_task
from pipeline.config.global_settings import settings
from pipeline.tasks.common import Image

_DATA_STORE: dict[str, list[Image]] = {}

P = ParamSpec("P")
R = TypeVar("R")
ZOOM_START = (512, 512)
ZOOM_SIZE = (100, 100)


def plotted_task(**kwargs):
    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        task_func = pipeline_task(**kwargs)(func)

        @wraps(func)
        def inner(images: list[Image], *args, **kwargs) -> list[Image]:
            if not _DATA_STORE:
                _DATA_STORE["initial"] = images
            result = task_func(images, *args, **kwargs)
            _DATA_STORE[func.__name__] = result
            return result

        return inner  # type: ignore

    return decorate


def extract_zoom(data: np.ndarray) -> np.ndarray:
    return data[ZOOM_START[0] : ZOOM_START[0] + ZOOM_SIZE[0], ZOOM_START[1] : ZOOM_START[1] + ZOOM_SIZE[1]]


def add_colorbar(label: str, fig: plt.Figure, ax: plt.Axes, im: AxesImage, height: float = 0.02) -> None:  # type: ignore
    cbax = ax.inset_axes([0, 0, 1.0, height], transform=ax.transAxes)  # type: ignore
    cbar = fig.colorbar(im, cax=cbax, orientation="horizontal", format="%1g")
    cbar.set_label(label, size=8)
    cbar.ax.tick_params(rotation=45, labelsize=6)


@pipeline_task()
def plot_images(output_path: Path) -> None:
    """Plot the images in the data store."""
    logger = get_logger()
    if not settings.plot:
        logger.info("Plotting is disabled. Skipping plot generation.")
        return
    output_path.mkdir(parents=True, exist_ok=True)

    cmap_data = cmr.torch
    cmap_zoom = cmr.rainforest
    cmap_diff = cmr.prinsenvlag
    # One column for data, one for variance

    all_data = np.concatenate(
        [images[0].data.flatten() for images in _DATA_STORE.values()]
        + [images[0].variance.flatten() for images in _DATA_STORE.values()]
    )
    all_data_zooms = np.concatenate(
        [extract_zoom(images[0].data).flatten() for images in _DATA_STORE.values()]
        + [extract_zoom(images[0].variance).flatten() for images in _DATA_STORE.values()]
    )

    min_c_data, max_c_data = np.percentile(all_data, [1, 99])
    min_c_data_zoom, max_c_data_zoom = np.percentile(all_data_zooms, [1, 99])

    prior_images = None
    for i, (key, images) in enumerate(_DATA_STORE.items()):
        num_cols = len(images) * 4

        fig, axes = plt.subplots(
            2,
            num_cols,
            figsize=(num_cols * 1.5 + 4, 10),
            gridspec_kw={"hspace": 0.2, "wspace": 0.1},
            height_ratios=[4, 1],
        )
        for k, image in enumerate(images):
            axd, axv = axes[0, 4 * k], axes[0, 4 * k + 2]  # data and variance axes
            axdz, axvz = axes[1, 4 * k], axes[1, 4 * k + 2]  # zoomed data and variance axes
            axdd, axvd = axes[0, 4 * k + 1], axes[0, 4 * k + 3]  # data and variance difference axes
            axdzd, axvzd = axes[1, 4 * k + 1], axes[1, 4 * k + 3]  # zoomed data and variance difference axes

            data, variance = image.data.astype(np.float64), image.variance.astype(np.float64)
            data[~np.isfinite(data)] = np.nan
            variance[~np.isfinite(variance)] = np.nan

            imd = axd.imshow(data.T, cmap=cmap_data, aspect="equal", origin="lower", vmin=min_c_data, vmax=max_c_data)
            add_colorbar(f"Data {k}", fig, axd, imd)
            imv = axv.imshow(
                variance.T, cmap=cmap_data, aspect="equal", origin="lower", vmin=min_c_data, vmax=max_c_data
            )
            add_colorbar(f"Variance {k}", fig, axv, imv)

            # Add callout rectangles
            dr = patches.Rectangle(
                ZOOM_START, ZOOM_SIZE[0], ZOOM_SIZE[1], linewidth=0.5, edgecolor="r", facecolor="none"
            )
            axd.add_patch(dr)
            vr = patches.Rectangle(
                ZOOM_START, ZOOM_SIZE[0], ZOOM_SIZE[1], linewidth=0.5, edgecolor="r", facecolor="none"
            )
            axv.add_patch(vr)

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
                data_diff = data - prior_data
                variance_diff = variance - prior_variance
            else:
                data_diff = np.zeros_like(data)
                variance_diff = np.zeros_like(variance)

            imdd = axdd.imshow(
                data_diff.T,
                cmap=cmap_diff,
                aspect="equal",
                origin="lower",
            )
            add_colorbar("ΔData", fig, axdd, imdd)
            imvd = axvd.imshow(
                variance_diff.T,
                cmap=cmap_diff,
                aspect="equal",
                origin="lower",
            )
            add_colorbar("ΔVar", fig, axvd, imvd)
            zoomed_data_diff = extract_zoom(data_diff)
            imddz = axdzd.imshow(
                zoomed_data_diff.T,
                cmap=cmap_diff,
                aspect="auto",
                origin="lower",
            )
            add_colorbar("Zoomed ΔData", fig, axdzd, imddz, height=0.04)
            zoomed_variance_diff = extract_zoom(variance_diff)
            imvdz = axvzd.imshow(
                zoomed_variance_diff.T,
                cmap=cmap_diff,
                aspect="auto",
                origin="lower",
            )
            add_colorbar("Zoomed ΔVar", fig, axvzd, imvdz, height=0.04)

            drd = patches.Rectangle(
                ZOOM_START, ZOOM_SIZE[0], ZOOM_SIZE[1], linewidth=0.5, edgecolor="r", facecolor="none"
            )
            axdd.add_patch(drd)
            vrd = patches.Rectangle(
                ZOOM_START, ZOOM_SIZE[0], ZOOM_SIZE[1], linewidth=0.5, edgecolor="r", facecolor="none"
            )
            axvd.add_patch(vrd)

            if k == 0:
                axd.set_title(key, size=8)

        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        output_location = output_path / f"{i}_{key}.png"
        logger.info(f"Saving plot to {output_location}")
        fig.savefig(output_location, dpi=300, bbox_inches="tight")
        plt.close(fig)

        prior_images = images
