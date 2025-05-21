from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_task
from pipeline.config.global_settings import settings
from pipeline.tasks.common import Image

_DATA_STORE: dict[str, list[Image]] = {}

P = ParamSpec("P")
R = TypeVar("R")
ZOOM_START = (512, 512)
ZOOM_SIZE = (128, 128)


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


@pipeline_task()
def plot_images(output_path: Path) -> None:
    """Plot the images in the data store."""
    logger = get_logger()
    if not settings.plot:
        logger.info("Plotting is disabled. Skipping plot generation.")
        return
    output_path.mkdir(parents=True, exist_ok=True)

    # One column for data, one for variance
    num_cols = max(len(images) for images in _DATA_STORE.values())

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

    for i, (key, images) in enumerate(_DATA_STORE.items()):
        fig, axes = plt.subplots(
            2,
            num_cols * 2,
            figsize=(num_cols * 3 + 4, 11),
            gridspec_kw={"hspace": 0.1, "wspace": 0.1},
            height_ratios=[4, 1],
        )
        for k, image in enumerate(images):
            axd, axv = axes[0, 2 * k], axes[0, 2 * k + 1]  # data and variance axes
            axdz, axvz = axes[1, 2 * k], axes[1, 2 * k + 1]  # zoomed data and variance axes

            data, variance = image.data.astype(np.float64), image.variance.astype(np.float64)
            data[~np.isfinite(data)] = np.nan
            variance[~np.isfinite(variance)] = np.nan
            axd.imshow(data.T, cmap="magma", aspect="auto", origin="lower", vmin=min_c_data, vmax=max_c_data)
            axv.imshow(variance.T, cmap="magma", aspect="auto", origin="lower", vmin=min_c_data, vmax=max_c_data)

            # Add callout rectangles
            dr = patches.Rectangle(
                ZOOM_START, ZOOM_SIZE[0], ZOOM_SIZE[1], linewidth=0.5, edgecolor="r", facecolor="none"
            )
            axd.add_patch(dr)
            vr = patches.Rectangle(
                ZOOM_START, ZOOM_SIZE[0], ZOOM_SIZE[1], linewidth=0.5, edgecolor="r", facecolor="none"
            )
            axv.add_patch(vr)

            axdz.imshow(
                extract_zoom(data).T,
                cmap="viridis",
                aspect="auto",
                origin="lower",
                vmin=min_c_data_zoom,
                vmax=max_c_data_zoom,
            )
            axvz.imshow(
                extract_zoom(variance).T,
                cmap="viridis",
                aspect="auto",
                origin="lower",
                vmin=min_c_data_zoom,
                vmax=max_c_data_zoom,
            )
            if k == 0:
                axd.set_title(key, size=8)

            axd.set_xlabel(f"Data {k}", size=8)
            axv.set_xlabel(f"Var {k}", size=8)
            axdz.set_xlabel(f"Zoomed Data {k}", size=8)
            axvz.set_xlabel(f"Zoomed Var {k}", size=8)

        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        output_location = output_path / f"{i}_{key}.png"
        logger.info(f"Saving plot to {output_location}")
        fig.savefig(output_location, dpi=300, bbox_inches="tight")
        plt.close(fig)
