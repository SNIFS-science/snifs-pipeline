from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_task
from pipeline.config.global_settings import settings
from pipeline.tasks.common import Image

_DATA_STORE: dict[str, list[Image]] = {}


def plotted_task(**kwargs):
    def decorate(func: Callable) -> Callable:
        task_func = pipeline_task(**kwargs)(func)

        def inner(images: list[Image], *args, **kwargs) -> list[Image]:
            if not _DATA_STORE:
                _DATA_STORE["initial"] = images
            result = task_func(images, *args, **kwargs)
            _DATA_STORE[func.__name__] = result
            return result

        return inner

    return decorate


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

    all_data = np.concatenate([images[0].data.flatten() for images in _DATA_STORE.values()])
    all_var = np.concatenate([images[0].variance.flatten() for images in _DATA_STORE.values()])
    min_c_data, max_c_data = np.percentile(all_data, [1, 99])
    min_c_var, max_c_var = np.percentile(all_var, [1, 99])

    for i, (key, images) in enumerate(_DATA_STORE.items()):
        fig, axes = plt.subplots(
            1,
            num_cols * 2,
            figsize=(num_cols * 3 + 2, 12),
            sharex=True,
            sharey=True,
            gridspec_kw={"hspace": 0.1, "wspace": 0.1},
        )
        for k, image in enumerate(images):
            axd, axv = axes[2 * k], axes[2 * k + 1]  # data and variance axes
            axd.imshow(image.data.T, cmap="magma", aspect="equal", origin="lower", vmin=min_c_data, vmax=max_c_data)
            axv.imshow(image.variance.T, cmap="magma", aspect="equal", origin="lower", vmin=min_c_var, vmax=max_c_var)
            if k == 0:
                axd.set_title(key, size=8)
            axd.set_xticks([])
            axd.set_yticks([])
            axd.set_xlabel(f"Data {k}", size=8)
            axv.set_xlabel(f"Var {k}", size=8)

        output_location = output_path / f"{i}_{key}.png"
        logger.info(f"Saving plot to {output_location}")
        fig.savefig(output_location, dpi=300, bbox_inches="tight")
        plt.close(fig)
