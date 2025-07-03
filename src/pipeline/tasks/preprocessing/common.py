import numpy as np

from pipeline.common import Image, flag_skip, pipeline_task
from pipeline.tasks.plotting import plot
from pipeline.tasks.plotting.plots import plot_standalone


@flag_skip("POISNOIS")
@plot()
@pipeline_task()
@plot_standalone("add_poisson_noise_to_variance")
def add_poisson_noise_to_variance(image: Image) -> Image:
    image = image.copy()
    # Poisson noise variance is equal to number of electron samples.
    image.variance += np.clip(image.data, 0, np.inf)  # type: ignore
    image.add_function_lineage(
        f"Added Poisson noise to variance based on data, avg value is {np.mean(image.data):0.3f}."
    )
    return image


@pipeline_task()
def ensure_float64(images: list[Image]) -> list[Image]:
    result = [image.copy() for image in images]
    for image in result:
        if image.data.dtype != np.float64:
            image.data = image.data.astype(np.float64)
        if image.variance.dtype != np.float64:
            image.variance = image.variance.astype(np.float64)
    return result
