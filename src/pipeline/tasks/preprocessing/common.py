import numpy as np

from pipeline.common.prefect_utils import pipeline_task
from pipeline.tasks.common import Image, flag_skip
from pipeline.tasks.preprocessing.plots import plot


@flag_skip("POISNOIS")
@plot()
@pipeline_task()
def add_poisson_noise_to_variance(image: Image) -> Image:
    image = image.copy()
    # Poisson noise variance is equal to number of electron samples.
    image.variance += np.clip(image.data, 0, np.inf)  # type: ignore
    return image
