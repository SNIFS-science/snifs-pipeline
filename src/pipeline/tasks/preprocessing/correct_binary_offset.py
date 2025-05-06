import numpy as np

from pipeline.common.prefect_utils import pipeline_task


# # https://arxiv.org/pdf/1802.06914
@pipeline_task()
def correct_binary_offset() -> np.ndarray:
    # Congrats, Poisson noise variance is equal to number of electron samples.
    return None  # type: ignore
