import json
from pathlib import Path

import numpy as np


def get_correction_amplitude(
    driver_2: np.ndarray,
    driver_3: np.ndarray,
    driver_other_3: np.ndarray,
    parameters: list[float],
) -> None:
    pass  # TODO: This is a placeholder for the actual implementation


# # https://arxiv.org/pdf/1802.06914
def correct_binary_offset(data_list: list[np.ndarray], bom_path: Path) -> None:
    assert bom_path.exists(), f"Binary offset model file {bom_path} does not exist"
    model = json.loads(bom_path.read_text())
    parameters = model["parameters"]
    assert len(data_list) == 2, "Binary offset model only supports two chips"
    assert len(parameters) == 2, "Binary offset model should have two lists of parameters"

    # The original code (imagesnifs.cxx CorrectBinaryOffset) goes pixel by pixel
    # reading right to left (in x) to do the correction in place. However, each
    # driver reads to the left, so we can vectorise this without worry. ie the correction
    # is not using prior corrections. Please forgive me for these bad
    # variable names. I don't know what they mean, and the original code is not
    # very clear either.

    driver_2 = data_list[0][:, :-2]
    driver_3 = data_list[0][:, 1:-3]
    driver_other_2 = data_list[1][:, :-2]
    driver_other_3 = data_list[1][:, 1:-3]
    correction = get_correction_amplitude(driver_2, driver_3, driver_other_3, parameters[0])
    correction_other = get_correction_amplitude(driver_other_2, driver_other_3, driver_3, parameters[1])

    corrected = data_list[0] - correction  # noqa: F841
    corrected_other = data_list[1] - correction_other  # noqa: F841

    # The BOM doesnt work for the first three columns of data so it seems the original algorithm
    # determines a mean correction for the rest of the row and applies that mean to first three columns
    # TODO
