import json
from pathlib import Path

import numpy as np

from pipeline.tasks.common import Image
from pipeline.tasks.preprocessing.plots import plotted_task


def count_bits(x: np.ndarray) -> np.ndarray:
    """Count the number of bits set in a binary number.

    For example...
    9 is 1001 in binary, which means the 0th and 3rd bits are set.
    Thus the count is 2.

    11 is 1011 in binary, which means the 0th, 1st and 3rd bits are set.
    Thus the count is 3.
    """
    assert np.issubdtype(x.dtype, np.unsignedinteger), "count_bits only works on unsigned int arrays"
    return np.bitwise_count(x)


def sum_binary_index(x: np.ndarray) -> np.ndarray:
    """We also need to calculate the set binary index sum.

    For example...
    9 is 1001 in binary, which means the 0th and 3rd bits are set.
    Thus the sum is 0 + 3 = 3.

    10 is 1010 in binary, which means the 1st and 3rd bits are set.
    Thus the sum is 1 + 3 = 4.

    259 is 100000011 in binary, which means the 0th, 1st and 8th bits are set.
    Thus the sum is 0 + 1 + 8 = 9.

    Function assumes that readings are 64 bit or less. In reality, SNIFS
    readings are 16 bit, but we use 64 bit because why make tighter assumptions
    when we don't have to?

    Talking about assumptions, we're assuming big endian, just like the original code.
    """
    assert len(x.shape) == 2, "sum_binary_index only works on 2D arrays"
    assert x.dtype == np.uint16, "sum_binary_index only works on uint16 arrays"
    return (
        (
            np.unpackbits(x[:, :, None].view(np.uint8)[:, :, ::-1], axis=2, count=16)
            * np.arange(16, dtype=np.uint8)[::-1].reshape(1, 1, 16)
        )
        .sum(axis=2)
        .astype(np.uint16)
    )


def get_correction_amplitude(
    driver_2: np.ndarray,
    driver_3: np.ndarray,
    driver_other_3: np.ndarray,
    params: list[float],
) -> np.ndarray:
    # Because the correction is dependent on how many bits are set in the readout
    # we want to count the number of bits in the *3 drivers.
    bits_driver_3 = count_bits(driver_3)
    bits_driver_other_3 = count_bits(driver_other_3)

    # Behold this abomination of linear combinations.
    correction = (
        params[0] * count_bits(driver_2 & driver_3)
        + params[1] * count_bits(driver_2 & (~driver_3))
        + params[2] * sum_binary_index(driver_2 & driver_3)
        + params[3] * sum_binary_index(driver_2 & (~driver_3))
        + params[4] * bits_driver_3
        + params[5] * bits_driver_other_3
        + params[6] * bits_driver_3 * bits_driver_3
        + params[7] * bits_driver_other_3 * bits_driver_other_3
        + params[8] * bits_driver_3 * bits_driver_other_3
    )
    return correction


# # https://arxiv.org/pdf/1802.06914
@plotted_task()
def correct_binary_offset(images: list[Image], bom_path: Path) -> list[Image]:
    """Corrects the binary offset of the data using the binary offset model.

    Note that the input data_list should be a list of two 2D numpy arrays, each
    representing a chip and have data type uint16. The function will return
    a list of two 2D numpy arrays, each representing the corrected chip data, but
    the data type will now be changed to float64 as the correction is not integer.
    """
    assert bom_path.exists(), f"Binary offset model file {bom_path} does not exist"
    model = json.loads(bom_path.read_text())
    parameters = model["parameters"]
    assert len(images) == 2, f"Binary offset model only supports two chips. You've passed in {len(images)}"
    assert len(parameters) == 2, f"Binary offset model should have two lists of parameters. You have {len(parameters)}"
    for p in parameters:
        assert len(p) == 9, f"Binary offset model should have nine parameters per chip. Yours has {len(p)}"

    # The original code (imagesnifs.cxx CorrectBinaryOffset) goes pixel by pixel
    # reading right to left (in x) to do the correction in place. However, each
    # driver reads to the left, so we can vectorise this without worry. ie the correction
    # is not using prior corrections. Please forgive me for these bad
    # variable names. I don't know what they mean, and the original code is not
    # very clear either.

    driver_2 = images[0].data[:, 1:-2]
    driver_3 = images[0].data[:, :-3]
    driver_other_2 = images[1].data[:, 1:-2]
    driver_other_3 = images[1].data[:, :-3]
    correction = get_correction_amplitude(driver_2, driver_3, driver_other_3, parameters[0])
    correction_other = get_correction_amplitude(driver_other_2, driver_other_3, driver_3, parameters[1])

    # The BOM doesnt work for the first three columns of data so it seems the original algorithm
    # determines a mean correction for the rest of the row and applies that mean to first three columns
    row_mean = np.mean(correction, axis=1)[:, None]
    row_mean_other = np.mean(correction_other, axis=1)[:, None]

    # Before we set anything, copy the original data images and set their data type to float64
    # This is because the correction is not integer, and we don't want to lose precision
    images = [image.copy(type_coercion=np.float64) for image in images]

    # Initial correction works for all but the first three columns
    images[0].data[:, 3:] -= correction
    images[1].data[:, 3:] -= correction_other
    # The first three columns are corrected with the mean
    images[0].data[:, :3] -= row_mean
    images[1].data[:, :3] -= row_mean_other

    return images


if __name__ == "__main__":
    # Test the function with some dummy data
    x = np.array([[259]], dtype=np.uint16)
    print(sum_binary_index(x))  # noqa: T201
