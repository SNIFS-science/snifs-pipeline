from pathlib import Path

import numpy as np

from pipeline.common.prefect_utils import pipeline_task
from pipeline.config.global_settings import settings
from pipeline.tasks.common import Image, flag_skip
from pipeline.tasks.preprocessing.bichips import build_bichip_from_fits
from pipeline.tasks.preprocessing.models import DarkModel, subtract_bias, subtract_dark
from pipeline.tasks.preprocessing.plots import plot, plot_images


@flag_skip("POISNOIS")
@plot()
@pipeline_task()
def add_variance(image: Image) -> Image:
    image = image.copy()
    # Poisson noise variance is equal to number of electron samples.
    image.variance += np.clip(image.data, 0, np.inf)  # type: ignore
    return image


def preprocess_exposure(
    path: Path,
    bias_image_file: Path,
    bias_model_file: Path,
    prefer_bias_image_over_model: bool,
    dark_image_file: Path,
    dark_model_file: Path,
    prefer_dark_image_over_model: bool,
    binary_offset_model_file: Path,
):
    # Both R and B channels have one CCD read by two amplifiers.
    # The 'chip' terminology means the amps, not that there are two CCDs
    bichip = build_bichip_from_fits(path, binary_offset_model_file)
    chip = bichip.assemble()  # noqa: F841

    chip.image = add_variance(chip.image)

    if prefer_bias_image_over_model:
        bias_reference = Image.from_fits_file(bias_image_file, transpose=True)
    else:
        bias_reference = DarkModel.model_validate_json(bias_model_file.read_text())
    chip.image = subtract_bias(chip.image, bias_reference, chip.primary_headers)

    if prefer_dark_image_over_model:
        dark_reference = Image.from_fits_file(dark_image_file, transpose=True)
    else:
        dark_reference = DarkModel.model_validate_json(dark_model_file.read_text())
    chip.image = subtract_dark(chip.image, dark_reference, chip.primary_headers)

    plot_images(settings.output_path / path.stem)
