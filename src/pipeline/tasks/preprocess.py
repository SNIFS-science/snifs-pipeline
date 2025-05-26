from pathlib import Path

import numpy as np

from pipeline.common.log import get_logger
from pipeline.config.global_settings import settings
from pipeline.resolver.resolver import Resolver
from pipeline.tasks.common import Image
from pipeline.tasks.preprocessing.bichips import build_bichip_from_fits
from pipeline.tasks.preprocessing.plots import plot_images, plotted_task


@plotted_task()
def add_variance(image: Image) -> Image:
    image = image.copy()
    header_key = "POISNOIS"
    if header_key in image.header:
        return image

    # Poisson noise variance is equal to number of electron samples.
    image.variance += np.clip(image.data, a_min=0)  # type: ignore
    image.header[header_key] = 1
    return image


@plotted_task()
def subtract_bias(image: Image, bias_image: Image) -> Image:
    assert bias_image.header.get_bool("BIASFRAM"), "Bias image must have BIASFRAM set to True"
    image = image.copy()
    logger = get_logger()
    header_key = "BIASDONE"
    if header_key in image.header:
        logger.warning("Bias subtraction already done, skipping.")
        return image

    if bias_image.data.shape != image.data.shape:
        raise ValueError(f"Bias image shape {bias_image.data.shape} does not match image shape {image.data.shape}")

    image.data -= bias_image.data
    image.variance += bias_image.variance
    image.header[header_key] = 1

    # TODO: I'm a bit confused about imagesnifs.cxx:368 - Are we supposed to be determining the dark frame?
    return image


def preprocess_exposure(path: Path, bias_path: Path | None, resolver: Resolver):
    # Both R and B channels have one CCD read by two amplifiers.
    # The 'chip' terminology means the amps, not that there are two CCDs
    bichip = build_bichip_from_fits(path, resolver)
    chip = bichip.assemble()  # noqa: F841

    if bias_path is not None:
        bias_image = Image.from_fits_file(resolver.get_file_metadata(bias_path).file_path)
        chip.image = subtract_bias(chip.image, bias_image)

    # TODO: Add a way to flag functions instead of having a bunch of my own checks
    if "POISNOIS" not in chip.image.header:
        chip.image = add_variance(chip.image)

    # TODO: apparently custom flats can be an option and its specifically for R channel hot lines?
    plot_images(settings.output_path / path.stem)
