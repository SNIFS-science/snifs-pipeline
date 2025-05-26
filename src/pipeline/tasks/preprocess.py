from pathlib import Path

import numpy as np

from pipeline.common.prefect_utils import pipeline_task
from pipeline.config.global_settings import settings
from pipeline.resolver.resolver import Resolver
from pipeline.tasks.common import Image, flag_skip
from pipeline.tasks.preprocessing.bichips import build_bichip_from_fits
from pipeline.tasks.preprocessing.plots import plot, plot_images


@flag_skip("POISNOIS")
@plot()
@pipeline_task()
def add_variance(image: Image) -> Image:
    image = image.copy()
    # Poisson noise variance is equal to number of electron samples.
    image.variance += np.clip(image.data, 0, np.inf)  # type: ignore
    return image


@flag_skip("BIASDONE")
@plot()
@pipeline_task()
def subtract_bias(image: Image, bias_image: Image) -> Image:
    assert bias_image.header.get_bool("BIASFRAM"), "Bias image must have BIASFRAM set to True"
    image = image.copy()

    if bias_image.data.shape != image.data.shape:
        raise ValueError(f"Bias image shape {bias_image.data.shape} does not match image shape {image.data.shape}")

    image.data -= bias_image.data
    image.variance += bias_image.variance

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

    chip.image = add_variance(chip.image)

    # TODO: apparently custom flats can be an option and its specifically for R channel hot lines?
    plot_images(settings.output_path / path.stem)
