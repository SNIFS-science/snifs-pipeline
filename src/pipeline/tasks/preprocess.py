from pathlib import Path

import numpy as np

from pipeline.common.prefect_utils import pipeline_task
from pipeline.config.global_settings import settings
from pipeline.resolver.common import FileType
from pipeline.resolver.resolver import Resolver
from pipeline.tasks.common import Image, flag_skip
from pipeline.tasks.preprocessing.bichips import build_bichip_from_fits
from pipeline.tasks.preprocessing.models import DarkModel, subtract_bias
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
    bias_path: Path | None,
    resolver: Resolver,
):
    # TODO: Figure out how to pass everything in.
    # Both R and B channels have one CCD read by two amplifiers.
    # The 'chip' terminology means the amps, not that there are two CCDs
    bichip = build_bichip_from_fits(path, resolver)
    chip = bichip.assemble()  # noqa: F841

    chip.image = add_variance(chip.image)

    # TODO: I see a bias and darks fits file, so check in with Greg as to which one should be the default
    if bias_path is not None:
        bias_reference = Image.from_fits_file(resolver.get_file_metadata(bias_path).file_path)
    else:
        bias_reference = DarkModel.model_validate_json(resolver.get_match_path(FileType.BIAS_MODEL, path).read_text())
    chip.image = subtract_bias(chip.image, bias_reference, chip.primary_headers)

    # TODO: In preprocessor the poisson noise is between subtract bias and subtract bias model
    # But I dont quite understand why its not done first? Im going to do it first here.

    # TODO: apparently custom flats can be an option and its specifically for R channel hot lines?
    plot_images(settings.output_path / path.stem)
