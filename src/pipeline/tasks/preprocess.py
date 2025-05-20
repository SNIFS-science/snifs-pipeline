from pathlib import Path

import numpy as np

from pipeline.common.prefect_utils import pipeline_task
from pipeline.config.global_settings import settings
from pipeline.resolver.resolver import Resolver
from pipeline.tasks.preprocessing.bichips import build_bichip_from_fits
from pipeline.tasks.preprocessing.plots import plot_images


@pipeline_task()
def add_variance(exposure: np.ndarray, variance: np.ndarray) -> np.ndarray:
    # Congrats, Poisson noise variance is equal to number of electron samples.
    return variance + exposure


def preprocess_exposure(path: Path, resolver: Resolver):
    # Both R and B channels have one CCD read by two amplifiers.
    # The 'chip' terminology means the amps, not that there are two CCDs
    bichip = build_bichip_from_fits(path, resolver)
    chip = bichip.assemble()  # noqa: F841

    # TODO: binary offset model comes from somewhere and does something
    # TODO: We never provide a bias file so this subtraction is useless

    # Check to see if the noise has already been added
    # This is normally done by checking that POISNOIS, if it exists in the header, is not 1
    # TODO: In general I dislike all these magic header values and ideally would do something a bit more transparent
    # Anyway, if the noise isnt there, then we add poisson noise
    # TODO: Dont like magic strings, will pull this into a subconfig.
    # if headers.get("POISNOIS") != 1:
    #     variance = add_variance(path, variance)

    # TODO: figure out intermediate file situation. Ideally I dont want tons and tons of files with process
    # TODO: information locked away in random file headers.
    # if bias model: subtract bais model (what is the bias model passed in)
    # if there's a dark file: subtract it (I dont think we have darks)
    # if there's a dark map: subtract it (I dont think we have dark maps)
    # if there's a dark model: subtract it (I dont think we have dark models)

    # exposure = handle_cosmetic(exposure, header)

    # if we have a flat file: apply the flat
    # TODO: apparently custom flats can be an option and its specifically for R channel hot lines?
    plot_images(settings.output_path / path.stem)
