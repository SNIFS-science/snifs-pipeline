from pathlib import Path

import numpy as np

from pipeline.common.prefect_utils import pipeline_task
from pipeline.resolver.resolver import Resolver
from pipeline.tasks.common import Headers, load_all_data_extensions, load_headers


@pipeline_task()
def assemble_exposure(path: Path, headers: Headers) -> np.ndarray:
    """Ensures we have a 2048x4096 exposure image from the raw file.


    The first thing we need to do is ensure that we load in the data in the same format.
    Both the B and R channels have one CCD, read from two amplifiers. R is read by
    otcom, which packages the two amplifiers into a single file and data array (2048x4096).
    The B channel is read by detcom, which puts the two amplifiers into different extensions
    in the FITS file (2x 1024x4096). Only the P channel has two CCDs, which we don't worry about.
    """

    # TODO: part of the assemble involves overriding header values for the GAIN. See HackFitsKeywords and kGainBlue etc
    data = load_all_data_extensions(path)
    return data


@pipeline_task()
def add_variance(exposure: np.ndarray, variance: np.ndarray) -> np.ndarray:
    # Congrats, Poisson noise variance is equal to number of electron samples.
    return variance + exposure


def preprocess_exposure(path: Path, resovler: Resolver):
    headers = load_headers(path)
    exposure = assemble_exposure(path, headers)  # noqa: F841

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
