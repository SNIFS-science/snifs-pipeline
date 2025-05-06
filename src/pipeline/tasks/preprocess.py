from pathlib import Path

import numpy as np
from pydantic import BaseModel

from pipeline.common.prefect_utils import pipeline_task
from pipeline.resolver.resolver import Resolver
from pipeline.tasks.common import Headers, load_all_data_extensions, load_headers, load_headers_from_data

SATURATE = 65535  # TODO add this in. find better way of doing fits header bullshit.


class Chip(BaseModel):
    primary_headers: Headers
    data: np.ndarray
    variance: np.ndarray


class BiChip(BaseModel):
    primary_headers: Headers
    data1: np.ndarray
    data2: np.ndarray
    variance1: np.ndarray
    variance2: np.ndarray
    headers1: Headers
    headers2: Headers

    @property
    def data(self) -> np.ndarray:
        return np.hstack((self.data1, self.data2))

    @property
    def variance(self) -> np.ndarray:
        return np.hstack((self.variance1, self.variance2))

    def assemble(self) -> Chip:
        """Ensures we have a 2048x4096 exposure image from the raw file.

        The first thing we need to do is ensure that we load in the data in the same format.
        Both the B and R channels have one CCD, read from two amplifiers. R is read by
        otcom, which packages the two amplifiers into a single file and data array (2048x4096).
        The B channel is read by detcom, which puts the two amplifiers into different extensions
        in the FITS file (2x 1024x4096). Only the P channel has two CCDs, which we don't worry about.

        Note though that the arrays wont be the exact same shape as the comment above, because there
        are extra pixels because there are extra pixels in the readout in the overscan region.
        """
        return Chip(
            primary_headers=self.primary_headers,
            data=self.data,
            variance=self.variance,
        )


def build_bichip_from_fits(path: Path) -> BiChip:
    """Load a BiChip from a FITS file."""
    data_list = load_all_data_extensions(path)
    data_headers = load_headers_from_data(path)  # noqa: F841
    primary_headers = load_headers(path)  # noqa: F841
    if len(data_list) == 1:
        pass
    # TODO: preprocessor.cxx 259
    # TODO: BuildRawBiChip logic
    # TODO: HackFitsKeywords
    # TODO: create variance
    # TODO: handle saturation
    # TODO: binary offset invocation
    # TODO: overscan subtraction
    return BiChip(...)  # type: ignore


def handle_saturation(data: np.ndarray, level: float, variance: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Handle saturation in the data

    For more details on this, see Emmanuel Gangler's thesis, section 3.3.2, PDF page 56.
    You can find the original French and a Google-Translated version in the
    docs/pdfs folder in this repository.

    The process is to look for readings above the saturation level, and then
    set their variance to infinity. Because saturation has a bleed, we also set
    pixels in the touching rows (not columns though) to infinity as well.
    """
    saturation_mask = data > level
    number_saturated_pixels = int(np.sum(saturation_mask))
    saturation_mask[:, :-1] |= saturation_mask[:, 1:]
    saturation_mask[:, 1:] |= saturation_mask[:, :-1]
    variance[saturation_mask] = np.inf
    return number_saturated_pixels, data, variance


@pipeline_task()
def add_variance(exposure: np.ndarray, variance: np.ndarray) -> np.ndarray:
    # Congrats, Poisson noise variance is equal to number of electron samples.
    return variance + exposure


def preprocess_exposure(path: Path, resovler: Resolver):
    bichip = build_bichip_from_fits(path)
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
