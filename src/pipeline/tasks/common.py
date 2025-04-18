from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.io.fits import Header

from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_task


@pipeline_task()
def load_header(science_file: Path, hdu_index: int = 0) -> dict[str, str | bool | int | float]:
    """
    Load the primary header of a FITS file.
    """
    logger = get_logger()
    with fits.open(science_file) as hdul:  # type: ignore
        assert len(hdul) > hdu_index, f"FITS file {science_file} does not have HDU {hdu_index}"
        header: Header = hdul[hdu_index].header
        result = {k: v for k, v in header.items() if v is not None}
        logger.debug(f"Loaded header from {science_file} with {len(result)} keys")
        return result


@pipeline_task()
def load_image_data(science_file: Path, hdu_index: int = 0) -> np.ndarray:
    logger = get_logger()
    with fits.open(science_file) as hdul:  # type: ignore
        assert len(hdul) > hdu_index, f"FITS file {science_file} does not have HDU {hdu_index}"
        data = hdul[hdu_index].data
        logger.debug(f"Loaded image data from {science_file} with shape {data.shape} and dtype {data.dtype}")
        return data
