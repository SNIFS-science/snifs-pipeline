from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.io.fits import Header

from pipeline.common import Headers, Image, get_logger, pipeline_task


@pipeline_task()
def load_images_from_file(science_file: Path, transpose: bool = False) -> list[Image]:
    """
    Load all data extensions and their headers from a FITS file.
    """
    logger = get_logger()
    with fits.open(science_file) as hdul:  # type: ignore
        data = [
            Image(
                data=hdu.data,  # type: ignore
                header=Headers.from_astropy_header(hdu.header),  # type: ignore
                variance=np.zeros_like(hdu.data, dtype=np.float64),  # type: ignore
            )
            for hdu in hdul
            if isinstance(hdu.data, np.ndarray)  # type: ignore
        ]
        logger.debug(f"Loaded {len(data)} data extensions with headers from {science_file}")
        if transpose:
            for d in data:
                d.data = d.data.T
                d.variance = d.variance.T
    return data


@pipeline_task()
def load_headers(science_file: Path, hdu_index: int = 0) -> Headers:
    """
    Load the primary header of a FITS file.
    """
    logger = get_logger()
    with fits.open(science_file) as hdul:  # type: ignore
        assert len(hdul) > hdu_index, f"FITS file {science_file} does not have HDU {hdu_index}"
        header: Header = hdul[hdu_index].header  # type: ignore
        result = Headers.from_astropy_header(header)
        logger.debug(f"Loaded header from {science_file} with {len(result)} keys")
        return result
