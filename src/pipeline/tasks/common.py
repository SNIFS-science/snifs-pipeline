from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.io.fits import Header

from pipeline.common.log import get_logger

type Headers = dict[str, str | bool | int | float]


def load_headers(science_file: Path, hdu_index: int = 0) -> Headers:
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


def load_image_data(science_file: Path, hdu_index: int = 0) -> np.ndarray:
    logger = get_logger()
    with fits.open(science_file) as hdul:  # type: ignore
        assert len(hdul) > hdu_index, f"FITS file {science_file} does not have HDU {hdu_index}"
        data = hdul[hdu_index].data
        logger.debug(f"Loaded image data from {science_file} with shape {data.shape} and dtype {data.dtype}")
        return data


def load_all_data_extensions(science_file: Path) -> list[np.ndarray]:
    """
    Load all data extensions from a FITS file.
    """
    logger = get_logger()
    with fits.open(science_file) as hdul:  # type: ignore
        data = [hdu.data for hdu in hdul if isinstance(hdu.data, np.ndarray)]
        logger.debug(f"Loaded {len(data)} data extensions from {science_file}")
        return data
