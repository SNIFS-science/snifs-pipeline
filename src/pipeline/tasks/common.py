from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.io.fits import Header

from pipeline.common.log import get_logger


class Headers(dict[str, str | bool | int | float]):
    def get_optional_int(self, key: str, default: int | None = None) -> int | None:
        value = self.get(key)
        if isinstance(value, (int, float)):
            return int(value)
        elif value is None:
            return default
        raise ValueError(f"Key {key} is not an int: {value} has type {type(value)}")

    def get_int(self, key: str, default: int = None) -> int:  # type: ignore
        value = self.get_optional_int(key, default=default)
        assert value is not None, f"Key {key} is not not available in the header"
        return value

    def get_optional_float(self, key: str, default: float | None = None) -> float | None:
        value = self.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        elif value is None:
            return default
        raise ValueError(f"Key {key} is not a float: {value} has type {type(value)}")

    def get_float(self, key: str, default: float = None) -> float:  # type: ignore
        value = self.get_optional_float(key, default=default)
        assert value is not None, f"Key {key} is not not available in the header"
        return value

    def get_optional_bool(self, key: str, default: bool | None = None) -> bool | None:
        value = self.get(key)
        if isinstance(value, bool):
            return value
        elif value is None:
            return default
        raise ValueError(f"Key {key} is not a bool: {value} has type {type(value)}")

    def get_bool(self, key: str, default: str = None) -> bool:  # type: ignore
        value = self.get_optional_bool(key)
        assert value is not None, f"Key {key} is not not available in the header"
        return value

    def get_optional_str(self, key: str, default: str | None = None) -> str | None:
        value = self.get(key)
        if isinstance(value, str):
            return value
        elif value is None:
            return default
        raise ValueError(f"Key {key} is not a str: {value} has type {type(value)}")

    def get_str(self, key: str, default: str = None) -> str:  # type: ignore
        value = self.get_optional_str(key)
        assert value is not None, f"Key {key} is not not available in the header"
        return value

    def set_default(self, defaults: dict[str, str | bool | int | float]) -> None:
        """
        Set default values in the header.
        """
        logger = get_logger()
        for key, value in defaults.items():
            if key not in self:
                self[key] = value
                logger.debug(f"Header {key} was not set. Setting to default: {value}")

    def merge(self, other: "Headers") -> "Headers":
        """
        Merge another header into this one.
        """
        original = self.copy()
        for key, value in other.items():
            original[key] = value
        return Headers(**original)


def _stupid_header_to_dict(header: Header) -> Headers:
    return Headers(**{k: v for k, v in sorted(header.items()) if v is not None})


def load_headers(science_file: Path, hdu_index: int = 0) -> Headers:
    """
    Load the primary header of a FITS file.
    """
    logger = get_logger()
    with fits.open(science_file) as hdul:  # type: ignore
        assert len(hdul) > hdu_index, f"FITS file {science_file} does not have HDU {hdu_index}"
        header: Header = hdul[hdu_index].header
        result = _stupid_header_to_dict(header)
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


def load_headers_from_data(science_file: Path) -> list[Headers]:
    """
    Load all headers from a FITS file.
    """
    logger = get_logger()
    with fits.open(science_file) as hdul:  # type: ignore
        headers = [_stupid_header_to_dict(hdu.header) for hdu in hdul if isinstance(hdu.data, np.ndarray)]
        logger.debug(f"Loaded {len(headers)} headers from {science_file}")
        return headers
