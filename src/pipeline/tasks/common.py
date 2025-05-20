from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.io.fits import Header
from pydantic import BaseModel, ConfigDict

from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_task


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

    @classmethod
    def merge_all(cls, *headers: "Headers") -> "Headers":
        """
        Merge all headers into one.
        """
        result = Headers()
        for header in headers:
            for key, value in header.items():
                if key not in result:
                    result[key] = value
                else:
                    logger = get_logger()
                    logger.debug(f"Header {key} already exists. Overwriting with {value}")
        return result

    def merge(self, other: "Headers") -> "Headers":
        """
        Merge another header into this one.
        """
        original = self.copy()
        for key, value in other.items():
            original[key] = value
        return Headers(**original)

    def copy(self) -> "Headers":
        return Headers(**self)


# add a named tuple for the section
Section = namedtuple("Section", ["x_min", "x_max", "x_dir", "y_min", "y_max", "y_dir"])


def get_section_range(label: str) -> Section:
    """There is a header convention in fits files that defines a data range"""
    x_min, x_max, y_min, y_max = [int(i) for i in label[1:-1].replace(":", ",").split(",")]
    x_dir, y_dir = 1, 1
    if x_max < x_min:
        x_dir = -1
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_dir = -1
        y_min, y_max = y_max, y_min
    return Section(x_min - 1, x_max, x_dir, y_min - 1, y_max, y_dir)


def extract_section(pixels: np.ndarray, label: str) -> np.ndarray:
    """Extract a section from the pixels array based on the label."""
    section = get_section_range(label)
    return pixels[section.x_min : section.x_max : section.x_dir, section.y_min : section.y_max : section.y_dir]


class Image(BaseModel):
    """
    A class to hold the data and header of a FITS file.
    """

    data: np.ndarray
    header: Headers
    variance: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def copy(self, type_coercion: np.dtype | type[Any] | None = None) -> "Image":
        """
        Return a copy of the data and header.
        """
        image = Image(
            data=self.data.copy(),
            header=self.header.copy(),
            variance=self.variance.copy(),
        )
        if type_coercion is not None:
            image.data = image.data.astype(type_coercion)
            if image.variance is not None:
                image.variance = image.variance.astype(type_coercion)
        return image

    def get_data_section(self, enfore_datasec: bool = True) -> np.ndarray:
        data_section = self.header.get_optional_str("DATASEC")
        if data_section is None:
            if enfore_datasec:
                raise ValueError("DATASEC is not set in the header")
            return self.data

        return extract_section(self.data, data_section)

    def get_bias_section(self) -> np.ndarray:
        bias_section = self.header.get_str("BIASSEC")
        return extract_section(self.data, bias_section)

    @classmethod
    def from_array_and_dict(
        cls,
        header: dict[str, str | bool | int | float],
        data: np.ndarray,
        variance: np.ndarray,
    ) -> "Image":
        """
        Create a DataHeader from an array and a dictionary.
        """
        return Image(data=data, header=Headers(**header), variance=variance)


def _stupid_header_to_dict(header: Header) -> Headers:
    return Headers(**{k: v for k, v in sorted(header.items()) if v is not None})


@pipeline_task()
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


def load_all_data_extensions(science_file: Path, transpose: bool = False) -> list[np.ndarray]:
    """
    Load all data extensions from a FITS file.
    """
    logger = get_logger()
    with fits.open(science_file) as hdul:  # type: ignore
        data = [hdu.data for hdu in hdul if isinstance(hdu.data, np.ndarray)]
        logger.debug(f"Loaded {len(data)} data extensions from {science_file}")
        if transpose:
            data = [d.T for d in data]
            logger.debug(f"Transposed data with shapes {[d.shape for d in data]}")
        else:
            logger.debug(f"Data shapes: {[d.shape for d in data]}")
        return data


@pipeline_task()
def load_all_data_extensions_with_headers(science_file: Path, transpose: bool = False) -> list[Image]:
    """
    Load all data extensions and their headers from a FITS file.
    """
    logger = get_logger()
    with fits.open(science_file) as hdul:  # type: ignore
        data = [
            Image(data=hdu.data, header=_stupid_header_to_dict(hdu.header), variance=np.zeros_like(hdu.data))
            for hdu in hdul
            if isinstance(hdu.data, np.ndarray)
        ]
        logger.debug(f"Loaded {len(data)} data extensions with headers from {science_file}")
        if transpose:
            for d in data:
                d.data = d.data.T
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
