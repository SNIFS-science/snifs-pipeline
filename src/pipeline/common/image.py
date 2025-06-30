from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from pydantic import BaseModel, ConfigDict

from pipeline.common import Headers, Section, get_logger


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

    def _get_section_from_header(self, key: str, ensure_header_exists: bool = True) -> Section | None:
        header_value = self.header.get_optional_str(key)
        if header_value is None:
            if ensure_header_exists:
                raise ValueError("DATASEC is not set in the header")
            return None
        return Section.from_str(header_value)

    def get_section(self, section: Section) -> tuple[np.ndarray, np.ndarray]:
        return self._extract_section(self.data, section), self._extract_section(self.variance, section)

    def mask_bad_section(self, sec: Section) -> None:
        self.variance[sec.x_min : sec.x_max : sec.x_dir, sec.y_min : sec.y_max : sec.y_dir] = np.inf

    def _extract_section(self, pixels: np.ndarray, section: Section) -> np.ndarray:
        return pixels[section.x_min : section.x_max : section.x_dir, section.y_min : section.y_max : section.y_dir]

    def get_data_section(self, enforce_datasec: bool = True) -> tuple[Section, np.ndarray, np.ndarray]:
        section = self._get_section_from_header("DATASEC", ensure_header_exists=enforce_datasec)
        if section is None:
            section = Section(x_min=0, x_max=self.data.shape[0], x_dir=1, y_min=0, y_max=self.data.shape[1], y_dir=1)

        data, var = self.get_section(section)
        return section, data, var

    def get_bias_section(self) -> tuple[Section, np.ndarray, np.ndarray]:
        section = self._get_section_from_header("BIASSEC", ensure_header_exists=True)
        assert section is not None, "BIASSEC is not set in the header"
        data, var = self.get_section(section)
        return section, data, var

    def get_ccd_section(self) -> tuple[Section, np.ndarray, np.ndarray]:
        section = self._get_section_from_header("CCDSEC", ensure_header_exists=True)
        assert section is not None, "CCDSEC is not set in the header"
        data, var = self.get_section(section)
        return section, data, var

    def add(self, image: "Image", scale: float = 1.0) -> "Image":
        """
        Add another image to this one, scaling the data by the given scale factor.
        """

        if self.data.shape != image.data.shape:
            logger = get_logger()
            logger.warning(f"Image shapes do not match: {self.data.shape} vs {image.data.shape}")
        new_data = self.data + scale * image.data
        new_variance = self.variance + scale**2 * image.variance
        return Image(data=new_data, header=self.header.copy(), variance=new_variance)

    def subtract(self, image: "Image", scale: float = 1.0) -> "Image":
        return self.add(image, scale=-scale)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Image):
            return NotImplemented
        return (
            np.array_equal(self.data, other.data)
            and np.array_equal(self.variance, other.variance)
            and self.header == other.header
        )

    @classmethod
    def from_array_and_dict(
        cls,
        header: dict[str, str | bool | int | float | list[str] | list[int] | list[float]],
        data: np.ndarray,
        variance: np.ndarray,
    ) -> "Image":
        """
        Create a DataHeader from an array and a dictionary.
        """
        return Image(data=data, header=Headers(**header), variance=variance)

    @classmethod
    def from_fits_file(
        cls,
        fits_file: Path | str,
        data_index: int = 0,
        variance_index: int = 1,
        required_variance: bool = False,
        transpose: bool = False,
    ) -> "Image":
        if isinstance(fits_file, str):
            fits_file = Path(fits_file)
        with fits.open(fits_file) as hdul:  # type: ignore
            has_variance = len(hdul) > variance_index and isinstance(hdul[variance_index].data, np.ndarray)  # type: ignore
            has_data = len(hdul) > data_index and isinstance(hdul[data_index].data, np.ndarray)  # type: ignore
            if not has_data:
                raise ValueError(f"FITS file {fits_file} does not have a data extension at index {data_index}")
            data = hdul[data_index].data  # type: ignore
            if required_variance and not has_variance:
                raise ValueError(f"FITS file {fits_file} does not have a variance extension at index {variance_index}")
            image = Image(
                data=data,
                header=Headers.from_astropy_header(hdul[data_index].header),  # type: ignore
                variance=hdul[variance_index].data if has_variance else np.zeros_like(data),  # type: ignore
            )
            if transpose:
                image.data = image.data.T
                image.variance = image.variance.T

            return image

    @classmethod
    def stack_from_fits_file(cls, fits_file: Path | str, transpose: bool = False) -> list["Image"]:
        if isinstance(fits_file, str):
            fits_file = Path(fits_file)
        images = []
        with fits.open(fits_file) as hdul:  # type: ignore
            for hdu in hdul:
                if isinstance(hdu.data, np.ndarray):  # type: ignore
                    image = Image(
                        data=hdu.data,  # type: ignore
                        header=Headers.from_astropy_header(hdu.header),  # type: ignore
                        variance=np.zeros_like(hdu.data, dtype=np.float64),  # type: ignore
                    )
                    if transpose:
                        image.data = image.data.T
                        image.variance = image.variance.T
                    images.append(image)
        return images

    def to_fits(self, fits_file: Path | str, primary_header: Headers) -> None:
        if isinstance(fits_file, str):
            fits_file = Path(fits_file)
        parent = fits_file.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        logger = get_logger()
        logger.debug(f"Saving image to {fits_file}")
        hdul = fits.HDUList(
            [
                fits.PrimaryHDU(data=None, header=primary_header.to_astropy_header()),
                fits.ImageHDU(data=self.data, header=self.header.to_astropy_header(), name="FLUX"),
                fits.ImageHDU(data=self.variance, name="VARIANCE") if self.variance is not None else None,
            ]
        )
        hdul.writeto(fits_file, overwrite=True)
