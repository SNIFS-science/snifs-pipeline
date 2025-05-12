from collections import namedtuple
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from pipeline.resolver.common import FileType
from pipeline.resolver.resolver import Resolver
from pipeline.tasks.common import Headers, load_all_data_extensions, load_headers, load_headers_from_data
from pipeline.tasks.preprocessing.binary_offset import correct_binary_offset

GAINS = {
    "B": [0.773, 0.744],
    "R": [0.757, 0.770],
    "Phot": [1.618, 1.576, 1.51, 1.52],
}


class Chip(BaseModel):
    primary_headers: Headers
    data: np.ndarray
    variance: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChipMaker:
    def assemble(self) -> Chip:
        """Assemble the chip from the data and variance."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class BiChip(BaseModel, ChipMaker):
    primary_headers: Headers
    data1: np.ndarray
    data2: np.ndarray
    variance1: np.ndarray
    variance2: np.ndarray
    headers1: Headers
    headers2: Headers

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    return pixels[section.y_min : section.y_max : section.y_dir, section.x_min : section.x_max : section.x_dir]


def split_otcom_chip(pixels: np.ndarray, headers: Headers) -> tuple[list[np.ndarray], list[Headers]]:
    data_list = []
    chip_headers = []
    num_amps = headers.get_int("CCDNAMP", 2)
    assert num_amps == 2, f"Expected 2 amplifiers, got {num_amps}"
    for i in range(num_amps):
        data = extract_section(pixels, headers.get_str(f"DATASEC{i}"))
        bias = extract_section(pixels, headers.get_str(f"BIASSEC{i}"))
        combined = np.hstack((data, bias))
        data_list.append(combined)

        chip_headers.append(
            headers
            | {
                "ORIGINAL_GAIN": headers[f"CCD{i}GAIN"],  # This is set in the hack fits keywords
                "CCDNAMP": 1,
                "DATASEC": f"[1:{data.shape[1]},1:{data.shape[0]}]",
                "BIASSEC": f"[{data.shape[1]} + 1:{data.shape[1] + bias.shape[1]},1:{bias.shape[0]}]",
                "CCDSEC": headers[f"CCDSEC{i}"],
                "AMPSEC": headers[f"AMPSEC{i}"],
                "DETSEC": headers[f"DETSEC{i}"],
                "CCDBIN": headers[f"CCDBIN{i}"],
                "CCDTEMP": headers.get_optional_float("CCDTMP", headers.get_optional_float("DETTEMP", default=None)),
            }
        )
    return data_list, chip_headers


def build_bichip_from_fits(path: Path, resolver: Resolver) -> BiChip:
    """Load a BiChip from a FITS file."""
    data_list = load_all_data_extensions(path)
    data_headers = load_headers_from_data(path)  # noqa: F841
    primary_headers = load_headers(path)  # noqa: F841

    # Ensure some default values are set in the header
    primary_headers.set_default({"SATURATE": 65535})

    # In the original preprocessing, there was an algorithm for both
    # detcom and a SNFactory variant. We'll just be using the variant.
    if len(data_list) == 1:
        # One extension means otcom, as it's packaged the two amplifiers together
        split_otcom_chip(data_list[0], data_headers[0])  # type: ignore

    # Set up the variance, and start with the Poisson noise that'd we'd expect
    variances = [data.copy() for data in data_list]
    # And handle saturation by inf'ing out the variance and accounting for bleed
    for data, variance, header in zip(data_list, variances, data_headers, strict=True):
        handle_saturation(data, variance, primary_headers.merge(header).get_float("SATURATE", 65535.0))

    # Binary offset model is only derived for 2 chip models.
    if len(data_list) == 2:
        # Load the binary offset model
        bom_path = resolver.get_match_path(FileType.BINARY_OFFSET_MODEL, path)
        correct_binary_offset(data_list, bom_path)
    #
    # TODO: preprocessor.cxx 259
    # TODO: BuildRawBiChip logic
    # TODO: HackFitsKeywords
    # TODO: create variance
    # TODO: handle saturation
    # TODO: binary offset invocation
    # TODO: overscan subtraction
    return BiChip(...)  # type: ignore


def handle_saturation(data: np.ndarray, variance: np.ndarray, level: float) -> None:
    """Handle saturation in the data

    For more details on this, see Emmanuel Gangler's thesis, section 3.3.2, PDF page 56.
    You can find the original French and a Google-Translated version in the
    docs/pdfs folder in this repository.

    The process is to look for readings above the saturation level, and then
    set their variance to infinity. Because saturation has a bleed, we also set
    pixels in the touching rows (not columns though) to infinity as well.

    Note that the first axis is the Y axis, and the second axis is the X axis, and
    the bleed which happens (according to image.cxx:593) is in the Y direction, aka
    across the rows of 1024 size.
    """
    saturation_mask = data > level
    saturation_mask[:-1, :] |= saturation_mask[1:, :]
    saturation_mask[1:, :] |= saturation_mask[:-1, :]
    variance[saturation_mask] = np.inf
