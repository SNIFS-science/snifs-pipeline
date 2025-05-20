from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from pipeline.resolver.common import FileType
from pipeline.resolver.resolver import Resolver
from pipeline.tasks.common import (
    Headers,
    Image,
    extract_section,
    load_all_data_extensions_with_headers,
    load_headers,
)
from pipeline.tasks.preprocessing.binary_offset import correct_binary_offset
from pipeline.tasks.preprocessing.overscan import correct_even_odd
from pipeline.tasks.preprocessing.plots import plotted_task

GAINS = {
    "B": [0.773, 0.744],
    "R": [0.757, 0.770],
    "Phot": [1.618, 1.576, 1.51, 1.52],
}


class Chip(BaseModel):
    primary_headers: Headers
    image: Image

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChipMaker:
    def assemble(self) -> Chip:
        """Assemble the chip from the data and variance."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class BiChip(BaseModel, ChipMaker):
    primary_headers: Headers
    images: list[Image] = Field(min_length=2, max_length=2)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def data(self) -> np.ndarray:
        return np.hstack((self.images[0].data, self.images[1].data))

    @property
    def variance(self) -> np.ndarray:
        assert len(self.images) == 2, "Variance is only available for two images"
        assert self.images[0].variance is not None and self.images[1].variance is not None
        return np.hstack((self.images[0].variance, self.images[1].variance))

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
        combined_header = Headers.merge_all(*[image.header for image in self.images])
        return Chip(
            primary_headers=self.primary_headers,
            image=Image(data=self.data, header=combined_header, variance=self.variance),
        )


@plotted_task()
def split_otcom_chip(images: list[Image]) -> list[Image]:
    assert len(images) == 1, f"Expected one image, got {len(images)}"
    data = images[0]
    new_data_headers = []
    num_amps = data.header.get_int("CCDNAMP", 2)
    assert num_amps == 2, f"Expected 2 amplifiers, got {num_amps}"
    for i in range(num_amps):
        data_array = extract_section(data.data, data.header.get_str(f"DATASEC{i}"))
        bias_array = extract_section(data.data, data.header.get_str(f"BIASSEC{i}"))
        combined = np.hstack((data_array, bias_array))

        chip_header = data.header | {
            "ORIGINAL_GAIN": data.header[f"CCD{i}GAIN"],  # This is set in the hack fits keywords
            "CCDNAMP": 1,
            "DATASEC": f"[1:{data.data.shape[0]},1:{data.data.shape[1]}]",
            "BIASSEC": f"[{data.data.shape[0]} + 1:{data.data.shape[0] + bias_array.shape[0]},1:{bias_array.shape[1]}]",
            "CCDSEC": data.header[f"CCDSEC{i}"],
            "AMPSEC": data.header[f"AMPSEC{i}"],
            "DETSEC": data.header[f"DETSEC{i}"],
            "CCDBIN": data.header[f"CCDBIN{i}"],
            "CCDTEMP": data.header.get_optional_float(
                "CCDTMP", data.header.get_optional_float("DETTEMP", default=None)
            ),
        }
        new_data_headers.append(Image.from_array_and_dict(chip_header, combined, np.zeros_like(combined)))

    return new_data_headers


def build_bichip_from_fits(path: Path, resolver: Resolver) -> BiChip:
    """Load a BiChip from a FITS file."""
    images = load_all_data_extensions_with_headers(path, transpose=True)
    primary_headers = load_headers(path)

    # In the original preprocessing, there was an algorithm for both
    # detcom and a SNFactory variant. We'll just be using the variant.
    if len(images) == 1:
        # One extension means otcom, as it's packaged the two amplifiers together
        split_otcom_chip(images)

    # Set up the variance, and start with the Poisson noise that'd we'd expect
    for image in images:
        image.variance = image.data.copy().astype(np.float64)

    # And handle saturation by inf'ing out the variance and accounting for bleed
    images = handle_saturation(images)

    # Binary offset model is only derived for 2 chip models.
    if len(images) == 2:
        bom_path = resolver.get_match_path(FileType.BINARY_OFFSET_MODEL, path)
        images = correct_binary_offset(images, bom_path)

    # The next section is overscan substraction, and its strange. The logic seems to be:
    # 1. Only for the first chip, see if it has overscan (aka BIASSEC) is set
    # 2. If it does set a bool flag that fOddEven=True
    # 3. Now "Correct" every chip (overscan.cxx:479)
    # Now, every chip I've seen has a BIASSEC, so I feel like this should always be true and thus
    # we should always be doing the odd even correction. Hopefully I'm not wrong.
    images = correct_even_odd(images)

    # 4. Check if the fOddEven is set, double check OEPARAM is not set as that means its already corrected
    # 5. Otherwise, "SubstractOddEven" (overscan.cxx:372). This is a monster of a function.
    # 6. Set the header OEPARAM to the two-length list of param coming out from substract odd even
    # 7. Add overscan variance
    #    a. check that OVSCNOIS is not set in the header or is 0
    #    b. extract overscan region, take variance of the whole thing and add it flat to the image variance
    #    c. set OVSCNOIS to 1
    # 8. Subtract offset
    #    a. This calls computeLines, which calls ComputeLinesMean and ImproveLinesMean
    #    b. then Subtract calls SubstractRamp, and this is also a big ol monster function

    # TODO: preprocessor.cxx 259
    # TODO: BuildRawBiChip logic
    # TODO: HackFitsKeywords
    # TODO: create variance
    # TODO: handle saturation
    # TODO: binary offset invocation
    # TODO: overscan subtraction
    return BiChip(primary_headers=primary_headers, images=images)  # type: ignore


@plotted_task()
def handle_saturation(images: list[Image]) -> list[Image]:
    return [handle_saturation_image(image) for image in images]


def handle_saturation_image(image: Image) -> Image:
    """Handle saturation in the data

    For more details on this, see Emmanuel Gangler's thesis, section 3.3.2, PDF page 56.
    You can find the original French and a Google-Translated version in the
    docs/pdfs folder in this repository.

    The process is to look for readings above the saturation level, and then
    set their variance to infinity. Because saturation has a bleed, we also set
    pixels in the touching rows (not columns though) to infinity as well.

    Note that the first axis is the Y axis, and the second axis is the X axis, and
    the bleed which happens (according to image.cxx:593) is in the Y direction, aka
    across the rows of 4096 size.

    Greg agrees with this: There are channel stops between the columns (y-dir with 4096 pixels),
    so charge will bleed up a column in the y direction.
    """
    level = image.header.get_float("SATURATE", 65535.0)
    new_image = image.copy()
    assert new_image.variance is not None, "Variance must be set before handling saturation"
    saturation_mask = new_image.data > level
    saturation_mask[:, :-1] |= saturation_mask[:, 1:]
    saturation_mask[:, 1:] |= saturation_mask[:, :-1]
    new_image.variance[saturation_mask] = np.inf
    return new_image
