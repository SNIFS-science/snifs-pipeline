from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from pipeline.common.prefect_utils import pipeline_task
from pipeline.tasks.common import (
    Headers,
    Image,
    listify,
    load_all_data_extensions_with_headers,
    load_headers,
)
from pipeline.tasks.preprocessing.plots import log_image_data, plot

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

        This will also flip the R channel, as per bichip.cxx:283
        """

        # Let's do some basic checks before assembling into a single chip
        data_secs = {image.header.get_str("DATASEC") for image in self.images}
        assert len(data_secs) == 1, f"All images must have the same DATASEC, got {data_secs}"

        # Go through and reverse directions as needed
        datas = [image.get_data_section() * image.header.get_float("GAIN") for image in self.images]
        variances = [image.get_data_section_variance() * (image.header.get_float("GAIN") ** 2) for image in self.images]
        channel = self.primary_headers.get_str("CHANNEL")
        # If you're in the R channel, chip0 is on the right and chip1 is on the left
        if channel == "R":
            datas = datas[::-1]
            variances = variances[::-1]

        # If not R channel, flip the first amplifier in the X direction
        if channel != "R":
            datas[0] = datas[0][::-1, :]
            variances[0] = variances[0][::-1, :]

        # The second amplifier is always flipped in the x direction
        datas[1] = datas[1][::-1, :]
        variances[1] = variances[1][::-1, :]

        # Images get concatenated along the X axis (see bichip.cxx:267)
        compound_data = np.vstack(datas)
        compound_variance = np.vstack(variances)

        primary_headers = self.primary_headers.copy()
        for i, image in enumerate(self.images):
            primary_headers[f"CCD{i}GAIN"] = image.header.get_float("GAIN")
            primary_headers[f"RDNOISE{i}"] = image.header.get_float("RDNOISE") * image.header.get_float("GAIN")
            primary_headers[f"OVSCMAX{i}"] = image.header.get_float("OVSCMAX")
            primary_headers[f"OVSCMED{i}"] = image.header.get_float("OVSCMED")
            primary_headers[f"OEPARAM{i}"] = image.header.get_float_list("OEPARAM")

        # So the saturation calculation is more complex
        saturations: list[float] = [
            (
                image.header.get_float("SATURATE")
                - image.header.get_float("OVSCMAX")
                - abs(
                    max(
                        image.header.get_float_list("OEPARAM")[0],
                        image.header.get_float_list("OEPARAM")[0]
                        + image.header.get_float_list("OEPARAM")[1] * image.data.shape[1],
                    )
                )
            )
            * image.header.get_float("GAIN")
            * 0.99
            for image in self.images
        ]
        primary_headers["SATURATE"] = saturations
        for key in ["RDNOISE", "OVSCMED", "GAIN", "AMPSEC", "DETSEC", "BIASSEC"]:
            if key in primary_headers:
                del primary_headers[key]  # These are not needed in the final header

        combined_header = Headers.merge_all(*[image.header for image in self.images])
        # Ensure certain keys are not in the header
        for key in ["BIASSEC"]:
            if key in combined_header:
                del combined_header[key]

        final_image = Image(data=compound_data, header=combined_header, variance=compound_variance)
        log_image_data("bichip.assemble", final_image)
        return Chip(
            primary_headers=self.primary_headers,
            image=final_image,
        )


@plot()
@pipeline_task()
def split_chip(images: list[Image]) -> list[Image]:
    if len(images) == 2:
        # If this is a detcom file and thus has two extensions to start with, great!
        # All we need to do then is standardise some header values and return the images.
        for i, image in enumerate(images):
            image.header["CCDNUM"] = i
            image.header["GAIN"] = image.header.get_float(f"CCD{i}GAIN")
            image.header["CCDNAMP"] = 1
            image.header["SATURATE"] = image.header.get_int("CCD{i}SAT", 65535)
            # As per algocams.cxx:125, detcom images drop the first 11 columns of overscan!
            # The fact this is twelve below is because this is 1-indexed
            # EDIT: Actually, going off the comment both detcom and otcom want 10 columns dropped?
            # comment: remove a few pixels (10)
            b, _, _ = image.get_bias_section()
            image.header["BIASSEC"] = f"[{b.x_min + 11}:{b.x_max},{b.y_min + 1}:{b.y_max}]"

        return images
    image = images[0]
    new_data_headers = []
    num_amps = image.header.get_int("CCDNAMP", 2)
    assert num_amps == 2, f"Expected 2 amplifiers, got {num_amps}"
    full_data = image.get_data_section()  # TODO: standardise this
    n_data = full_data.shape[0] // num_amps
    _, full_bias, _ = image.get_bias_section()
    n_bias = full_bias.shape[0] // num_amps
    for i in range(num_amps):
        # data_array = extract_section_from_label(data.data, data.header.get_str(f"DATASEC{i}"))
        # bias_array = extract_section_from_label(data.data, data.header.get_str(f"BIASSEC{i}"))
        # Ha - psyche! You can't use the DATASEC and BIASSEC index keywords, they're wrong!
        # Instead just cut the data and bias up!
        data_array = full_data[i * n_data : (i + 1) * n_data, :]
        bias_array = full_bias[i * n_bias : (i + 1) * n_bias, :]

        data_array = data_array[:, ::-1]
        bias_array = bias_array[:, ::-1]

        if i == 1:
            # The second chip is flipped in the X direction
            data_array = data_array[::-1, :]
            bias_array = bias_array[::-1, :]

        combined = np.vstack((data_array, bias_array))

        chip_header = image.header | {
            "GAIN": image.header[f"CCD{i}GAIN"],  # This is set in the hack fits keywords
            "CCDNAMP": 1,
            "DATASEC": f"[1:{data_array.shape[0]},1:{data_array.shape[1]}]",
            # As per algocams.cxx:235, otcom images discard the first 10 rows.
            "BIASSEC": (
                f"[{data_array.shape[0] + 11}:{data_array.shape[0] + bias_array.shape[0] - 1},1:{bias_array.shape[1]}]"
            ),
            "CCDSEC": image.header[f"CCDSEC{i}"],
            "AMPSEC": image.header[f"AMPSEC{i}"],
            "DETSEC": image.header[f"DETSEC{i}"],
            "CCDBIN": image.header[f"CCDBIN{i + 1}"],
            "SATURATE": image.header.get_int(f"CCD{i}SAT", 65535),
            "CCDTEMP": image.header.get_optional_float(
                "CCDTMP", image.header.get_optional_float("DETTEMP", default=None)
            ),
            "CCDNUM": i,
        }
        new_data_headers.append(
            Image.from_array_and_dict(chip_header, combined, np.zeros_like(combined, dtype=np.float64))
        )

    return new_data_headers  # R channel has chips reversed


def override_headers(images: list[Image], primary_headers: Headers) -> list[Image]:
    result = [image.copy() for image in images]
    channel = primary_headers.get_str("CHANNEL")
    for i, image in enumerate(result):
        image.header["GAIN"] = GAINS[channel][i]

    return result


def build_bichip_from_fits(path: Path) -> tuple[Headers, list[Image]]:
    """Load a BiChip from a FITS file. Note for conventions used and broken,
    we're following most of what the older C++ code did. One of those
    things which may be confusing, is that in a 2D numpy array,
    this is generally the to as the (columns, rows) ordering. In the
    case for this code, we have shape = (rows, columns) = (x,y) ordering."""
    images = load_all_data_extensions_with_headers(path, transpose=True)
    primary_headers = load_headers(path)

    # In the original preprocessing, there was an algorithm for both
    # detcom and a SNFactory variant. We'll just be using the variant.
    # Here we want to ensure both detcom and otcom come back looking the same,
    # which in this case means two images, one from each amplifier.
    images = split_chip(images)
    images = override_headers(images, primary_headers)
    return primary_headers, images


def handle_saturation_image(image: Image) -> Image:
    """Handle saturation in the data

    The process is to look for readings above the saturation level, and then
    set their variance to infinity. Because saturation has a bleed, we also set
    pixels in the touching rows (not columns though) to infinity as well.

    Note that the first axis is the Y axis, and the second axis is the X axis, and
    the bleed which happens (according to image.cxx:593) is in the Y direction, aka
    across the rows of 4096 size.

    Greg agrees with this: There are channel stops between the columns (y-dir with 4096 pixels),
    so charge will bleed up a column in the y direction.
    """
    level = image.header.get_int("SATURATE", 65535)
    new_image = image.copy()
    assert new_image.variance is not None, "Variance must be set before handling saturation"
    saturation_mask = new_image.data >= level
    saturation_mask[:, :-1] |= saturation_mask[:, 1:]
    saturation_mask[:, 1:] |= saturation_mask[:, :-1]
    new_image.variance[saturation_mask] = np.inf
    return new_image


handle_saturation = plot()(pipeline_task()(listify(handle_saturation_image)))
