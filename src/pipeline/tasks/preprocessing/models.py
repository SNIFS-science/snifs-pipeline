import numpy as np
from pydantic import BaseModel

from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_task
from pipeline.tasks.common import Headers, Image, Section, flag_skip
from pipeline.tasks.preprocessing.plots import plot

BOLTZMANN_CONSTANT = 8.617333262145e-5  # eV/K
ABS_ZERO = 273.15  # Celsius to Kelvin conversion factor


class DarkModelSection(Section):
    i0: float
    i1: float
    i2: float
    beta: float

    @property
    def betap1(self) -> float:
        return self.beta + 1

    def get_bias_sub(self, detector_temp: float, time_on: float | None) -> float:
        if time_on is None:
            return self.i0 + self.i2 * self.temperature_term(detector_temp)
        return self.i0 + self.i1 * self.bias_time_term(time_on) + self.i2 * self.temperature_term(detector_temp)

    def get_dark_sub(self, detector_temp: float, time_on: float | None, dark_time: float) -> float:
        if time_on is None:
            return (self.i0 + self.i2 * self.temperature_term(detector_temp)) * dark_time
        return (
            self.i0 + self.i2 * self.temperature_term(detector_temp) + self.i1 * self.dark_time_term(dark_time, time_on)
        )

    def time_term(self, time_on: float, time_off: float) -> float:
        if self.beta == -1:
            return np.log(time_off / time_on)
        return (np.power(time_off, self.betap1) - np.power(time_on, self.betap1)) / self.betap1

    def bias_time_term(self, time_on: float) -> float:
        return self.time_term(time_on, time_on + 40)

    def dark_time_term(self, dark_time: float, time_on: float) -> float:
        return self.time_term(time_on - dark_time, time_on)

    def temperature_term(self, detector_temp: float) -> float:
        kelvin_temp = ABS_ZERO + detector_temp
        electron_gap = 1.11557 - 7.021e-4 * np.power(kelvin_temp, 2) / (kelvin_temp + 1108.0)
        return np.exp(-electron_gap / (2 * BOLTZMANN_CONSTANT * kelvin_temp)) * np.power(kelvin_temp, 1.5)


class DarkModel(BaseModel):
    sections: list[DarkModelSection]


@flag_skip("BIASDONE")
@plot()
@pipeline_task()
def subtract_bias(image: Image, reference: Image | DarkModel, primary_headers: Headers) -> Image:
    if isinstance(reference, DarkModel):
        return subtract_bias_model(image, reference, primary_headers)
    elif isinstance(reference, Image):
        return subtract_bias_image(image, reference)
    raise ValueError(f"Reference must be either a DarkModel or an Image, got {type(reference)} instead.")


def subtract_bias_model(image: Image, model: DarkModel, primary_headers: Headers) -> Image:
    """Subtracts bias model following imagesnifs.cxx:298"""
    detector_temp = primary_headers.get_float("DETTEMP")
    time_on_str = primary_headers.get_optional_str("TIMEON")
    time_on = float(time_on_str) if time_on_str is not None and "." in time_on_str else None

    image = image.copy()
    for s in model.sections:
        to_remove = s.get_bias_sub(detector_temp, time_on)
        image.data[s.x_min : s.x_max, s.y_min : s.y_max] -= to_remove
    return image


def subtract_bias_image(image: Image, bias_image: Image) -> Image:
    assert bias_image.header.get_bool("BIASFRAM"), "Bias image must have BIASFRAM set to True"
    return image.subtract(bias_image)


@flag_skip("DARKDONE")
@plot()
@pipeline_task()
def subtract_dark(image: Image, model: DarkModel, dark_images: list[Image] | None, primary_headers: Headers) -> Image:
    if dark_images is None:
        return subtract_dark_model(image, model, primary_headers)
    return subtract_dark_stack(image, dark_images, model, primary_headers)


def subtract_dark_model(image: Image, model: DarkModel, primary_headers: Headers) -> Image:
    detector_temp = primary_headers.get_float("DETTEMP")
    time_on_str = primary_headers.get_optional_str("TIMEON")
    time_on = float(time_on_str) if time_on_str is not None and "." in time_on_str else None
    dark_time = primary_headers.get_float("DARKTIME")

    # This warning comes from imagesnifs.cxx:410
    if time_on is not None and time_on < dark_time:
        get_logger().warning(
            f"TIMEON {time_on} is less than DARKTIME {dark_time}. This may lead to incorrect dark subtraction."
        )

    image = image.copy()
    for s in model.sections:
        to_remove = s.get_dark_sub(detector_temp, time_on, dark_time)
        image.data[s.x_min : s.x_max, s.y_min : s.y_max] -= to_remove
    return image


def subtract_dark_stack(image: Image, dark_images: list[Image], model: DarkModel, primary_headers: Headers) -> Image:
    assert len(dark_images) == 3, "Dark stack must contain exactly 3 images (i0, i1, i2 terms)"
    for dark_image in dark_images:
        assert image.data.shape == dark_image.data.shape, (
            f"Dark image shape {dark_image.data.shape} does not match image shape {image.data.shape}"
        )
    assert len(model.sections) == 1, "Dark model should have exactly one section for dark subtraction"
    section = model.sections[0]

    logger = get_logger()
    if not image.header.get_bool("OVSCDONE"):
        logger.warning("Image does not have OVSCDONE set, dark subtraction may not be correct.")

    dark_time = primary_headers.get_float("DARKTIME")
    time_on = primary_headers.get_float("TIMEON")
    temperature = primary_headers.get_float("DETTEMP")

    coefficients = [
        section.i0 * dark_time,
        section.i1 * section.dark_time_term(dark_time=dark_time, time_on=time_on),
        section.i2 * section.temperature_term(temperature) * dark_time,
    ]
    for dark_image, coeff in zip(dark_images, coefficients, strict=True):
        logger.debug(f"Subtracting dark image with coefficient {coeff}")
        image = image.subtract(dark_image, coeff)
    return image
