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

    def time_term(self, time_on: float, time_off: float) -> float:
        if self.beta == -1:
            return np.log(time_off / time_on)
        return (np.power(time_off, self.betap1) - np.power(time_on, self.betap1)) / self.betap1

    def bias_time_term(self, time_on: float) -> float:
        return self.time_term(time_on, time_on + 40)

    def temperature_term(self, detector_temp: float) -> float:
        kelvin_temp = ABS_ZERO + detector_temp
        electron_gap = 1.11557 - 7.021e-4 * np.power(kelvin_temp, 2) / (kelvin_temp + 1108.0)
        return np.exp(-electron_gap / (2 * BOLTZMANN_CONSTANT * kelvin_temp)) * np.power(kelvin_temp, 1.5)


class DarkModel(BaseModel):
    sections: list[DarkModelSection]


@flag_skip("BIASDONE")
@plot()
@pipeline_task()
def subtract_bias(image: Image, reference: Image | DarkModel, primary_header: Headers) -> Image:
    if isinstance(reference, DarkModel):
        return subtract_bias_model(image, reference, primary_header)
    elif isinstance(reference, Image):
        return subtract_bias_image(image, reference)
    raise ValueError(f"Reference must be either a DarkModel or an Image, got {type(reference)} instead.")


def subtract_bias_model(image: Image, model: DarkModel, primary_header: Headers) -> Image:
    """Subtracts bias model following imagesnifs.cxx:298"""
    detector_temp = primary_header.get_float("DETTEMP")
    time_on_str = primary_header.get_optional_str("TIMEON")
    time_on = float(time_on_str) if time_on_str is not None and "." in time_on_str else None

    image = image.copy()
    for s in model.sections:
        to_remove = s.get_bias_sub(detector_temp, time_on)
        image.data[s.x_min : s.x_max, s.y_min : s.y_max] -= to_remove
    return image


def subtract_bias_image(image: Image, bias_image: Image) -> Image:
    assert bias_image.header.get_bool("BIASFRAM"), "Bias image must have BIASFRAM set to True"
    image = image.copy()

    assert bias_image.data.shape == image.data.shape, (
        f"Bias image shape {bias_image.data.shape} does not match image shape {image.data.shape}"
    )

    image.data -= bias_image.data
    image.variance += bias_image.variance

    # TODO: I'm a bit confused about imagesnifs.cxx:368 - Are we supposed to be determining the dark frame?
    return image


@flag_skip("DARKDONE")
@plot()
@pipeline_task()
def subtract_dark(image: Image, dark_image: Image) -> Image:
    assert dark_image.header.get_bool("DARKFRAM"), "Dark image must have DARKFRAM set to True"
    logger = get_logger()
    if not image.header.get_bool("OVSCDONE"):
        logger.warning("Image does not have OVSCDONE set, dark subtraction may not be correct.")
    assert image.data.shape == dark_image.data.shape, (
        f"Dark image shape {dark_image.data.shape} does not match image shape {image.data.shape}"
    )
    exposure_time = image.header.get_float("DARKTIME")
    dark_time = dark_image.header.get_float("DARKTIME")

    image = image.copy()
    scale = -exposure_time / dark_time
    image.data += scale * dark_image.data
    image.variance += dark_image.variance * (scale**2)
    return image
