from pathlib import Path

import numpy as np
from pydantic import BaseModel

from pipeline.common import Image, Section, flag_skip, get_logger, pipeline_task
from pipeline.tasks.plotting import plot
from pipeline.tasks.plotting.plots import plot_standalone
from pipeline.tasks.preprocessing.common import add_poisson_noise_to_variance

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


def subtract_bias_and_add_poisson(
    image: Image,
    prefer_bias_image_over_model: bool,
    bias_image_file: Path | None = None,
    bias_model_file: Path | None = None,
) -> Image:
    # You have two options for bias subtraction: either a bias image or a bias model.
    if prefer_bias_image_over_model and bias_image_file is not None:
        bias_reference = Image.from_fits_file(bias_image_file, transpose=True)

        # Interestingly, if we have a bias image, we want to subtract is and *then* add the poisson noise.
        image = subtract_bias(image, bias_reference)
        return add_poisson_noise_to_variance(image)
    elif bias_model_file is not None:
        bias_reference = DarkModel.model_validate_json(bias_model_file.read_text())
        # But if we have a bias model, we want to add the poisson noise *before* subtracting the bias.
        image = add_poisson_noise_to_variance(image)
        return subtract_bias(image, bias_reference)
    else:
        raise ValueError(
            "No bias image or bias model provided. Please provide either a bias image file or a bias model file."
        )


def subtract_dark(
    image: Image, dark_model_file: Path, dark_image_file: Path | None = None, use_dark_stack_if_possible: bool = True
) -> Image:
    # The darks can use a model *with* a stacked image, so it's not either or.
    dark_model = DarkModel.model_validate_json(dark_model_file.read_text())
    dark_images = None
    if use_dark_stack_if_possible and dark_image_file is not None:
        dark_images = Image.stack_from_fits_file(dark_image_file, transpose=True)
        return subtract_dark_stack(image, dark_images, dark_model)
    return subtract_dark_model(image, dark_model)


@flag_skip("bias_done")
@plot()
@pipeline_task()
@plot_standalone("subtract_bias")
def subtract_bias(image: Image, reference: Image | DarkModel) -> Image:
    if isinstance(reference, DarkModel):
        return subtract_bias_model(image, reference)
    elif isinstance(reference, Image):
        return subtract_bias_image(image, reference)
    raise ValueError(f"Reference must be either a DarkModel or an Image, got {type(reference)} instead.")


def subtract_bias_model(image: Image, model: DarkModel) -> Image:
    """Subtracts bias model following imagesnifs.cxx:298 code."""
    detector_temp = image.header.get_float("detector_temperature")
    time_on_str = image.header.get_optional_str("time_on_seconds")
    time_on = float(time_on_str) if time_on_str is not None and "." in time_on_str else None

    original = image.data
    image = image.copy()
    for s in model.sections:
        to_remove = s.get_bias_sub(detector_temp, time_on)
        image.data[s.x_min : s.x_max, s.y_min : s.y_max] -= to_remove

    mean_difference = np.mean(image.data - original)
    image.header.set("bias_done", True)
    image.add_function_lineage(f"Subtracted bias model with {detector_temp=}, {time_on=}, and {mean_difference=:.4f}")
    return image


def subtract_bias_image(image: Image, bias_image: Image) -> Image:
    assert bias_image.header.get_bool("bias_frame"), "Bias image must have bias_frame set to True"
    new_image = image.subtract(bias_image)
    mean_difference = np.mean(new_image.data - image.data)
    new_image.add_function_lineage(f"Subtracted bias image with {mean_difference=:.4f} mean difference")
    new_image.header.set("bias_done", True)
    return new_image


@flag_skip("dark_done")
@plot()
@pipeline_task()
@plot_standalone("subtract_dark_model")
def subtract_dark_model(image: Image, model: DarkModel) -> Image:
    detector_temp = image.header.get_float("detector_temperature")
    time_on_str = image.header.get_optional_str("time_on_seconds")
    time_on = float(time_on_str) if time_on_str is not None and "." in time_on_str else None
    dark_time = image.header.get_float("dark_seconds")

    # This warning comes from imagesnifs.cxx:410
    if time_on is not None and time_on < dark_time:
        get_logger().warning(
            f"time_on_seconds {time_on} is less than dark_seconds"
            f" {dark_time}. This may lead to incorrect dark subtraction."
        )

    original = image.data
    image = image.copy()
    for s in model.sections:
        to_remove = s.get_dark_sub(detector_temp, time_on, dark_time)
        image.data[s.x_min : s.x_max, s.y_min : s.y_max] -= to_remove
    mean_difference = np.mean(image.data - original)
    image.add_function_lineage(
        f"Subtracted dark model with {detector_temp=}, {time_on=}, {dark_time=}, and {mean_difference=:.4f}"
    )
    image.header.set("dark_done", True)
    return image


@flag_skip("dark_done")
@plot()
@pipeline_task()
@plot_standalone("subtract_dark_stack")
def subtract_dark_stack(image: Image, dark_images: list[Image], model: DarkModel) -> Image:
    assert len(dark_images) == 3, "Dark stack must contain exactly 3 images (i0, i1, i2 terms)"
    for dark_image in dark_images:
        assert image.data.shape == dark_image.data.shape, (
            f"Dark image shape {dark_image.data.shape} does not match image shape {image.data.shape}"
        )
    assert len(model.sections) == 1, "Dark model should have exactly one section for dark subtraction"
    section = model.sections[0]

    logger = get_logger()
    if not image.header.get_bool("overscan_done"):
        logger.warning("Image does not have overscan_done set, dark subtraction may not be correct.")

    dark_time = image.header.get_float("dark_seconds")
    time_on = image.header.get_float("time_on_seconds")
    temperature = image.header.get_float("detector_temperature")

    coefficients = [
        section.i0 * dark_time,
        section.i1 * section.dark_time_term(dark_time=dark_time, time_on=time_on),
        section.i2 * section.temperature_term(temperature) * dark_time,
    ]
    original = image.data
    for dark_image, coeff in zip(dark_images, coefficients, strict=True):
        logger.debug(f"Subtracting dark image with coefficient {coeff}")
        image = image.subtract(dark_image, coeff)

    mean_difference = np.mean(image.data - original)
    image.add_function_lineage(
        f"Subtracted dark stack with {temperature=}, {time_on=}, {dark_time=}, and {mean_difference=:.4f}"
    )
    image.header.set("dark_done", True)
    return image
