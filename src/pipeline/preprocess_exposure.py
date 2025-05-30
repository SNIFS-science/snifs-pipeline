from pathlib import Path

from pydantic import Field, FilePath
from pydantic_settings import BaseSettings

from pipeline import settings
from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_flow
from pipeline.resolver.common import FileType
from pipeline.tasks.common import Image
from pipeline.tasks.preprocessing import (
    DarkModel,
    add_poisson_noise_to_variance,
    build_bichip_from_fits,
    plot_images,
    subtract_bias,
    subtract_dark,
)


class PreprocessExposure(BaseSettings):
    primary_file: FilePath = Field(description="Location of the continuum exposure file. Relative to the data path.")
    bias_image_file: FileType.BIAS_IMAGE.Path = Field(default=None)  # type: ignore
    bias_model_file: FileType.BIAS_MODEL.Path = Field(default=None)  # type: ignore
    dark_image_file: FileType.DARK_IMAGE.Path = Field(default=None)  # type: ignore
    dark_model_file: FileType.DARK_MODEL.Path = Field(default=None)  # type: ignore
    binary_offset_model_file: FileType.BINARY_OFFSET_MODEL.Path = Field(default=None)  # type: ignore
    prefer_bias_image_over_model: bool = Field(default=True)
    use_dark_stack_if_possible: bool = Field(default=True)


@pipeline_flow()
def preprocess_exposure(config: PreprocessExposure) -> None:
    logger = get_logger()
    logger.info(f"Starting preprocessing with settings:\n {config.model_dump_json(indent=2)}\n")

    # Both R and B channels have one CCD read by two amplifiers.
    # The 'chip' terminology means the amps, not that there are two CCDs
    bichip = build_bichip_from_fits(config.primary_file, config.binary_offset_model_file)
    chip = bichip.assemble()  # noqa: F841

    chip.image = add_poisson_noise_to_variance(chip.image)

    if config.prefer_bias_image_over_model:
        bias_reference = Image.from_fits_file(config.bias_image_file, transpose=True)
    else:
        bias_reference = DarkModel.model_validate_json(config.bias_model_file.read_text())
    chip.image = subtract_bias(chip.image, bias_reference, chip.primary_headers)

    # The darks can use a model with a stacked image, so it's not either or.
    dark_model = DarkModel.model_validate_json(config.dark_model_file.read_text())
    dark_images = None
    if config.use_dark_stack_if_possible:
        dark_images = Image.stack_from_fits_file(config.dark_image_file, transpose=True)
    chip.image = subtract_dark(chip.image, dark_model, dark_images, chip.primary_headers)

    plot_images(settings.output_path / config.primary_file.stem)


if __name__ == "__main__":
    continuum_file = Path(__file__).parents[2] / "data/raw/runs/run_id=25_057_001/continuum_blue.fits"
    config = PreprocessExposure(primary_file=continuum_file)
    preprocess_exposure(config)
