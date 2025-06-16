from pathlib import Path

from pydantic import Field, FilePath

from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_flow
from pipeline.resolver.common import FileType
from pipeline.tasks.build_filestore import FlowConfig
from pipeline.tasks.common import Image
from pipeline.tasks.preprocessing import (
    add_poisson_noise_to_variance,
    build_bichip_from_fits,
    plot_images,
)
from pipeline.tasks.preprocessing.bichips import BiChip, handle_saturation
from pipeline.tasks.preprocessing.binary_offset import correct_binary_offset
from pipeline.tasks.preprocessing.common import ensure_float64
from pipeline.tasks.preprocessing.cosmetics import handle_cosmetics
from pipeline.tasks.preprocessing.overscan import add_overscan_variance, correct_even_odd, subtract_offset
from pipeline.tasks.preprocessing.plots import clear_output_path, plot, plot_bias_sections


class PreprocessExposure(FlowConfig):
    primary_file: FilePath = Field(description="Location of the continuum exposure file. Relative to the data path.")
    bias_image_file: FileType.BIAS.Path = Field(default=None)  # type: ignore
    bias_model_file: FileType.BIAS_MODEL.Path = Field(default=None)  # type: ignore
    dark_image_file: FileType.DARK.Path = Field(default=None)  # type: ignore
    dark_model_file: FileType.DARK_MODEL.Path = Field(default=None)  # type: ignore
    binary_offset_model_file: FileType.BINARY_OFFSET_MODEL.Path = Field(default=None)  # type: ignore
    prefer_bias_image_over_model: bool = Field(default=True)
    use_dark_stack_if_possible: bool = Field(default=True)
    refresh_filestore: bool = Field(default=True)


# This nasty little code loads in the preprocessed fits file Daniel sent me, so we can compare the results
@plot()
def debug_comparison(image: Image, channel: str) -> Image:
    import numpy as np

    # 005 and 006 are R and B continuum
    # 011 are the arcs
    # file_name = "P25_057_001_005_07_R.fits" if channel == "R" else "P25_057_001_006_07_B.fits"
    file_name = "P25_159_030_005_07_R.fits" if channel == "R" else "P25_057_001_006_07_B.fits"
    dan = Image.from_fits_file(Path(__file__).parents[2] / f".data_dump/{file_name}", transpose=True)
    dan.variance[dan.variance > 1e6] = np.inf
    return dan


@pipeline_flow()
def preprocess_exposure(config: PreprocessExposure) -> None:
    logger = get_logger()
    primary = config.metadata(config.primary_file)
    logger.info(f"Starting preprocessing with settings:\n{config.model_dump_json(indent=2)}\n")
    logger.info(f"Primary file:\n{primary.model_dump_json(indent=2)}\n")
    clear_output_path(primary)

    primary_headers, images = build_bichip_from_fits(config.primary_file)
    images = handle_saturation(images)

    if len(images) == 2 and False:  # Binary offset model is only derived for 2 chip models.
        images = correct_binary_offset(images, config.binary_offset_model_file)

    images = ensure_float64(images)
    images = correct_even_odd(images)
    images = add_overscan_variance(images)
    images = subtract_offset(images)
    chip = BiChip(primary_headers=primary_headers, images=images).assemble()
    chip.image = add_poisson_noise_to_variance(chip.image)

    # if config.prefer_bias_image_over_model:
    #     bias_reference = Image.from_fits_file(config.bias_image_file, transpose=True)
    # else:
    #     bias_reference = DarkModel.model_validate_json(config.bias_model_file.read_text())
    # chip.image = subtract_bias(chip.image, bias_reference, chip.primary_headers)

    # The darks can use a model with a stacked image, so it's not either or.
    # dark_model = DarkModel.model_validate_json(config.dark_model_file.read_text())
    # dark_images = None
    # if config.use_dark_stack_if_possible:
    #     dark_images = Image.stack_from_fits_file(config.dark_image_file, transpose=True)
    # chip.image = subtract_dark(chip.image, dark_model, dark_images, chip.primary_headers)

    chip.image = handle_cosmetics(chip.image, chip.primary_headers)
    chip.image = debug_comparison(chip.image, chip.primary_headers.get_str("CHANNEL"))
    plot_bias_sections(primary)
    plot_images(primary)


if __name__ == "__main__":
    # file = Path(__file__).parents[2] / "data/raw/runs/run_id=25_121_118/bias_red.fits"
    # file = Path(__file__).parents[2] / "data/raw/runs/run_id=25_057_001/continuum_red.fits"
    file = Path(__file__).parents[2] / "data/raw/runs/run_id=25_159_030/continuum_red.fits"
    # file = Path(__file__).parents[2] / "data/raw/runs/run_id=25_057_001/continuum_blue.fits"
    config = PreprocessExposure(primary_file=file)
    preprocess_exposure(config)
