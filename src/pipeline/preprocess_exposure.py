from pathlib import Path

from pydantic import Field, FilePath

from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_flow
from pipeline.resolver.common import FileType
from pipeline.tasks.build_filestore import FlowConfig
from pipeline.tasks.common import Image, load_all_data_extensions_with_headers, load_headers
from pipeline.tasks.preprocessing import plot_detailed_images
from pipeline.tasks.preprocessing.bichips import assemble_bichip_to_image, handle_saturation, split_and_standardise
from pipeline.tasks.preprocessing.binary_offset import correct_binary_offset
from pipeline.tasks.preprocessing.common import add_poisson_noise_to_variance, ensure_float64
from pipeline.tasks.preprocessing.cosmetics import cheat_cosmetics, handle_special_red_cosmetics
from pipeline.tasks.preprocessing.flats import apply_custom_red_flat
from pipeline.tasks.preprocessing.models import DarkModel, subtract_bias, subtract_dark
from pipeline.tasks.preprocessing.overscan import add_overscan_variance, correct_even_odd, subtract_offset
from pipeline.tasks.preprocessing.plots import clear_output_path, plot


class PreprocessExposure(FlowConfig):
    primary_file: FilePath = Field(description="Location of the continuum exposure file. Relative to the data path.")
    bias_image_file: FileType.BIAS.Path = Field(default=None)  # type: ignore
    bias_model_file: FileType.BIAS_MODEL.Path = Field(default=None)  # type: ignore
    dark_image_file: FileType.DARK.Path = Field(default=None)  # type: ignore
    dark_model_file: FileType.DARK_MODEL.Path = Field(default=None)  # type: ignore
    flat_image_file: FileType.CONTINUUM.Path = Field(default=None)  # type: ignore
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
    file_name = "P25_057_001_005_07_R.fits" if channel == "R" else "P25_057_001_006_07_B.fits"
    # file_name = "P25_159_030_005_07_R.fits" if channel == "R" else "P25_057_001_006_07_B.fits"
    dan = Image.from_fits_file(Path(__file__).parents[2] / f".data_dump/{file_name}", transpose=True)
    dan.variance[dan.variance > 1e6] = np.inf
    return dan


@pipeline_flow()
def preprocess_exposure(config: PreprocessExposure) -> None:
    logger = get_logger()
    primary = config.fetch_metadata(config.primary_file)
    logger.info(f"Starting preprocessing with settings:\n{config.model_dump_json(indent=2)}\n")
    logger.info(f"Primary file:\n{primary.model_dump_json(indent=2)}\n")
    assert primary.channel is not None, "Primary file must have a channel defined in the headers."

    # Start the preprocessing pipeline
    images = load_all_data_extensions_with_headers(config.primary_file, transpose=True)
    primary_headers = load_headers(config.primary_file)
    images = split_and_standardise(images)
    images = handle_saturation(images)

    if len(images) == 2:  # Binary offset model is only derived for 2 chip models.
        images = correct_binary_offset(images, config.binary_offset_model_file)

    images = ensure_float64(images)
    images = correct_even_odd(images)
    images = add_overscan_variance(images)
    images = subtract_offset(images)
    image, primary_headers = assemble_bichip_to_image(images, primary_headers)
    image = add_poisson_noise_to_variance(image)

    # You have two options for bias subtraction: either a bias image or a bias model.
    if config.prefer_bias_image_over_model:
        bias_reference = Image.from_fits_file(config.bias_image_file, transpose=True)
    else:
        bias_reference = DarkModel.model_validate_json(config.bias_model_file.read_text())
    image = subtract_bias(image, bias_reference, primary_headers)

    # The darks can use a model *with* a stacked image, so it's not either or.
    dark_model = DarkModel.model_validate_json(config.dark_model_file.read_text())
    dark_images = None
    if config.use_dark_stack_if_possible:
        dark_images = Image.stack_from_fits_file(config.dark_image_file, transpose=True)
    image = subtract_dark(image, dark_model, dark_images, primary_headers)

    if primary.channel == "R":
        image = handle_special_red_cosmetics(image, primary_headers)
    image = cheat_cosmetics(image, primary.channel)
    image = debug_comparison(image, primary.channel)

    if config.flat_image_file is not None:
        pass
    elif primary.channel == "R":
        image = apply_custom_red_flat(image)

    clear_output_path(primary)
    # plot_bias_sections(primary)
    plot_detailed_images(primary, start="subtract_bias")


if __name__ == "__main__":
    # file = Path(__file__).parents[2] / "data/raw/runs/run_id=25_121_118/bias_red.fits"
    file = Path(__file__).parents[2] / "data/raw/runs/run_id=25_057_001/continuum_red.fits"
    # file = Path(__file__).parents[2] / "data/raw/runs/run_id=25_159_030/continuum_red.fits"
    # file = Path(__file__).parents[2] / "data/raw/runs/run_id=25_057_001/continuum_blue.fits"
    config = PreprocessExposure(primary_file=file)
    preprocess_exposure(config)
