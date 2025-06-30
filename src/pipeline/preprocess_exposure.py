import re
from functools import cached_property
from pathlib import Path

from pydantic import Field, FilePath, computed_field

from pipeline.common.image import Image
from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_flow
from pipeline.resolver.common import FileType, PipelineStage
from pipeline.tasks.build_filestore import FlowConfig
from pipeline.tasks.loaders import load_headers, load_images_from_file
from pipeline.tasks.preprocessing import (
    DarkModel,
    add_overscan_variance,
    add_poisson_noise_to_variance,
    apply_custom_red_flat,
    assemble_bichip_to_image,
    cheat_cosmetics,
    clear_output_path,
    correct_binary_offset,
    correct_even_odd,
    determine_timeon,
    ensure_float64,
    handle_saturation,
    handle_special_red_cosmetics,
    plot,
    plot_bias_sections,
    plot_detailed_images,
    split_and_standardise,
    subtract_bias,
    subtract_dark,
    subtract_offset,
)


class PreprocessExposure(FlowConfig):
    primary_file: FilePath = Field(description="Location of the continuum exposure file. Relative to the data path.")
    bias_image_file: FileType.BIAS.Path = Field(default=None)  # type: ignore
    bias_model_file: FileType.BIAS_MODEL.Path = Field(default=None)  # type: ignore
    dark_image_file: FileType.DARK.Path = Field(default=None)  # type: ignore
    dark_model_file: FileType.DARK_MODEL.Path = Field(default=None)  # type: ignore
    flat_image_file: FileType.CONTINUUM.OptionalPath = Field(default=None)
    ccd_on_time_file: FileType.CCD_ON_TIMES.Path = Field(default=None)  # type: ignore
    binary_offset_model_file: FileType.BINARY_OFFSET_MODEL.Path = Field(default=None)  # type: ignore
    prefer_bias_image_over_model: bool = Field(default=True)
    use_dark_stack_if_possible: bool = Field(default=True)
    refresh_filestore: bool = Field(default=True)

    @cached_property
    def output_folder(self) -> Path:
        primary = config.fetch_metadata(config.primary_file)

        return (
            self.resolver.output_path
            / f"processed_runs/run_id={primary.run_id}/obstype={primary.type}/channel={primary.channel}"
        )

    @computed_field
    @property
    def output_file(self) -> Path:
        return self.output_folder / f"{PipelineStage.PREPROCESSED.value}.fits"


@plot()
def debug_comparison(image: Image, channel: str, run_id: str | None, obstype: str) -> Image:
    import numpy as np

    assert run_id is not None, "Run ID must be provided for debugging comparison."

    # 005 and 006 are R and B continuum
    # 011 are the arcs
    data_dump_dir = Path(__file__).parents[2] / ".data_dump"
    expected_pattern = f"P{run_id}_*_{channel}.fits"
    found_files = list(data_dump_dir.glob(expected_pattern))
    if len(found_files) > 1:
        extra_match = f".*_011_0._{channel}.fits" if obstype == "ARC" else f".*_00[56]_0._{channel}.fits"
        found_files = [f for f in found_files if re.match(extra_match, f.name)]
    assert len(found_files) == 1, (
        f"Expected exactly one file matching {expected_pattern}, found {len(found_files)}: {found_files}."
    )
    get_logger().info(f"Debugging comparison with file: {found_files[0]}")
    dan = Image.from_fits_file(found_files[0], transpose=True)
    dan.variance[dan.variance > 1e6] = np.inf
    return dan


@pipeline_flow()
def preprocess_exposure(config: PreprocessExposure) -> Path:
    logger = get_logger()
    primary = config.fetch_metadata(config.primary_file)
    logger.info(f"Starting preprocessing with settings:\n{config.model_dump_json(indent=2)}\n")
    logger.info(f"Primary file:\n{primary.model_dump_json(indent=2)}\n")
    assert primary.channel is not None, "Primary file must have a channel defined in the headers."
    clear_output_path(config.output_folder)

    images = load_images_from_file(config.primary_file, transpose=True)
    primary_headers = load_headers(config.primary_file)

    # We need to augment the primary headers with some information. Namely, some of the files
    # will have the time the detector was last switched on (needed for dark subtraction),
    # but sometimes this information won't be present.
    if "TIMEON" not in primary_headers:
        primary_headers["TIMEON"] = determine_timeon(config.ccd_on_time_file, primary_headers)

    images = split_and_standardise(images, primary.channel)
    images = handle_saturation(images)

    if len(images) == 2:  # Binary offset model is only derived for 2 chip models.
        images = correct_binary_offset(images, config.binary_offset_model_file)

    images = ensure_float64(images)
    images = correct_even_odd(images)
    images = add_overscan_variance(images)
    images = subtract_offset(images)
    image, primary_headers = assemble_bichip_to_image(images, primary_headers)

    # You have two options for bias subtraction: either a bias image or a bias model.
    if config.prefer_bias_image_over_model:
        bias_reference = Image.from_fits_file(config.bias_image_file, transpose=True)

        # Interestingly, if we have a bias image, we want to subtract is and *then* add the poisson noise.
        image = subtract_bias(image, bias_reference, primary_headers)
        image = add_poisson_noise_to_variance(image)
    else:
        bias_reference = DarkModel.model_validate_json(config.bias_model_file.read_text())
        # But if we have a bias model, we want to add the poisson noise *before* subtracting the bias.
        image = add_poisson_noise_to_variance(image)
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

    # Apply the custom red flat if we can't find a flat at all, or we're processing a red flat.
    if config.flat_image_file is not None and config.flat_image_file != config.primary_file:
        pass
    elif primary.channel == "R":
        image = apply_custom_red_flat(image)

    debug_comparison(image, primary.channel, primary.run_id, primary.type)

    image.to_fits(config.output_file, primary_headers)
    plot_bias_sections(primary, config.output_folder)
    plot_detailed_images(primary, config.output_folder)

    return config.output_file


if __name__ == "__main__":
    raw_dir = Path(__file__).parents[2] / "data/raw"
    files = [
        raw_dir / "runs/run_id=25_057_001/continuum_red.fits",
        raw_dir / "runs/run_id=25_159_030/continuum_red.fits",
        raw_dir / "runs/run_id=25_057_001/continuum_blue.fits",
    ]
    for file in files:
        config = PreprocessExposure(primary_file=file)
        preprocess_exposure(config)
