import shutil
from functools import cached_property
from pathlib import Path

from pydantic import Field, computed_field

from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_flow
from pipeline.config.deployment import SnifsDeploymentConfig, registry
from pipeline.resolver.common import FileType, PipelineStage
from pipeline.resolver.resolver import FlowConfig
from pipeline.tasks.loaders import clear_directory, load_headers, load_images_from_file
from pipeline.tasks.plotting.plots import plot_bias_sections, plot_detailed_images  # noqa: F401
from pipeline.tasks.preprocessing import (
    add_overscan_variance,
    apply_custom_red_flat,
    assemble_bichip_to_image,
    cheat_cosmetics,
    correct_binary_offset,
    correct_even_odd,
    determine_timeon,
    ensure_float64,
    handle_saturation,
    handle_special_red_cosmetics,
    split_and_standardise,
    subtract_dark,
    subtract_offset,
)
from pipeline.tasks.preprocessing.cosmic_rays import remove_cosmic_rays  # noqa: F401
from pipeline.tasks.preprocessing.models import subtract_bias_and_add_poisson
from pipeline.tasks.summaries import summarise_image, write_summary


class PreprocessExposureConfig(FlowConfig):
    primary_file: Path = Field(description="Location of the continuum exposure file. Relative to the data path.")
    bias_image_file: FileType.BIAS.Path = Field(default=None, validate_default=True)
    bias_model_file: FileType.BIAS_MODEL.Path = Field(default=None, validate_default=True)
    dark_image_file: FileType.DARK.Path = Field(default=None, validate_default=True)
    dark_model_file: FileType.DARK_MODEL.Path = Field(default=None, validate_default=True)
    flat_image_file: FileType.CONTINUUM.OptionalPath = Field(default=None, validate_default=True)
    ccd_on_time_file: FileType.CCD_ON_TIMES.Path = Field(default=None, validate_default=True)
    binary_offset_model_file: FileType.BINARY_OFFSET_MODEL.Path = Field(default=None, validate_default=True)
    prefer_bias_image: bool = Field(default=True)
    use_dark_stack_if_possible: bool = Field(default=True)
    refresh_filestore: bool = Field(default=True)

    @cached_property
    def output_folder(self) -> Path:
        primary = self.fetch_metadata(self.primary_file)

        return (
            self.resolver.output_path
            / f"processed_runs/run_id={primary.run_id}/obstype={primary.type}/channel={primary.channel}"
        )

    @cached_property
    def public_folder(self) -> Path:
        primary = self.fetch_metadata(self.primary_file)
        return (
            self.resolver.public_path
            / f"processed_runs/run_id={primary.run_id}/obstype={primary.type}/channel={primary.channel}"
        )

    @property
    def raw_file_duplication_path(self) -> Path:
        """This is the path where the raw file will be duplicated to, for posterity."""
        return self.output_folder / f"{PipelineStage.RAW.value}.fits"

    @computed_field
    @property
    def output_image_file(self) -> Path:
        return self.output_folder / f"{PipelineStage.PREPROCESSED.value}.asdf"

    @computed_field
    @property
    def output_summary_file(self) -> Path:
        return self.public_folder / f"{PipelineStage.PREPROCESSED.value}_summary.json"


@registry.register(SnifsDeploymentConfig(max_walltime=10 * 60, qos="debug"))
@pipeline_flow()
def preprocess_exposure(conf: PreprocessExposureConfig) -> Path:
    logger = get_logger()
    primary = conf.fetch_metadata(conf.primary_file)
    conf.initialise_and_log()
    logger.info(f"Primary file:\n{primary.model_dump_json(indent=2)}\n")

    assert primary.channel is not None, "Primary file must have a channel defined in the headers."
    assert conf.ccd_on_time_file is not None, "CCD on time file must be provided."
    assert conf.binary_offset_model_file is not None, "Binary offset model file must be provided."
    assert conf.dark_model_file is not None, "Dark model file must be provided."

    clear_directory(conf.output_folder)
    clear_directory(conf.public_folder)
    images = load_images_from_file(conf.primary_file, transpose=True)
    primary_headers = load_headers(conf.primary_file)

    if "TIMEON" not in primary_headers:
        primary_headers["TIMEON"] = determine_timeon(conf.ccd_on_time_file, primary_headers)
    images = split_and_standardise(images, primary.channel, primary_headers)
    images = handle_saturation(images)
    if len(images) == 2:  # Binary offset model is only derived for 2 chip models.
        images = correct_binary_offset(images, conf.binary_offset_model_file)
    images = ensure_float64(images)
    images = correct_even_odd(images)
    images = add_overscan_variance(images)
    images = subtract_offset(images)
    image = assemble_bichip_to_image(images, primary_headers)
    image = subtract_bias_and_add_poisson(image, conf.prefer_bias_image, conf.bias_image_file, conf.bias_model_file)
    image = subtract_dark(image, conf.dark_model_file, conf.dark_image_file, conf.use_dark_stack_if_possible)
    if primary.channel == "R":
        image = handle_special_red_cosmetics(image)
    image = cheat_cosmetics(image, primary.channel)
    # Apply the custom red flat if we can't find a flat at all, or we're processing a red flat.
    if primary.channel == "R" and (conf.flat_image_file is None or conf.flat_image_file == conf.primary_file):
        image = apply_custom_red_flat(image)
    # image = remove_cosmic_rays(image)

    # TODO: need add_parangle.py from Daniel, along with the parangel.txt file to get the information from

    # debug_comparison(image, primary.channel, primary.run_id, primary.type)
    conf.raw_file_duplication_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(conf.primary_file, conf.raw_file_duplication_path)
    image.to_asdf(conf.output_image_file)
    # plot_bias_sections(primary, conf.public_folder)
    # plot_detailed_images(primary, conf.public_folder)

    summary = summarise_image(image, primary, conf.output_summary_file, discriminator="preprocess_exposure")
    write_summary(conf.resolver, summary)

    return conf.output_image_file


if __name__ == "__main__":
    raw_dir = Path(__file__).parents[2] / "data/raw"
    files = [
        raw_dir / "runs/run_id=25_057_001/continuum_red.fits",
        raw_dir / "runs/run_id=25_159_030/continuum_red.fits",
        raw_dir / "runs/run_id=25_057_001/continuum_blue.fits",
    ]
    for file in files:
        config = PreprocessExposureConfig(primary_file=file)
        preprocess_exposure(config)
