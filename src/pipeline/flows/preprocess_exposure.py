from datetime import datetime as dt
from datetime import timezone as tz
from functools import cached_property
from pathlib import Path

from pydantic import BaseModel, Field, computed_field

from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import create_markdown_artifact, pipeline_flow
from pipeline.config.deployment import SnifsNerscDeploymentConfig, registry
from pipeline.resolver.common import FileType, PipelineStage
from pipeline.resolver.resolver import FlowConfig, get_run_id
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
    use_cache: bool = Field(default=True, description="Whether to use cached files where possible.")

    @cached_property
    def output_folder(self) -> Path:
        primary = self.fetch_metadata(self.primary_file)

        return (
            self.resolver.output_path / f"level=preprocessed/"
            f"run_id={primary.run_id}/type={primary.file_type}/channel={primary.channel}/object={primary.object}/observation={primary.observation_id}/flow_run_id={get_run_id()}"
        )

    @cached_property
    def public_folder(self) -> Path:
        primary = self.fetch_metadata(self.primary_file)
        return (
            self.resolver.public_path / f"level=preprocessed/"
            f"run_id={primary.run_id}/type={primary.file_type}/channel={primary.channel}/object={primary.object}/observation={primary.observation_id}/flow_run_id={get_run_id()}"
        )

    @computed_field
    @property
    def output_image_file(self) -> Path:
        return self.output_folder / f"{PipelineStage.PREPROCESSED.value}.asdf"

    @computed_field
    @property
    def output_summary_file(self) -> Path:
        return self.public_folder / f"{PipelineStage.PREPROCESSED.value}_summary.json"

    @computed_field
    @property
    def flow_summary_file(self) -> Path:
        return self.public_folder / f"{PipelineStage.PREPROCESSED.value}_flow_summary.json"


class PreprocessSummary(BaseModel):
    source_path: str
    output_path: str
    file_type: FileType
    channel: str
    time_observation: dt | None = None
    time_processed: dt | None = None
    object: str | None = None
    run_id: str | None = None
    observation_id: str | None = None


@registry.register(SnifsNerscDeploymentConfig(max_walltime=10 * 60, memory=4 * 1952))
@pipeline_flow()
def preprocess_exposure(conf: PreprocessExposureConfig) -> PreprocessSummary:
    logger = get_logger()

    # I'll overly comment this function so it acts as a reference for future flows.
    # Please don't feel the need to put such verbose comments in future flows!

    # Useful for when this flow is called by process_run and you don't want to preprocess over and over
    # This should work fine right now, but if we wanted a better implementation, I'd be tempted to
    # implement some common caching mechanism in the resolver or as a method on the config object itself
    # that uses a hash (of the config) and checks that against the hash (which would be saved out) in
    # the summary file.
    if conf.use_cache and conf.flow_summary_file.exists() and conf.output_image_file.exists():
        logger.info(f"Using cached preprocessed exposure at {conf.output_image_file}")
        return PreprocessSummary.model_validate_json(conf.flow_summary_file.read_text())

    # These three lines I'd expect to be reusable almost everywhere. We get the descriptors
    # for the primary file (like run_id, object name, channel, etc), log out the config so
    # we know exactly how this flow is being run, and then log out the primary file metadata
    primary = conf.fetch_metadata(conf.primary_file)
    conf.initialise_and_log()
    logger.info(f"Primary file:\n{primary.model_dump_json(indent=2)}\n")

    # These asserts are effectively just making sure the resolver was successful in turning
    # the default unspecified attributres (as None) into actual file paths.
    assert primary.channel is not None, "Primary file must have a channel defined in the headers."
    assert conf.ccd_on_time_file is not None, "CCD on time file must be provided."
    assert conf.binary_offset_model_file is not None, "Binary offset model file must be provided."
    assert conf.dark_model_file is not None, "Dark model file must be provided."

    # Now clearing some folders and loading the primary imae and headers we want to work with.
    clear_directory(conf.output_folder)
    clear_directory(conf.public_folder)
    # Everything above this comment line could be mostly refactored into a common function to
    # reduce duplicated code.
    images = load_images_from_file(conf.primary_file, transpose=True)
    primary_headers = load_headers(conf.primary_file)

    # Here our actual algorithm begins. Each of these functions is designed to be fairly flat and
    # operate on the images/image with similar signatures. This makes the code easier to read, and
    # also a lot easier to test in isolation.
    if "time_on_seconds" not in primary_headers:
        primary_headers["time_on_seconds"] = determine_timeon(conf.ccd_on_time_file, primary_headers)
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
    image = remove_cosmic_rays(image)

    # And now we have the boilerplate file output section, along with some extra diagnostic plots.
    image.header["level"] = "preprocess"
    image.to_asdf(conf.output_image_file)
    image.to_fits(conf.output_image_file.with_suffix(".fits"))
    conf.resolver.ensure_file_exists(conf.output_image_file)
    plot_bias_sections(primary, conf.public_folder)
    plot_detailed_images(primary, conf.public_folder)

    # This summary could be removed if the pipeline database is done differently (replaced with an INSERT or API)
    # but right now the purpose of this is get a file out that's got data mirrored in SQLite, which is what
    # the streamlit dashboard (removed from repo, can find it in prior commits) uses.
    summary = summarise_image(image, primary, conf.output_summary_file, discriminator="preprocess_exposure")
    if "task_run_id" in summary:
        write_summary(conf.resolver, summary)
    del image

    # This summary is what's returned by this flow. Notice that, because we might be passing from one node
    # to another, a different slurm job, or in the future an entirely different server/HPC system, we're
    # not returning massive data arrays, just simple json-serializable metadata about what was done,
    # including the output file location so we can load it in downstream steps if needed.
    result = PreprocessSummary(
        source_path=str(conf.primary_file.resolve()),
        output_path=str(conf.output_image_file.resolve()),
        file_type=primary.file_type,
        channel=primary.channel,
        time_observation=primary.time_observation,
        time_processed=dt.now(tz=tz.utc),
        object=primary.object,
        run_id=primary.run_id,
        observation_id=primary.observation_id,
    )
    conf.flow_summary_file.write_text(result.model_dump_json(indent=2))
    # This markdown artifact step is used to regenerate the PreprocessSummary object for cached
    # results, and to ensure we can pass this object back when invoking this flow via
    # run_deployment (in prefect_utils.py) to parallelise this section.
    # For an example in how to inline flows or run them via Prefect, see
    # https://github.com/SNIFS-science/snifs-pipeline/blob/743f50af79cb46ce9ef301eb6e65cb695f657c22/src/pipeline/flows/process_run.py#L73
    create_markdown_artifact(f"""```json\n{result.model_dump_json(indent=2)}\n```""", key="result")
    return result


if __name__ == "__main__":
    raw_dir = Path(__file__).parents[3] / "data/level=raw"
    files = [
        raw_dir / "runs/run_id=25_056_084/science_red.fits",
        # raw_dir / "runs/run_id=25_056_084/science_blue.fits",
        # raw_dir / "runs/run_id=25_057_001/continuum_red.fits",
        # raw_dir / "runs/run_id=25_057_001/continuum_blue.fits",
        # raw_dir / "runs/run_id=25_121_118/bias_red.fits",
        # raw_dir / "runs/run_id=25_159_030/continuum_red.fits",
    ]
    for file in files:
        assert Path(file).exists(), f"File {file} does not exist."
        config = PreprocessExposureConfig(primary_file=file)
        preprocess_exposure(config)
