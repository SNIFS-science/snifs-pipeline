import asyncio
from datetime import datetime as dt
from functools import cached_property
from pathlib import Path

import polars as pl
from pydantic import BaseModel, Field, computed_field

from pipeline.common.image import Image
from pipeline.common.prefect_utils import pipeline_flow
from pipeline.config.deployment import SnifsNerscDeploymentConfig, registry
from pipeline.flows.preprocess_exposure import PreprocessExposureConfig, preprocess_exposure
from pipeline.resolver.common import FileType, PipelineStage
from pipeline.resolver.resolver import FlowConfig, get_run_id
#from pipeline.tasks.processing.make_parameter_matrix import make_matrix
from pipeline.tasks.processing.wavelength_arc_calibration import calibrate_wavelength_arc


class ProcessRunConfig(FlowConfig):
    run_id: str = Field(description="The ID of the run to process.", examples=["25_057_001"])
    refresh_filestore: bool = Field(default=True)

    @cached_property
    def output_folder(self) -> Path:
        return self.resolver.output_path / f"level=processed/runs/run_id={self.run_id}/flow_run_id={get_run_id()}"

    @cached_property
    def public_folder(self) -> Path:
        return self.resolver.public_path / f"level=processed/runs/run_id={self.run_id}/flow_run_id={get_run_id()}"

    @computed_field
    @property
    def output_file(self) -> Path:
        return self.output_folder / f"{PipelineStage.PROCESSED.value}_runid={self.run_id}.asdf"

    @computed_field
    @property
    def output_summary_file(self) -> Path:
        return self.public_folder / f"{PipelineStage.PROCESSED.value}_summary.json"


class ProcessRunSummary(BaseModel):
    output_path: str
    time_processed: dt | None = None
    object: str | None = None
    object_ra: str | None = None
    object_dec: str | None = None
    run_id: str | None = None
    observation_id: str | None = None


@registry.register(SnifsNerscDeploymentConfig(max_walltime=120 * 60, memory=3 * 1952))
@pipeline_flow()
async def process_run(conf: ProcessRunConfig) -> None:
    conf.initialise_and_log()

    # First, we want to grab all raw exposures associated with this run.
    run_filter = pl.col("run_id").eq(conf.run_id) & pl.col("level").eq("raw")
    exposure_paths = [Path(p) for p in conf.resolver.file_store.filter(run_filter)["file_path"]]
    for path in exposure_paths:
        print(f"Found exposure file: {path}")
    # For each of these exposures, we want to trigger the preprocess flow to clean them up.
    processed = [preprocess_exposure(PreprocessExposureConfig(primary_file=path)) for path in exposure_paths]

    # The joy of things being on disk is that they can be deleted at any time. Double
    # check that all the expected output images exist
    for p in processed:
        conf.resolver.ensure_file_exists(p.output_path)

    # We need to separate science exposures from others because they're the main focus
    arc_exposures = [p for p in processed if p.file_type == FileType.ARC]
    science_exposures = [p for p in processed if p.file_type == FileType.SCIENCE]

    """for arc in arc_exposures:
        if arc.channel == "B":
            # wavelength_calibration_config = WavelengthConfig(
            #    arc_path=arc.output_path,
            # )
            sparse_matrix = make_matrix(112, arc.run_id, partial=False) # type: ignore
            # calibrate_wavelength_arc(arc)
    for science_image in science_exposures:
        if science_image.channel == "B":
            print(science_image.run_id)
            sparse_matrix = make_matrix(112, science_image.run_id, partial=False) # type: ignore
            # fit(sparse_matrix, Path(science_image.output_path)) """
    # for file_entry in science_exposures:
    #     image = Image.from_asdf(file_entry.output_path)

        continuums = [e for e in processed if e.file_type == FileType.CONTINUUM and e.channel == file_entry.channel]
        assert len(continuums) == 1
        continuum_image = Image.from_asdf(continuums[0].output_path)

        arcs = [e for e in processed if e.file_type == FileType.ARC and e.channel == file_entry.channel]
        assert len(arcs) == 1
        arc_image = Image.from_asdf(arcs[0].output_path)

        flat_fielded = calibrate_continuum(image, continuum_image)
        wavelength_calibrated = calibrate_wavelengths(flat_fielded, arc_image)
        wavelength_calibrated.to_asdf(conf.output_file)
        conf.resolver.ensure_file_exists(conf.output_file)
        summarise_image(
            image,
            conf.resolver.get_file_metadata(file_entry.output_path),
            conf.output_summary_file,
            discriminator="process_exposure",
        )
        # write_summary(conf.resolver, summary)


if __name__ == "__main__":

    async def main() -> None:
        config = ProcessRunConfig(run_id="25_291_003")
        await process_run(config)

    asyncio.run(main())
