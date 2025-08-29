import asyncio
from collections.abc import Awaitable
from datetime import datetime as dt
from functools import cached_property
from pathlib import Path

import polars as pl
from prefect.client.schemas import FlowRun
from pydantic import BaseModel, Field, computed_field

from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_flow, run_deployment
from pipeline.config.deployment import SnifsNerscDeploymentConfig, registry
from pipeline.flows.preprocess_exposure import PreprocessExposureConfig, preprocess_exposure
from pipeline.resolver.common import FileStoreEntry, PipelineStage
from pipeline.resolver.resolver import FlowConfig, get_run_id


class ProcessRunConfig(FlowConfig):
    run_id: str = Field(description="The ID of the run to process.", examples=["25_057_001"])
    inline_flows: bool = Field(default=True, description="Whether to run inline flows or give them to slurm")
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


@registry.register(SnifsNerscDeploymentConfig(max_walltime=120 * 60, memory=2 * 1024))
@pipeline_flow()
async def process_run(conf: ProcessRunConfig) -> None:
    conf.initialise_and_log()
    logger = get_logger()
    # All exposures for the run should be processed
    exposures = [
        FileStoreEntry.model_validate(entry)
        for entry in conf.resolver.file_store.filter(
            pl.col("run_id").eq(conf.run_id) & pl.col("level").eq("raw")
        ).to_dicts()
    ]

    if conf.inline_flows:
        processed = [
            preprocess_exposure(PreprocessExposureConfig(primary_file=Path(exp.file_path))) for exp in exposures
        ]
    else:
        coros: list[Awaitable[FlowRun]] = [
            run_deployment(
                flow_name="preprocess-exposure",
                deployment_name="preprocess-exposure",
                flow_run_name=f"preprocess_{exp.file_path}",
                parameters={"conf": {"primary_file": Path(exp.file_path)}},
                poll_interval=60,
            )
            for exp in exposures
        ]  # type: ignore
        flow_runs = await asyncio.gather(*coros)
        for run in flow_runs:
            logger.info(f"Flow run {run.name} ({run.id}) finished with state {run.state}")
            logger.info(f"Flow run {run.name} ({run.id}) finished with state result {run.state.result()}")
        processed = [await flow_run.state.result() for flow_run in flow_runs]  # type: ignore

    for p in processed:
        conf.resolver.ensure_file_exists(p.output_path)

    # We need to separate science exposures from others because they're the main focus
    # science_exposures = [p for p in processed if p.file_type == FileType.SCIENCE]

    # # TODO: This part should be done by the resolver, which means ensuring that output
    # # TODO: files from other flows are automatically added to the resolver on creation
    # for file_entry in science_exposures:
    #     image = Image.from_asdf(file_entry.output_path)

    #     continuums = [e for e in processed if e.file_type == FileType.CONTINUUM and e.channel == file_entry.channel]
    #     assert len(continuums) == 1
    #     continuum_image = Image.from_asdf(continuums[0].output_path)

    #     arcs = [e for e in processed if e.file_type == FileType.ARC and e.channel == file_entry.channel]
    #     assert len(arcs) == 1
    #     arc_image = Image.from_asdf(arcs[0].output_path)

    #     flat_fielded = calibrate_continuum(image, continuum_image)
    #     wavelength_calibrated = calibrate_wavelengths(flat_fielded, arc_image)
    #     wavelength_calibrated.to_asdf(conf.output_file)
    #     conf.resolver.ensure_file_exists(conf.output_file)
    #     summary = summarise_image(
    #         image,
    #         conf.resolver.get_file_metadata(file_entry.output_path),
    #         conf.output_summary_file,
    #         discriminator="process_exposure",
    #     )
    #     write_summary(conf.resolver, summary)


if __name__ == "__main__":

    async def main() -> None:
        config = ProcessRunConfig(run_id="25_056_084")
        await process_run(config)

    asyncio.run(main())
