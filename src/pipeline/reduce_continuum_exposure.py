from datetime import datetime as dt
from pathlib import Path

from pydantic import Field, FilePath
from pydantic_settings import BaseSettings

from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_flow
from pipeline.resolver.resolver import Resolver
from pipeline.tasks import (
    preprocess_exposure,
)
from pipeline.tasks.build_filestore import build_filestore


class ContinuumReduction(BaseSettings):
    continuum_file: FilePath = Field(description="Location of the continuum exposure file. Relative to the data path.")
    arc_file: FilePath = Field(default=None, description="Location of the arc file(s).")  # type: ignore
    weather_file: FilePath = Field(default=None, description="Location of the weather file")  # type: ignore
    detector_last_on_time: dt | None = Field(default=None, description="The time the detector was last turned on.")
    use_cache: bool = Field(default=True, description="Use cached data when possible")
    make_plots: bool = Field(default=True, description="Make plots of the data")

    def resolve_missing(self, resolver: Resolver) -> None:
        logger = get_logger()
        logger.info("Resolving missing paths for continuum reduction config")
        primary = resolver.get_file_metadata(self.continuum_file)
        if self.arc_file is None:
            self.arc_file = resolver.get_match_path("ARC", primary)
        if self.weather_file is None:
            self.weather_file = resolver.get_match_path("WEATHER", primary)

        logger.info(f"Final config:\n {self.model_dump_json(indent=2)}")


@pipeline_flow()
def reduce_continuum_channel_exposure(config: ContinuumReduction) -> None:
    logger = get_logger()
    logger.info(f"Starting contiuum exposure reduction with settings:\n {config.model_dump_json(indent=2)}")

    # Before we get into the processing, we want to update any external data sources we might need
    # In the future, this would be managed by Prefect automatically scheduling these updates
    # update_cfht_weather()

    # Load in the existing file store and ensure its up to date
    resolver = build_filestore()
    config.resolve_missing(resolver)

    # augment_science_file()
    preprocess_exposure(config.continuum_file, resolver)
    # correct_dichoric()
    # remove_continuum()
    # calibrate_with_flats()


if __name__ == "__main__":
    continuum_file = Path(__file__).parents[2] / "data/runs/run_id=25_057_001/continuum_blue.fits"
    config = ContinuumReduction(continuum_file=continuum_file)
    reduce_continuum_channel_exposure(config)
