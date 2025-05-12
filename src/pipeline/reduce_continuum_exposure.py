from datetime import datetime as dt
from pathlib import Path
from typing import Self

from pydantic import Field, FilePath, model_validator

from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_flow
from pipeline.preprocess_exposure import PreprocessExposure
from pipeline.tasks import (
    preprocess_exposure,
)


class ContinuumReduction(PreprocessExposure):
    arc_file: FilePath | None = Field(default=None, description="Location of the arc file.")  # type: ignore
    weather_file: FilePath | None = Field(default=None, description="Location of the weather file")  # type: ignore
    detector_last_on_time: dt | None = Field(default=None, description="The time the detector was last turned on.")

    @model_validator(mode="after")
    def resolve_missing(self) -> Self:
        assert self.resolver is not None, "Resolver should not be None at this point"
        logger = get_logger()
        primary = self.resolver.get_file_metadata(self.primary_file)
        if self.arc_file is None:
            self.arc_file = self.resolver.get_match_path("ARC", primary)
        if self.weather_file is None:
            self.weather_file = self.resolver.get_match_path("WEATHER", primary)

        logger.info(f"Final config:\n {self.model_dump_json(indent=2)}")
        return self


@pipeline_flow()
def reduce_continuum_channel_exposure(config: ContinuumReduction) -> None:
    logger = get_logger()
    logger.info(f"Starting contiuum exposure reduction with settings:\n {config.model_dump_json(indent=2)}")

    # Before we get into the processing, we want to update any external data sources we might need
    # In the future, this would be managed by Prefect automatically scheduling these updates
    # update_cfht_weather()

    assert config.resolver is not None, "Resolver should not be None at this point"

    # augment_science_file()
    preprocess_exposure(config.primary_file, config.resolver)
    # correct_dichoric()
    # remove_continuum()
    # calibrate_with_flats()


if __name__ == "__main__":
    continuum_file = Path(__file__).parents[2] / "data/raw/runs/run_id=25_057_001/continuum_red.fits"
    config = ContinuumReduction(primary_file=continuum_file)
    reduce_continuum_channel_exposure(config)
