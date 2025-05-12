from pathlib import Path
from typing import Self

from pydantic import Field, FilePath, model_validator
from pydantic_settings import BaseSettings

from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_flow
from pipeline.resolver.resolver import Resolver
from pipeline.tasks import preprocess_exposure as preprocess_exposure_task
from pipeline.tasks.build_filestore import build_filestore


class PreprocessExposure(BaseSettings):
    primary_file: FilePath = Field(description="Location of the continuum exposure file. Relative to the data path.")
    use_cache: bool = Field(default=True, description="Use cached data when possible")
    make_plots: bool = Field(default=True, description="Make plots of the data")
    resolver: Resolver | None = Field(default=None, description="Resolver to use for file paths. Default None.")  # type: ignore

    @model_validator(mode="after")
    def check_resolver(self) -> Self:
        if self.resolver is None:
            self.resolver = build_filestore(refresh=not self.use_cache)
        assert self.resolver is not None, "Resolver should not be None at this point"

        self.resolver.get_file_metadata(self.primary_file)  # Check that this file exists
        return self


@pipeline_flow()
def preprocess_exposure(config: PreprocessExposure) -> None:
    logger = get_logger()
    logger.info(f"Starting preprocessing with settings:\n {config.model_dump_json(indent=2)}")
    assert config.resolver is not None, "Resolver should not be None at this point"
    preprocess_exposure_task(config.primary_file, config.resolver)


if __name__ == "__main__":
    continuum_file = Path(__file__).parents[2] / "data/raw/runs/run_id=25_057_001/continuum_blue.fits"
    config = PreprocessExposure(primary_file=continuum_file)
    preprocess_exposure(config)
