from pathlib import Path
from typing import Self

from pydantic import Field, FilePath, model_validator
from pydantic_settings import BaseSettings

from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_flow
from pipeline.resolver.common import FileType
from pipeline.tasks import preprocess_exposure as preprocess_exposure_fn
from pipeline.tasks.build_filestore import build_filestore


class PreprocessExposure(BaseSettings):
    primary_file: FilePath = Field(description="Location of the continuum exposure file. Relative to the data path.")
    refresh: bool = Field(default=True, description="Refresh means do it from scratch instead of using cached data.")
    bias_image_file: FilePath | None = Field(
        default=None, description="Location of the bias image. Relative to the data path."
    )
    bias_model_file: FilePath | None = Field(
        default=None, description="Location of the bias model. Relative to the data path."
    )
    prefer_bias_image_over_model: bool = Field(
        default=True,
        description="If both bias image and bias model are provided, prefer the image over the model.",
    )
    dark_image_file: FilePath | None = Field(
        default=None, description="Location of the dark image. Relative to the data path."
    )
    dark_model_file: FilePath | None = Field(
        default=None, description="Location of the dark model. Relative to the data path."
    )
    prefer_dark_image_over_model: bool = Field(
        default=True,
        description="If both dark image and dark model are provided, prefer the image over the model.",
    )
    binary_offset_model_file: FilePath | None = Field(
        default=None, description="Location of the binary offset file. Relative to the data path."
    )

    @model_validator(mode="after")
    def find_files(self) -> Self:
        resolver = build_filestore(self.refresh)

        primary = resolver.get_file_metadata(self.primary_file)  # type: ignore # Check that this file exists

        if self.bias_image_file is None:
            self.bias_image_file = resolver.get_match_path(FileType.BIAS_IMAGE, primary)
        if self.bias_model_file is None:
            self.bias_model_file = resolver.get_match_path(FileType.BIAS_MODEL, primary)
        if self.dark_image_file is None:
            self.dark_image_file = resolver.get_match_path(FileType.DARK_IMAGE, primary)
        if self.dark_model_file is None:
            self.dark_model_file = resolver.get_match_path(FileType.DARK_MODEL, primary)
        if self.binary_offset_model_file is None:
            self.binary_offset_model_file = resolver.get_match_path(FileType.BINARY_OFFSET_MODEL, primary)
        return self


@pipeline_flow()
def preprocess_exposure(config: PreprocessExposure) -> None:
    logger = get_logger()
    logger.info(f"Starting preprocessing with settings:\n {config.model_dump_json(indent=2)}")

    preprocess_exposure_fn(
        config.primary_file,
        config.bias_image_file,  # type: ignore
        config.bias_model_file,  # type: ignore
        config.prefer_bias_image_over_model,
        config.dark_image_file,  # type: ignore
        config.dark_model_file,  # type: ignore
        config.prefer_dark_image_over_model,
        config.binary_offset_model_file,  # type: ignore
    )


if __name__ == "__main__":
    continuum_file = Path(__file__).parents[2] / "data/raw/runs/run_id=25_057_001/continuum_blue.fits"
    config = PreprocessExposure(primary_file=continuum_file, refresh=False)
    preprocess_exposure(config)
