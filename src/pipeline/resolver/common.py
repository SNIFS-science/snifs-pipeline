from datetime import datetime as dt
from datetime import timezone as tz
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Annotated, overload

import polars.selectors as cs
from astropy.io import fits
from pandera.engines.polars_engine import DateTime
from pandera.polars import DataFrameModel, Field
from pandera.typing.polars import DataFrame, Series
from pydantic import BaseModel, BeforeValidator, FilePath, ValidationInfo

UTCDatetime = Annotated[DateTime, False, "UTC", "ms"]
DATETIME_CONVERSION_EXPR = cs.datetime().dt.cast_time_unit("ms").dt.convert_time_zone("UTC")


@overload
def resolve_type(
    value: str | Path | None,
    info: ValidationInfo,
    file_type: "FileType",
    must_match: bool = True,
) -> Path: ...
@overload
def resolve_type(
    value: str | Path | None,
    info: ValidationInfo,
    file_type: "FileType",
    must_match: bool = False,
) -> Path | None: ...


def resolve_type(
    value: str | Path | None,
    info: ValidationInfo,
    file_type: "FileType",
    must_match: bool = True,
) -> Path | None:
    from pipeline.tasks.build_filestore import build_filestore

    resolver = build_filestore(refresh=info.data.get("refresh_filestore", False))
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    try:
        return resolver.get_match_path(file_type, info.data.get("primary_file"))
    except FileNotFoundError:
        if must_match:
            raise
        return None


class PipelineStage(StrEnum):
    RAW = "0_RAW"
    PREPROCESSED = "1_PREPROCESSED"


class FileType(StrEnum):
    SCIENCE = "OBJECT"
    CONTINUUM = "FLAT"
    RASTER = "RASTER"
    WEATHER = "WEATHER"
    ARC = "ARC"
    RAW_LOGS = "RAW_LOGS"
    CCD_ON_TIMES = "CCD_ON_TIMES"
    DICHROIC_REFERENCE = "DICHROIC_REFERENCE"
    BINARY_OFFSET_MODEL = "BINARY_OFFSET_MODEL"
    DARK_MODEL = "DARK_MODEL"
    DARK = "DARK"
    BIAS_MODEL = "BIAS_MODEL"
    BIAS = "BIAS"

    @property
    def Path(self) -> type[FilePath]:
        return Annotated[FilePath, BeforeValidator(partial(resolve_type, file_type=self))]  # type: ignore

    @property
    def OptionalPath(self) -> type[FilePath | None]:
        return Annotated[FilePath | None, BeforeValidator(partial(resolve_type, file_type=self, must_match=False))]  # type: ignore


class FileStoreModel(DataFrameModel):
    file_path: Series[str] = Field(unique=True)
    file_name: Series[str] = Field()
    type: Series[FileType] = Field(coerce=True)
    object: Series[str] = Field(nullable=True)
    object_ra: Series[str] = Field(nullable=True)
    object_dec: Series[str] = Field(nullable=True)
    run_id: Series[str] = Field(nullable=True)
    observation_id: Series[str] = Field(nullable=True)
    time_added: Series[UTCDatetime] = Field(coerce=True)
    time_creation: Series[UTCDatetime] = Field(nullable=True, coerce=True)
    time_observation: Series[UTCDatetime] = Field(nullable=True, coerce=True)
    exposure_seconds: Series[float] = Field(nullable=True)
    dark_seconds: Series[float] = Field(nullable=True)
    altitude: Series[float] = Field(nullable=True)
    azimuth: Series[float] = Field(nullable=True)
    cass_rotation_angle: Series[float] = Field(nullable=True)
    filter: Series[str] = Field(nullable=True)
    channel: Series[str] = Field(nullable=True)
    detector: Series[str] = Field(nullable=True)


FileStoreDataFrame = DataFrame[FileStoreModel]


class FileStoreEntry(BaseModel):
    file_path: str
    file_name: str
    type: FileType
    object: str | None
    object_ra: str | None
    object_dec: str | None
    run_id: str | None
    observation_id: str | None
    num_extensions: int | None
    num_data_extensions: int | None
    time_added: dt
    time_creation: dt | None
    time_observation: dt | None
    exposure_seconds: float | None
    dark_seconds: float | None
    altitude: float | None
    azimuth: float | None
    cass_rotation_angle: float | None
    filter: str | None
    channel: str | None
    detector: str | None


HEADER_MAP = {
    "type": "OBSTYPE",
    "run_id": "RUNID",
    "observation_id": "OBSID",
    "time_added": "TIME_ADDED",
    "time_creation": "UTC",
    "time_observation": "DATE-OBS",
    "exposure_seconds": "EXPTIME",
    "dark_seconds": "DARKTIME",
    "cass_rotation_angle": "ROTANGLE",
    "object_ra": "OBJRA",
    "object_dec": "OBJDEC",
}


def extra_details_from_fits(path: Path) -> dict[str, str | int | float | dt]:
    values = {}
    with fits.open(path) as hdul:  # type: ignore
        # Assume headers are in the first HDU
        header = hdul[0].header  # type: ignore
        # Extract relevant information from the header
        for column in FileStoreModel.to_schema().columns:
            expected_column_name = HEADER_MAP.get(column, column)
            if expected_column_name in header:
                value = header[expected_column_name]
                if isinstance(value, str):
                    value = value.strip()
                if column.startswith("time"):
                    if isinstance(value, int):
                        value = dt.fromtimestamp(value, tz=tz.utc)
                    elif isinstance(value, str):
                        value = dt.strptime(value, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=tz.utc)
                values[column] = value

        values["num_extensions"] = len(hdul)
        values["num_data_extensions"] = len([x for x in hdul if x.data is not None])  # type: ignore
    return values


def extract_file_details(path: Path, relative_path: Path) -> FileStoreDataFrame:
    values = {
        "file_path": str(relative_path),
        "file_name": path.name,
        "time_added": dt.now(tz=tz.utc),
    }
    if path.suffix == ".fits":
        values = extra_details_from_fits(path) | values

    # Add any hive partitions in the path
    for directory in str(relative_path).split("/"):
        if "=" in directory:
            key, value = directory.split("=")
            values[key] = value

    # There is some possibility to confuse high signal to noise continuum files
    # with low signal to noise raster readouts. In the existing quick_preprocess
    # scripts, this check is done by check to see if the file is less than 16MB in size
    # I'm not going to use 16MB because thats the rasters are <2MB and full flats
    # are 16-17MB anyway, so why the cutoff so close? I'll go for the midpoint of 8MB.
    if values.get("TYPE") == FileType.CONTINUUM:
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb < 8:
            values["type"] = FileType.RASTER

    return FileStoreDataFrame(values)
