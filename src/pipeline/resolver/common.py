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
from pydantic import BaseModel, BeforeValidator, ValidationError, ValidationInfo

from pipeline.common.log import get_logger

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

    logger = get_logger()
    logger.info(f"Resolving file type {file_type} for value {value} with must_match={must_match}")
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
    PROCESSED = "2_PROCESSED"


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
    def OptionalPath(self) -> type[Path | None]:
        return Annotated[Path | None, BeforeValidator(partial(resolve_type, file_type=self, must_match=False))]  # type: ignore

    @property
    def Path(self) -> type[Path | None]:
        return Annotated[Path | None, BeforeValidator(partial(resolve_type, file_type=self))]  # type: ignore


class FileStoreModel(DataFrameModel):
    level: Series[str] = Field()
    file_path: Series[str] = Field(unique=True)
    file_name: Series[str] = Field()
    file_type: Series[FileType] = Field(coerce=True)
    object: Series[str] = Field(nullable=True, coerce=True)
    object_ra: Series[str] = Field(nullable=True, coerce=True)
    object_dec: Series[str] = Field(nullable=True, coerce=True)
    run_id: Series[str] = Field(nullable=True, coerce=True)
    observation_id: Series[str] = Field(nullable=True, coerce=True)
    time_added: Series[UTCDatetime] = Field(coerce=True)
    time_creation: Series[UTCDatetime] = Field(nullable=True, coerce=True)
    time_observation: Series[UTCDatetime] = Field(nullable=True, coerce=True)
    exposure_seconds: Series[float] = Field(nullable=True, coerce=True)
    dark_seconds: Series[float] = Field(nullable=True, coerce=True)
    altitude: Series[float] = Field(nullable=True, coerce=True)
    azimuth: Series[float] = Field(nullable=True, coerce=True)
    filter: Series[str] = Field(nullable=True, coerce=True)
    channel: Series[str] = Field(nullable=True, coerce=True)
    detector: Series[str] = Field(nullable=True, coerce=True)


FileStoreDataFrame = DataFrame[FileStoreModel]


class FileStoreEntry(BaseModel):
    level: str
    file_path: str
    file_name: str
    file_type: FileType
    time_added: dt
    object: str | None = None
    object_ra: str | None = None
    object_dec: str | None = None
    run_id: str | None = None
    observation_id: str | None = None
    time_creation: dt | None = None
    time_observation: dt | None = None
    exposure_seconds: float | None = None
    dark_seconds: float | None = None
    altitude: float | None = None
    azimuth: float | None = None
    filter: str | None = None
    channel: str | None = None
    detector: str | None = None


FITS_HEADER_MAP = {
    "file_type": "OBSTYPE",
    "run_id": "RUNID",
    "observation_id": "OBSID",
    "time_added": "TIME_ADDED",
    "time_creation": "UTC",
    "time_observation": "DATE-OBS",
    "exposure_seconds": "EXPTIME",
    "dark_seconds": "DARKTIME",
    "time_on_seconds": "TIMEON",
    "detector_temperature": "DETTEMP",
    "object_ra": "OBJRA",
    "object_dec": "OBJDEC",
    "bias_frame": "BIASFRAM",
}


# TODO TODO TODO: use the above to rename header values on load Image.from_fits
def extract_details_from_fits(path: Path) -> dict[str, str | int | float | dt]:
    values = {}
    with fits.open(path) as hdul:  # type: ignore
        # Assume headers are in the first HDU
        header = hdul[0].header  # type: ignore
        # Extract relevant information from the header
        for column in FileStoreModel.to_schema().columns:
            expected_column_name = FITS_HEADER_MAP.get(column, column)
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

    return values


def extract_details_from_asdf(path: Path) -> dict[str, str | int | float | dt]:
    values = {}
    import asdf
    from asdf import AsdfFile

    af: AsdfFile
    with asdf.open(path) as af:
        if "metadata" not in af:
            return values
        metadata = {k.lower(): v for k, v in af["metadata"].items()}

        for column in FileStoreModel.to_schema().columns:
            if column in metadata:
                value = metadata[column]
                if isinstance(value, str):
                    value = value.strip()
                if column.startswith("time"):
                    if isinstance(value, int):
                        value = dt.fromtimestamp(value, tz=tz.utc)
                    elif isinstance(value, str):
                        value = dt.strptime(value, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=tz.utc)
                values[column] = value
    return values


def extract_file_details(path: Path) -> FileStoreDataFrame | None:  # noqa: C901
    path = path.resolve()
    values = {
        "file_path": str(path),
        "file_name": path.name,
        "time_added": dt.now(tz=tz.utc),
    }
    if path.suffix == ".fits":
        values = extract_details_from_fits(path) | values
    elif path.suffix == ".asdf":
        values = extract_details_from_asdf(path) | values

    # Ensure all lowercase
    values = {k.lower(): v for k, v in values.items()}

    # Add any hive partitions in the path
    for directory in str(path).split("/"):
        if "=" in directory:
            key, value = directory.split("=")
            values[key] = value

    # Rename any "type" to "file_type" for now
    if "type" in values:
        values["file_type"] = values.pop("type")
    if "obstype" in values:
        values["file_type"] = values.pop("obstype")

    # Remove "unknown" as a null value
    for key in list(values.keys()):
        if values[key] == "unknown" or values[key] == "":
            del values[key]

    # There is some possibility to confuse high signal to noise continuum files
    # with low signal to noise raster readouts. In the existing quick_preprocess
    # scripts, this check is done by check to see if the file is less than 16MB in size
    # I'm not going to use 16MB because thats the rasters are <2MB and full flats
    # are 16-17MB anyway, so why the cutoff so close? I'll go for the midpoint of 8MB.
    if values.get("file_type") == FileType.CONTINUUM:
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb < 8:
            values["file_type"] = FileType.RASTER

    try:
        values = FileStoreEntry.model_validate(values).model_dump()
    except ValidationError as e:
        logger = get_logger()
        logger.error(f"Validation error: {e} for path {path}, skipping")
        return None

    return FileStoreDataFrame(values)
