from functools import cached_property
from pathlib import Path

import polars as pl
import prefect
from loguru import logger
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from pipeline.common.log import get_logger
from pipeline.config.global_settings import settings
from pipeline.resolver.common import FileStoreDataFrame, FileStoreEntry, FileType, extract_file_details
from pipeline.resolver.registry import file_match_registry


class Resolver(BaseModel):
    # TODO: discuss how cloud focused this should be.
    # TODO: Ideally this resolve had both local pref and cloud fetching built in.
    file_store_path: Path
    database_path: Path

    data_path: Path
    output_path: Path
    public_path: Path

    @field_validator("file_store_path")
    @classmethod
    def check_file_store_path(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            return Path(v)
        return v

    def file_store_exists(self) -> bool:
        return self.file_store_path.exists()

    @cached_property
    def file_store(self) -> FileStoreDataFrame:
        assert self.file_store_path.exists(), (
            f"File store not found at {self.file_store_path}. Please build it via `build_filestore`"
        )
        logger.info(f"Loading filestore from {self.file_store_path}")
        return pl.read_parquet(self.file_store_path).pipe(FileStoreDataFrame)

    @cached_property
    def processed_data_path(self) -> Path:
        return self.output_path / "level=processed"

    @classmethod
    def create(cls, **kwargs) -> "Resolver":
        kwargs = {
            "data_path": settings.data_path,
            "output_path": settings.output_path,
            "public_path": settings.public_path,
            **kwargs,
        }
        kwargs.update(kwargs)
        if "file_store_path" not in kwargs:
            kwargs["file_store_path"] = kwargs["output_path"] / "filestore.parquet"
        if "database_path" not in kwargs:
            kwargs["database_path"] = kwargs["output_path"] / "database.sqlite"
        return cls(**kwargs)

    def ensure_file_exists(self, file_path: Path | str) -> None:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        file_store = self.file_store
        entry = extract_file_details(file_path)
        if entry is None:
            raise ValueError(f"Could not extract file details from {file_path}")

        # Get a hash of the dataframe as it exists now
        current_hash = hash(tuple(file_store.drop("time_added").hash_rows().to_list()))

        # add the entry to the dataframe and see if the hash has changed
        new_file_store = (
            pl.concat([file_store, entry], how="diagonal_relaxed", rechunk=True)
            .sort("file_path", "time_added")
            .unique("file_path", keep="last", maintain_order=True)
        )

        # Get a hash of the new dataframe
        new_hash = hash(tuple(new_file_store.drop("time_added").hash_rows().to_list()))

        # If the hash has changed, we need to update the file store
        if current_hash != new_hash:
            self.file_store = FileStoreDataFrame(new_file_store)
            self.save_filestore(self.file_store)

    def save_filestore(self, df: FileStoreDataFrame) -> None:
        self.file_store_path.parent.mkdir(parents=True, exist_ok=True)
        df.sort("file_path").write_parquet(self.file_store_path)

    def get_file_metadata(self, file_path: Path | str) -> FileStoreEntry:
        """Get the metadata for a file.

        Raises:
            FileNotFoundError: If the file is not found in the file store.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if self.file_store is None:
            raise FileNotFoundError(f"File store not found at {self.file_store_path}.")
        path = str(file_path.resolve())
        file_records = self.file_store.filter(pl.col("file_path").eq(path))
        if len(file_records) == 0:
            raise FileNotFoundError(f"File {path} not found in file store.")
        assert len(file_records) == 1, f"Found multiple records for {path}"
        return FileStoreEntry.model_validate(file_records.to_dicts()[0])

    def get_full_path(self, file_path: Path) -> Path:
        return self.data_path / file_path

    def get_match(
        self,
        file_type: str | FileType,
        primary: FileStoreEntry | str | Path | None,
    ) -> FileStoreEntry:
        """Get a single match for a file type."""
        if isinstance(file_type, FileType):
            file_type = file_type.value
        if isinstance(primary, str):
            primary = Path(primary)
        if isinstance(primary, Path):
            primary = self.get_file_metadata(primary)
        return file_match_registry.get_match(file_type, primary, self.file_store)

    def get_match_path(
        self,
        file_type: str | FileType,
        primary: FileStoreEntry | str | Path | None = None,
    ) -> Path:
        return self.data_path / self.get_match(file_type, primary).file_path

    def get_matches(
        self,
        file_type: str | FileType,
        primary: FileStoreEntry | str | Path | None = None,
    ) -> list[FileStoreEntry]:
        """Get all matches for a file type."""
        if isinstance(file_type, FileType):
            file_type = file_type.value
        if isinstance(primary, str):
            primary = Path(primary)
        if isinstance(primary, Path):
            primary = self.get_file_metadata(primary)
        return file_match_registry.get_matches(file_type, primary, self.file_store)

    def get_match_paths(
        self,
        file_type: str | FileType,
        primary: FileStoreEntry | str | Path | None = None,
    ) -> list[Path]:
        """Get all match paths for a file type."""
        return [self.data_path / match.file_path for match in self.get_matches(file_type, primary)]


def get_run_id() -> str:
    try:
        return prefect.context.FlowRunContext.get().flow_run.id
    except Exception:
        return "unknown"


class FlowConfig(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)

    @cached_property
    def resolver(self) -> Resolver:
        from pipeline.tasks.build_filestore import build_filestore

        return build_filestore(refresh=getattr(self, "refresh_filestore", False))

    def fetch_metadata(self, path: Path) -> FileStoreEntry:
        return self.resolver.get_file_metadata(path)

    @cached_property
    def output_folder(self) -> Path:
        raise NotImplementedError("This method should be overridden in subclasses to provide the output folder path.")

    @cached_property
    def public_folder(self) -> Path:
        raise NotImplementedError("This method should be overridden in subclasses to provide the public folder path.")

    def propagate_output_path(self) -> None:
        OUTPUT_PATH_MAP[get_run_id()] = self.output_folder

    def propagate_public_path(self) -> None:
        PUBLIC_PATH_MAP[get_run_id()] = self.public_folder

    def initialise_and_log(self) -> None:
        self.resolver  # noqa: B018
        self.propagate_output_path()
        self.propagate_public_path()
        get_logger().info(f"Config initialised:\n{self.model_dump_json(indent=2)}\n")


# A map from flow_id to the output path for explicit access
OUTPUT_PATH_MAP: dict[str, Path] = {}
PUBLIC_PATH_MAP: dict[str, Path] = {}
