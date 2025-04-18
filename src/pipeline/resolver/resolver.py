from functools import cached_property
from pathlib import Path

import polars as pl
from pydantic import BaseModel, field_validator

from pipeline.config.global_settings import settings
from pipeline.resolver import file_match_registry
from pipeline.resolver.common import FileStoreDataFrame, FileStoreEntry, FileType, extract_file_details


class Resolver(BaseModel):
    # TODO: discuss how cloud focused this should be.
    # TODO: Ideally this resolve had both local pref and cloud fetching built in.
    file_store_path: Path

    data_path: Path
    output_path: Path

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
        return pl.read_parquet(self.file_store_path).pipe(FileStoreDataFrame)

    @cached_property
    def processed_data_path(self) -> Path:
        return self.data_path / "processed"

    @classmethod
    def create(cls, **kwargs) -> "Resolver":
        kwargs = {"data_path": settings.data_path, "output_path": settings.output_path}
        kwargs.update(kwargs)
        if "file_store_path" not in kwargs:
            kwargs["file_store_path"] = kwargs["data_path"] / "filestore.parquet"
        return cls(**kwargs)

    def ensure_file_exists(self, file_path: Path) -> None:
        file_store = self.file_store
        entry = extract_file_details(file_path, file_path.relative_to(self.data_path))

        # Get a hash of the dataframe as it exists now
        current_hash = hash(tuple(file_store.drop("time_added").hash_rows().to_list()))

        # add the entry to the dataframe and see if the hash has changed
        new_file_store = (
            pl.concat([file_store, entry], how="diagonal_relaxed", rechunk=True)
            .sort("file_path", "time_added")
            .unique("file_path", keep="last", maintain_order=True)
        )

        # Get a hash of the new dataframe
        new_hash = hash(tuple(file_store.drop("time_added").hash_rows().to_list()))

        # If the hash has changed, we need to update the file store
        if current_hash != new_hash:
            # Save the new file store
            self.save_filestore(FileStoreDataFrame(new_file_store))

    def save_filestore(self, df: FileStoreDataFrame) -> None:
        self.file_store_path.parent.mkdir(parents=True, exist_ok=True)
        df.sort("file_path").write_parquet(self.file_store_path)

    def get_file_metadata(self, file_path: Path) -> FileStoreEntry:
        """
        Get the metadata for a file.
        """
        if self.file_store is None:
            raise FileNotFoundError(f"File store not found at {self.file_store_path}.")
        relative_path = str(file_path.relative_to(self.data_path))
        file_records = self.file_store.filter(pl.col("file_path").eq(relative_path))
        if len(file_records) == 0:
            raise FileNotFoundError(f"File {relative_path} not found in file store.")
        assert len(file_records) == 1, f"Found multiple records for {relative_path}"
        return FileStoreEntry.model_validate(file_records.to_dicts()[0])

    def get_match(
        self,
        file_type: str | FileType,
        primary: FileStoreEntry | None,
    ) -> FileStoreEntry:
        """
        Get a single match for a file type.
        """
        if isinstance(file_type, FileType):
            file_type = file_type.value
        return file_match_registry.get_match(file_type, primary, self.file_store)

    def get_match_path(
        self,
        file_type: str | FileType,
        primary: FileStoreEntry | None = None,
    ) -> Path:
        return self.data_path / self.get_match(file_type, primary).file_path

    def get_matches(
        self,
        file_type: str | FileType,
        primary: FileStoreEntry | None = None,
    ) -> list[FileStoreEntry]:
        """
        Get all matches for a file type.
        """
        return file_match_registry.get_matches(file_type, primary, self.file_store)

    def get_match_paths(
        self,
        file_type: str | FileType,
        primary: FileStoreEntry | None = None,
    ) -> list[Path]:
        """
        Get all match paths for a file type.
        """
        return [self.data_path / match.file_path for match in self.get_matches(file_type, primary)]
