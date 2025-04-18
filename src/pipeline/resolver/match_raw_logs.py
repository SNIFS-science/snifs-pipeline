import polars as pl

from pipeline.resolver.common import FileStoreDataFrame, FileStoreEntry, FileType
from pipeline.resolver.registry import file_match_registry


@file_match_registry.register(FileType.RAW_LOGS)
def find_raw_log_files(science_file: FileStoreEntry | None, file_store: FileStoreDataFrame) -> list[FileStoreEntry]:
    """
    Finds the raw logs file. Does not care about what the science file is right now.
    """
    files = file_store.filter((pl.col("type").eq(FileType.RAW_LOGS.value)))
    return [FileStoreEntry.model_validate(row) for row in files.to_dicts()]
