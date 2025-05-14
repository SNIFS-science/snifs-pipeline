import polars as pl

from pipeline.resolver.common import FileStoreDataFrame, FileStoreEntry, FileType
from pipeline.resolver.registry import file_match_registry


@file_match_registry.register(FileType.BINARY_OFFSET_MODEL)
def find_bom_files(primary_file: FileStoreEntry | None, file_store: FileStoreDataFrame) -> list[FileStoreEntry]:
    """
    Find the arc file for a given science file.
    """
    assert primary_file is not None, "science_file must be provided. There is no global suitable ARC file."
    # Try to match on the run_id
    files = file_store.filter(
        (pl.col("type").eq(FileType.BINARY_OFFSET_MODEL.value)) & (pl.col("channel").eq(primary_file.channel))
    )
    return [FileStoreEntry.model_validate(row) for row in files.to_dicts()]
