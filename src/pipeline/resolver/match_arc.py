import polars as pl

from pipeline.resolver.common import FileStoreDataFrame, FileStoreEntry, FileType
from pipeline.resolver.registry import file_match_registry


@file_match_registry.register(FileType.ARC)
def find_arc_files(science_file: FileStoreEntry | None, file_store: FileStoreDataFrame) -> list[FileStoreEntry]:
    """
    Find the arc file for a given science file.
    """
    assert science_file is not None, "science_file must be provided. There is no global suitable ARC file."
    # Try to match on the run_id
    files = file_store.filter(
        (pl.col("type").eq(FileType.ARC.value))
        & (pl.col("run_id").eq(science_file.run_id))
        & (pl.col("object").eq(science_file.object))
        & (pl.col("channel").eq(science_file.channel))
    )
    return [FileStoreEntry.model_validate(row) for row in files.to_dicts()]
