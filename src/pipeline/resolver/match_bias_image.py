import polars as pl

from pipeline.resolver.common import FileStoreDataFrame, FileStoreEntry, FileType
from pipeline.resolver.registry import file_match_registry


@file_match_registry.register(FileType.BIAS)
def find_bias_image_files(primary_file: FileStoreEntry | None, file_store: FileStoreDataFrame) -> list[FileStoreEntry]:
    assert primary_file is not None, "primary_file must be provided. There is no global suitable bias image file."
    # Try to match on the run_id
    files = file_store.filter(
        (pl.col("file_type").eq(FileType.BIAS.value)) & (pl.col("channel").eq(primary_file.channel))
    )
    return [FileStoreEntry.model_validate(row) for row in files.to_dicts()]
