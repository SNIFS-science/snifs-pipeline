import polars as pl

from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_task
from pipeline.resolver.common import FileStoreDataFrame, extract_file_details
from pipeline.resolver.resolver import Resolver


@pipeline_task()
def build_filestore(refresh: bool = False) -> Resolver:
    resolver = Resolver.create()

    logger = get_logger()
    file_store = resolver.file_store if resolver.file_store_exists() else None
    analysed_files = [] if file_store is None else file_store["file_path"].to_list()

    dfs = [] if file_store is None or refresh else [file_store]

    # If files are deleted, we want to reflect this as well.
    detected_filepaths = []
    all_files = list(resolver.data_path.rglob("*/**/*"))
    logger.info(f"Found {len(all_files)} files in the data path. Rebuilding the filestore.")
    for file in all_files:
        if file.is_dir():
            continue
        if file.suffix == ".md":
            continue
        relative_path = file.relative_to(resolver.data_path)
        detected_filepaths.append(str(relative_path))
        if refresh or relative_path not in analysed_files:
            dfs.append(extract_file_details(file, relative_path))

    df = (
        pl.concat(dfs, how="diagonal_relaxed", rechunk=True)
        .sort("time_added")
        .unique(subset=["file_path"], keep="last", maintain_order=True)
        .filter(pl.col("file_path").is_in(detected_filepaths))
        .drop_nulls(subset=["type"])
        .pipe(FileStoreDataFrame)
    )
    logger.info(f"Writing filestore with shape {df.shape} to {resolver.file_store_path}")
    resolver.save_filestore(df)
    # Validate the file store exists and can be loaded
    _ = resolver.file_store
    return resolver


if __name__ == "__main__":
    build_filestore(refresh=True)
