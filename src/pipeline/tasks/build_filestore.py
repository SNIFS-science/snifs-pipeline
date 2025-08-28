import sqlite3
from functools import lru_cache
from pathlib import Path

import polars as pl

from pipeline.common.log import get_logger
from pipeline.resolver.common import FileStoreDataFrame, extract_file_details
from pipeline.resolver.resolver import Resolver


def refresh_filestore(resolver: Resolver, refresh: bool = False) -> None:
    logger = get_logger()
    file_store = resolver.file_store if resolver.file_store_exists() else None
    analysed_files = [] if file_store is None else file_store["file_path"].to_list()

    dfs = [] if file_store is None or refresh else [file_store]

    # If files are deleted, we want to reflect this as well.
    detected_filepaths = []
    logger.info(f"Starting to build the filestore at {resolver.file_store_path}")
    all_files = list(resolver.data_path.rglob("**/*")) + list(resolver.output_path.rglob("**/*"))
    all_files = [
        a
        for a in all_files
        if not a.is_dir() and a.suffix in [".json", ".csv", ".fits", ".asdf", ".yml", ".yaml", ".logs.gz", ".parquet"]
    ]
    logger.info(f"Found {len(all_files)} files in the data path {resolver.data_path}. Rebuilding the filestore.")
    for file in all_files:
        path = file.resolve()
        detected_filepaths.append(str(path))
        if refresh or path not in analysed_files:
            logger.info(f"Found new file: {path}")
            res = extract_file_details(file)
            if res is not None:
                dfs.append(res)

    to_remove = set()
    for file in analysed_files:
        if not Path(file).exists():
            to_remove.add(file)

    df = (
        pl.concat(dfs, how="diagonal_relaxed", rechunk=True)
        .sort("time_added")
        .unique(subset=["file_path"], keep="last", maintain_order=True)
        .filter(pl.col("file_path").is_in(detected_filepaths))
        .drop_nulls(subset=["file_type"])
        .filter(~pl.col("file_path").is_in(to_remove))
        .pipe(FileStoreDataFrame)
    )
    logger.info(f"Writing filestore with shape {df.shape} to {resolver.file_store_path}")
    resolver.save_filestore(df)
    # Validate the file store exists and can be loaded
    _ = resolver.file_store


def build_database(resolver: Resolver) -> None:
    logger = get_logger()
    logger.info(f"Building database at {resolver.database_path}")

    if not resolver.database_path.exists():
        resolver.database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(resolver.database_path) as conn:
        cursor = conn.cursor()
        # Create tables and indices as needed
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                task_run_id TEXT PRIMARY KEY,
                flow_run_id TEXT NOT NULL,
                discriminator TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summary_info (
                task_run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value_str TEXT NOT NULL,
                value_num REAL,
                PRIMARY KEY (task_run_id, key),
                FOREIGN KEY (task_run_id) REFERENCES summaries(task_run_id)
            );
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_summaries_discriminator ON summaries(discriminator);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_summary_info_key ON summary_info(key);
        """)
        conn.commit()


@lru_cache
def build_filestore(refresh: bool = False) -> Resolver:
    resolver = Resolver.create()
    refresh_filestore(resolver, refresh)
    build_database(resolver)
    return resolver


if __name__ == "__main__":
    build_filestore(refresh=True)
