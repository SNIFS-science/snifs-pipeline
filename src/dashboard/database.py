import sqlite3
from pathlib import Path

import polars as pl


class SyncDatabase:
    def __init__(self, db_url: str | Path):
        self.db_url = Path(db_url) if isinstance(db_url, str) else db_url

    def get_discriminators(self) -> list[str]:
        """
        Get all unique discriminators from the database.
        """
        with sqlite3.connect(self.db_url) as conn:
            query = "SELECT DISTINCT discriminator FROM summaries"
            return [row[0] for row in conn.execute(query).fetchall()]

    def get_summaries(self, discriminator: str | None = None) -> pl.DataFrame:
        """
        Get all runs from the database.
        """
        with sqlite3.connect(self.db_url) as conn:
            query = "SELECT * FROM summaries"
            if discriminator:
                query += f" WHERE discriminator = '{discriminator}'"
            return pl.read_database(query, conn)

    def get_summary_datas(self, discriminator: str | None = None) -> pl.DataFrame:
        """
        Get summary data for plotting.
        """
        with sqlite3.connect(self.db_url) as conn:
            query = """
                SELECT flow_run_id, discriminator, created_at, key, value_str, value_num
                FROM summaries
                JOIN summary_info ON summaries.task_run_id = summary_info.task_run_id
            """
            if discriminator:
                query += f" WHERE discriminator = '{discriminator}'"
            df = pl.read_database(query, conn)

            def sanitise(col: str) -> str:
                if col.startswith("value_str_"):
                    return col.removeprefix("value_str_")
                elif col.startswith("value_num_"):
                    return col.removeprefix("value_num_") + "_num"
                return col

            df_wide = df.pivot(
                on="key", index=["flow_run_id", "discriminator", "created_at"], values=["value_str", "value_num"]
            )
            df_wide = df_wide.rename({col: sanitise(col) for col in df_wide.columns})
            # Drop all null columns
            df_wide = df_wide.select([s.name for s in df_wide if not (s.null_count() == df_wide.height)])
            # Add a link column if we can
            if "api_url" in df_wide.columns and "flow_run_id" in df_wide.columns:
                df_wide = df_wide.with_columns(
                    link=pl.concat_str(
                        [pl.col("api_url").str.replace("/api/", ""), pl.col("flow_run_id")], separator="/runs/flow-run/"
                    )
                )
            else:
                df_wide = df_wide.with_columns(link=pl.lit(None))

            if "name" not in df_wide.columns:
                df_wide = df_wide.with_columns(name=pl.col("run_id") + "_" + pl.col("channel"))
            return df_wide
