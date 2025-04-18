import pandera as pa
import polars as pl
from pandera.polars import DataFrameSchema


@pa.check_types
def validate_df_schema(df: pl.DataFrame, schema: type[DataFrameSchema]) -> pl.DataFrame:
    return schema.validate(df)  # type: ignore
