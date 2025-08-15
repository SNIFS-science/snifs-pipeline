from datetime import datetime as dt
from datetime import timezone as tz
from pathlib import Path

import polars as pl

from pipeline.common import Headers, get_logger, pipeline_task


@pipeline_task()
def determine_timeon(timeon_file: Path, primary_headers: Headers) -> str:
    channel = primary_headers.get_str("channel")
    date_obs = dt.fromisoformat(primary_headers.get_str("time_observation")).replace(tzinfo=tz.utc)
    """For historical reasons, this is a float in number of seconds since the detector was on... but as a string."""
    seconds_since_on = (
        pl.read_parquet(timeon_file)
        .filter(pl.col("channel").eq(channel))
        .filter(pl.col("time") <= date_obs)
        .sort("time", descending=True)
        .head(1)
        .with_columns(seconds_since_on=(date_obs - pl.col("time")).dt.total_seconds().cast(pl.Float64))[
            "seconds_since_on"
        ]
        .to_list()[0]
    )
    get_logger().info(
        f"Determined time_on_seconds value {seconds_since_on} seconds for channel {channel} "
        f"from file {timeon_file.name} at date {date_obs.isoformat()}"
    )
    # if seconds_since_on > 24 * 3600:
    #     raise ValueError(
    #         f"Timeon value {seconds_since_on} is greater than 24 hours, which is unexpected."
    #         "You may need to get the latest logs."
    #     )
    return str(seconds_since_on)
