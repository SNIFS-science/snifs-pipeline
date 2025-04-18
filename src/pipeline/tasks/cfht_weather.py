from datetime import datetime as dt
from datetime import timedelta as td
from datetime import timezone as tz
from pathlib import Path

import pandera as pa
import polars as pl
from pandera.polars import DataFrameModel
from pandera.typing.polars import DataFrame, Series

from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_task
from pipeline.resolver.common import UTCDatetime
from pipeline.resolver.resolver import Resolver


class Weather(DataFrameModel):
    time: Series[UTCDatetime] = pa.Field(unique=True)
    wind_speed: Series[int] = pa.Field(description="Wind speed in knots")
    wind_direction: Series[int] = pa.Field(description="Wind direction in degrees")
    temperature: Series[float] = pa.Field(description="Temperature in degrees Celsius")
    relative_humidity: Series[int] = pa.Field(description="Relative humidity in percent")
    pressure: Series[float] = pa.Field(nullable=True, description="Pressure in millibars")


WeatherDataFrame = DataFrame[Weather]


def should_fetch_data(path: Path) -> bool:
    if not path.exists():
        return True
    expect_datapoint_after = dt.now(tz=tz.utc) - td(minutes=5)
    df = pl.read_parquet(path).filter(pl.col("time") > expect_datapoint_after)
    if df.is_empty():
        return True
    return False


def plot_weather_data(
    df: WeatherDataFrame,
    plot_output_directory: Path,
    plot_lookback_days: int = 30,
) -> None:
    import matplotlib.pyplot as plt

    plot_output_directory.mkdir(parents=True, exist_ok=True)

    now = dt.now(tz=tz.utc)
    lookback_time = now - td(days=plot_lookback_days)
    df2 = df.filter(pl.col("time").is_between(lookback_time, now))

    cols_to_plot = ["temperature", "wind_speed", "wind_direction", "relative_humidity", "pressure"]
    fig, ax = plt.subplots(len(cols_to_plot), 1, figsize=(10, 8), sharex=True)
    for i, col in enumerate(cols_to_plot):
        df_sub = df2.select("time", col).drop_nulls()
        ax[i].plot(df_sub["time"], df_sub[col], label=col)
        ax[i].set_ylabel(col)
    ax[-1].set_xlabel("Time")
    ax[0].set_title("Weather Data from CFHT")

    logger = get_logger()
    output_path = plot_output_directory / "cfht_weather.png"
    logger.info(f"Plotting a month's worth of data to {output_path}")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


@pipeline_task()
def update_cfht_weather(
    lookback_time_days: int = 7,
    refresh: bool = False,
    make_plots: bool = False,
) -> None:
    resolver = Resolver.create()
    expected_location = resolver.data_path / "misc/type=WEATHER/station=CFHT/weather.parquet"
    logger = get_logger()
    if should_fetch_data(expected_location):
        logger.info("Fetching CFHT weather data")
        now = dt.now(tz=tz.utc)
        lookback_time = now - td(days=lookback_time_days)
        years_to_fetch = {lookback_time.year, now.year}
        dfs = []
        if not refresh and expected_location.exists():
            dfs.append(pl.read_parquet(expected_location))

        for year in years_to_fetch:
            url = f"http://mkwc.ifa.hawaii.edu/archive/wx/cfht/cfht-wx.{year}.dat"
            logger.info(f"Fetching weather data from {url}")
            df = (
                pl.read_csv(
                    url,
                    has_header=False,
                    separator=" ",
                    new_columns=[
                        "year",
                        "month",
                        "day",
                        "hour",
                        "minute",
                        "wind_speed",
                        "wind_direction",
                        "temperature",
                        "relative_humidity",
                        "pressure",
                    ],
                )
                .with_columns(
                    pl.datetime(
                        year=pl.col("year"),
                        month=pl.col("month"),
                        day=pl.col("day"),
                        hour=pl.col("hour"),
                        minute=pl.col("minute"),
                        time_unit="ms",
                        time_zone="HST",
                    )
                    .dt.convert_time_zone("UTC")
                    .alias("time")
                )
                .drop("year", "month", "day", "hour", "minute", pl.selectors.starts_with("column"))
                .select("time", pl.all().exclude("time"))
            )
            dfs.append(df)
        df = WeatherDataFrame(
            pl.concat(dfs, how="diagonal_relaxed").sort("time").unique("time", keep="last", maintain_order=True)
        )
        expected_location.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(expected_location)
    else:
        logger.info(f"Weather data already exists at {expected_location} and is up to date. Not fetching.")

    if make_plots:
        plot_weather_data(WeatherDataFrame(pl.read_parquet(expected_location)), resolver.output_path)


if __name__ == "__main__":
    update_cfht_weather.fn(
        lookback_time_days=60,
        refresh=True,
        make_plots=True,
    )
