from datetime import datetime as dt
from pathlib import Path

from pydantic import Field, FilePath
from pydantic_settings import BaseSettings

from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_flow
from pipeline.resolver.resolver import Resolver
from pipeline.tasks import (
    augment_science_file,
    calibrate_with_flats,
    correct_dichoric,
    preprocess_exposure,
    remove_continuum,
)
from pipeline.tasks.build_filestore import build_filestore
from pipeline.tasks.cfht_weather import update_cfht_weather


# TODO: I dont like any of these being FilePaths, but figured we'll start here
class ChannelReduction(BaseSettings):
    science_file: FilePath = Field(
        description="Location of the science file. This is actual observation. Relative to the data path."
    )

    arc_file: FilePath = Field(
        default=None,
        description="Location of the arc file(s). For a single exposure, the arc is usually taken "
        "immediately after the science exposure."
        "For two exposures, the arc is usually in the middle.",
    )  # type: ignore

    weather_file: FilePath = Field(default=None, description="Location of the weather file")  # type: ignore

    continuum_files: list[FilePath] = Field(
        default=[],
        description="Location of the continuum file. "
        "These are lamp exposures used to monitor shifts versus wavelength in the dichroic throughput."
        "For example, humidity changes the thickness of the multilayer interface coating"
        "Sometimes these images are referred to as 'raster' images or 'flats'. "
        "We normally have multiple continuum files per night. "
        "Five at the start, one in the middle of the night, and five more in the morning. "
        "These are often also called flats. "
        "And an image from a central CCD is often called a 'raster'.",
    )

    detector_last_on_time: dt | None = Field(
        default=None,
        description="The time the detector was last on (ie when the moment it was last turned off)."
        "This is needed because the amount of time the instrument has been idle impacts readings.",
    )

    use_cache: bool = Field(default=True, description="Use cached data when possible")

    make_plots: bool = Field(default=True, description="Make plots of the data")

    def resolve_missing(self, resolver: Resolver) -> None:
        """
        Resolves the paths for the channel reduction config.
        This is a bit of a hack, but it works for now.
        """
        logger = get_logger()
        logger.info("Resolving missing paths for channel reduction config")
        primary = resolver.get_file_metadata(self.science_file)
        if self.arc_file is None:
            self.arc_file = resolver.get_match_path("ARC", primary)
        if not self.continuum_files:
            self.continuum_files = resolver.get_match_paths("FLAT", primary)
        if self.weather_file is None:
            self.weather_file = resolver.get_match_path("WEATHER", primary)

        logger.info(f"Final config:\n {self.model_dump_json(indent=2)}")


@pipeline_flow()
def reduce_star_channel_exposure(config: ChannelReduction) -> None:
    logger = get_logger()
    logger.info(f"Starting channel exposure reduction with settings:\n {config.model_dump_json(indent=2)}")

    # Synchronise with any external data sources which may have changed
    update_cfht_weather()

    # Load in the existing file store and ensure its up to date
    resolver = build_filestore()

    # Ensure that our config is fully specified using the resolver
    config.resolve_missing(resolver)

    # And now we can run the reduction
    augment_science_file()
    preprocess_exposure(config, resolver)
    correct_dichoric()
    remove_continuum()
    calibrate_with_flats()


if __name__ == "__main__":
    # TODO: allow string and relative dir validation
    science_file = Path(__file__).parents[4] / "data/runs/run_id=25_056_084/science_blue.fits"
    config = ChannelReduction(science_file=science_file)
    reduce_star_channel_exposure(config)
