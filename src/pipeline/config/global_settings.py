from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_path: Path = Field(default_factory=lambda: Path(__file__).parents[3] / "data")
    output_path: Path = Field(default_factory=lambda: Path(__file__).parents[3] / "output")
    plot: bool = Field(default=True, description="Make plots of the data")


settings = Settings()
