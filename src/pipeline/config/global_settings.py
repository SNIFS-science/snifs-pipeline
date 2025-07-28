from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_path: Path = Field(default_factory=lambda: Path(__file__).parents[3] / "data")
    output_path: Path = Field(default_factory=lambda: Path(__file__).parents[3] / "output")
    public_path: Path = Field(default_factory=lambda: Path(__file__).parents[3] / "output")
    public_path_replacement: str = Field(default="/public", description="Replacement for public path")
    plot: bool = Field(default=True, description="Make plots of the data")
    refresh: bool = Field(default=True, description="Refresh the filestore on startup")


settings = Settings()
