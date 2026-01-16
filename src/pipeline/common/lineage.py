import inspect
import os
from datetime import datetime as dt
from datetime import timezone as tz
from functools import lru_cache

import git
from pydantic import BaseModel, Field

from pipeline.common.log import get_logger


@lru_cache()
def get_git_repo() -> str:
    return os.environ.get("GIT_REPO", "https://github.com/SNIFS-science/snifs-pipeline")


@lru_cache()
def get_git_commit() -> str:
    """
    Get the current git commit hash.
    """
    if "GIT_COMMIT_HASH" in os.environ:
        return os.environ["GIT_COMMIT_HASH"]
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except Exception as e:
        logger = get_logger()
        logger.error(f"Could not get git commit: {e}")
        return "unknown"


class Lineage(BaseModel):
    time: dt = Field(default_factory=lambda: dt.now(tz.utc), description="Time of the processing step")
    title: str = Field(default="", description="Title of the step. For example: 'add_poisson_noise_to_variance'")
    summary: str = Field(default="", description="Should include numeric summary of the step.")
    repository: str = Field(default_factory=get_git_repo, description="Repository URL")
    git_commit: str = Field(default_factory=get_git_commit, description="Git commit hash of the repository")


class LineageMixin(BaseModel):
    lineage: list[Lineage] = Field(default_factory=list, description="Lineage of processing steps applied to the data")

    def add_function_lineage(self, summary: str) -> None:
        """
        Add a lineage step with the given summary and the name of the function that called this method.
        """
        logger = get_logger()
        frame = inspect.currentframe()
        assert frame is not None, "This method must be called from within a function"
        assert frame.f_back is not None, "This method must be called from within a function"
        function_name = frame.f_back.f_code.co_name
        self.lineage.append(Lineage(title=function_name, summary=summary))
        logger.info(summary)

    def add_simple_lineage(self, title: str, summary: str) -> None:
        self.lineage.append(Lineage(title=title, summary=summary))

    def add_lineage(self, lineage: Lineage) -> None:
        self.lineage.append(lineage)
