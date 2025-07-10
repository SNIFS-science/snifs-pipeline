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
