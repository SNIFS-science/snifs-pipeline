import asyncio
import json
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar
from uuid import UUID

from prefect import Flow, flow, get_client, task
from prefect.artifacts import create_image_artifact as _create_image_artifact
from prefect.artifacts import create_markdown_artifact as _create_markdown_artifact
from prefect.client.schemas.filters import ArtifactFilter, ArtifactFilterFlowRunId, ArtifactFilterKey
from prefect.deployments import run_deployment as prefect_run_deployment
from prefect.exceptions import PrefectHTTPStatusError
from pydantic_settings import BaseSettings

from pipeline.common.log import get_logger


class Settings(BaseSettings):
    prefect_enabled: bool = False


settings = Settings()


P = ParamSpec("P")
R = TypeVar("R")


TASK_DEFAULT_KWARGS = {
    "retries": 0,
    "retry_delay_seconds": 10,
    "log_prints": False,
    "timeout_seconds": 3600,  # An hour timeout per task
    "cache_result_in_memory": False,
}


FLOW_DEFAULT_KWARGS = {
    "timeout_seconds": 3600 * 24 * 7,  # A week timeout per flow
    "log_prints": False,
    "cache_result_in_memory": True,
}


@wraps(_create_image_artifact)
def create_image_artifact(*args, **kwargs) -> UUID | None:
    if settings.prefect_enabled:
        return _create_image_artifact(*args, **kwargs)  # type: ignore
    return None


@wraps(_create_markdown_artifact)
def create_markdown_artifact(*args, **kwargs) -> UUID | None:
    if settings.prefect_enabled:
        return _create_markdown_artifact(*args, **kwargs)  # type: ignore
    return None


async def run_deployment(
    flow_name: str,
    deployment_name: str,
    flow_run_name: str | None = None,
    parameters: dict | None = None,
    timeout: int | None = None,
    poll_interval: int = 60,
) -> dict | None:
    flow_run = await prefect_run_deployment(
        f"{flow_name}/{deployment_name}",
        flow_run_name=flow_run_name,
        parameters=parameters,
        timeout=0,
    )  # type: ignore

    flow_run_id: UUID = flow_run.id
    async with get_client() as client:
        async with asyncio.timeout(timeout):
            while True:
                await asyncio.sleep(poll_interval)
                try:
                    flow_run = await client.read_flow_run(flow_run_id)
                    flow_state = flow_run.state
                    if flow_state and flow_state.is_final():
                        # Check for the "result" keyed markdown artifact
                        artifacts = await client.read_artifacts(
                            artifact_filter=ArtifactFilter(
                                flow_run_id=ArtifactFilterFlowRunId(any_=[flow_run_id]),
                                key=ArtifactFilterKey(any_=["result"]),
                            )
                        )

                        if not artifacts:
                            raise ValueError(f"No result artifact found for flow run id {flow_run_id}")

                        elif len(artifacts) > 1:
                            raise ValueError(f"Multiple result artifacts found for flow run id {flow_run_id}")

                        else:
                            logger = get_logger()
                            logger.info(f"Returning result artifact for flow run id {flow_run_id}")
                            logger.info(f"Artifact data is: {artifacts[0].data}")
                            return json.loads(
                                str(artifacts[0].data).replace("```json", "").replace("```", "").replace("\n", "")
                            )
                except PrefectHTTPStatusError:
                    pass
    return None


def pipeline_task(**kwargs):
    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        if settings.prefect_enabled:
            final_kwargs = {**TASK_DEFAULT_KWARGS, **kwargs}
            return task(**final_kwargs)(func)
        return func

    return decorate


def pipeline_flow(**kwargs):
    def decorate(func: Callable[P, R]) -> Flow[P, R] | Callable[P, R]:
        if settings.prefect_enabled:
            final_kwargs = {**FLOW_DEFAULT_KWARGS, **kwargs}
            return flow(**final_kwargs)(func)
        return func

    return decorate
