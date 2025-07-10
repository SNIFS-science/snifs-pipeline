from typing import Any, ParamSpec, TypeVar

from prefect import Flow
from pydantic import BaseModel, Field

from pipeline.config.global_settings import settings


class DeploymentConfig(BaseModel):
    work_pool_name: str = Field(
        default="docker",
        description="Name of the work queue to use.",
    )
    work_queue: str = Field(
        default="default",
        description="Name of Prefect work queue. Not NERSC. Probably don't touch this.",
    )
    qos: str = Field(
        default="shared",
        description="Quality of Service to use for the job. "
        "For NERSC, you have debug, regular, preempt, premium, interactive, shared_interactive, shared. "
        "Refer to https://docs.nersc.gov/jobs/policy/#qos-cost-factor-charge-multipliers-and-discounts for details.",
    )
    project: str = Field(
        default="snfactry",
        description="Project name to use for the job for billing.",
    )
    nodes: int = Field(
        default=1,
        description="Number of nodes to allocate..",
    )
    processes_per_node: int = Field(
        default=1,
        description="Number of processes to allocate per node.",
    )
    memory: int = Field(
        default=1024,
        description="Memory in MB to allocate.",
    )
    max_walltime: int = Field(
        default=3600,
        description="Maximum wall time in seconds.",
    )
    image: str = Field(
        default="ghcr.io/snifs-science/snifs-pipeline/image:latest",
        description="Docker image to use.",
    )

    def get_job_variables(self) -> dict[str, Any]:
        return {
            "volumes": [f"{settings.data_path}:/data:rw", f"{settings.output_path}:/output:rw"],
        }  # TODO: Blanking this out while we play with the docker worker
        return {
            "qos": self.qos,
            "project": self.project,
            "nodes": str(self.nodes),
            "processes_per_node": str(self.processes_per_node),
            "memory": str(self.memory),
            "max_walltime": str(self.max_walltime),
        }


P = ParamSpec("P")
R = TypeVar("R")


class Registry:
    def __init__(self):
        self.deployments: list[tuple[Flow, DeploymentConfig]] = []

    def register(self, config: DeploymentConfig):
        def decorator(func: Flow[P, R]) -> Flow[P, R]:
            self.deployments.append((func, config))
            return func

        return decorator


registry = Registry()
