from typing import Any, ParamSpec, TypeVar

from prefect import Flow
from pydantic import BaseModel, Field


class DeploymentConfig(BaseModel):
    work_pool_name: str = Field(
        default="slurm",
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
        default="UNSET",
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
        default=8192,
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
    volumes: list[tuple[str, str, str]] = Field(
        default_factory=list,
        description="List of volumes to mount in the container.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables to set in the container.",
    )

    def get_job_variables(self) -> dict[str, Any]:
        return {
            "qos": self.qos,
            "project": self.project,
            "nodes": str(self.nodes),
            "processes_per_node": str(self.processes_per_node),
            "memory": str(self.memory),
            "max_walltime": str(self.max_walltime),
            "volumes": self.volumes,
            "env": self.env,
        }


class SnifsDeploymentConfig(DeploymentConfig):
    project: str = Field(default="m112", description="Project name to use for the job for billing.")
    volumes: list[tuple[str, str, str]] = Field(
        default_factory=lambda: [
            # TODO make ro and change filestore location to output please
            ("/global/cfs/cdirs/m112/snifs/data", "/data", "rw"),
            ("/global/cfs/cdirs/m112/snifs/output", "/output", "rw"),
            ("/global/cfs/cdirs/m112/www/snifs", "/public", "rw"),
        ]
    )
    env: dict[str, str] = Field(
        default_factory=lambda: {
            "DATA_PATH": "/data",
            "OUTPUT_PATH": "/output",
            "PUBLIC_PATH": "/public",
            "PUBLIC_PATH_REPLACEMENT": "https://portal.nersc.gov/cfs/m112/",
        },
        description="Environment variables to set in the container.",
    )


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
