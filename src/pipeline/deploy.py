from collections import defaultdict

from loguru import logger
from prefect import deploy
from prefect.deployments.runner import RunnerDeployment

from pipeline import registry

THIS_IMAGE = "ghcr.io/snifs-science/snifs-pipeline/image"


def get_deployments() -> dict[str, list[RunnerDeployment]]:
    """Return a map of image names to deployments."""
    deployments = defaultdict(list)
    for flow, deployment in registry.deployments:
        runner_deployment = flow.to_deployment(
            name=flow.name,
            work_pool_name=deployment.work_pool_name,
            work_queue_name=deployment.work_queue,
            job_variables=deployment.get_job_variables(),
        )
        runner_deployment.entrypoint = "/" + runner_deployment.entrypoint  # type: ignore
        deployments[deployment.image].append(runner_deployment)
    return deployments


def register_deployments(deployment_map: dict[str, list[RunnerDeployment]]) -> None:
    # Each deploy call has a single work pool name (dont ask me why)
    # So we break this list into chunks of deployments with the same work pool name
    work_pool_deployments: dict[tuple[str, str], list[RunnerDeployment]] = defaultdict(list)
    for image, deployments in deployment_map.items():
        for deployment in deployments:
            assert deployment.work_pool_name is not None
            work_pool_deployments[(deployment.work_pool_name, image)].append(deployment)

    # Now we actually call said deployments
    for (work_pool_name, image), deployments in work_pool_deployments.items():
        logger.info(f"Registering {len(deployments)} deployments for work pool {work_pool_name} with image {image}")
        deploy(
            *deployments,
            work_pool_name=work_pool_name,
            image=image,
            build=False,
            push=False,
            print_next_steps_message=True,
        )


if __name__ == "__main__":
    url = "https://docs-3.prefect.io/v3/how-to-guides/cloud/connect-to-cloud#manually-configure-prefect-api-settings"
    logger.info("Please ensure you are logging into prefect cloud first.")
    logger.info(f"This probably means setting env vars as per {url}")
    logger.info("Starting deployment registration...")
    deployment_map = get_deployments()
    register_deployments(deployment_map)
    logger.info("Deployment registration complete.")
    # For a good debug run, try setting the primary file to /data/raw/runs/run_id=25_056_084/science_red.fits
