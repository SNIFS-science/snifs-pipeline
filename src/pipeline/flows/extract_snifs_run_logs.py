from pipeline.common.prefect_utils import pipeline_flow
from pipeline.config.deployment import SnifsNerscDeploymentConfig, registry
from pipeline.tasks import extract_snifs_run_logs
from pipeline.tasks.build_filestore import build_filestore


@registry.register(SnifsNerscDeploymentConfig(max_walltime=60))
@pipeline_flow()
def parse_snifs_runs_logs() -> None:
    # Load in the existing file store and ensure its up to date
    resolver = build_filestore(refresh=True)
    extract_snifs_run_logs(resolver)


if __name__ == "__main__":
    parse_snifs_runs_logs()
