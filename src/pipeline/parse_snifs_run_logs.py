from pipeline.common.prefect_utils import pipeline_flow
from pipeline.tasks import parse_snifs_run_logs
from pipeline.tasks.build_filestore import build_filestore


@pipeline_flow()
def parse_snifs_runs_logs() -> None:
    # Load in the existing file store and ensure its up to date
    resolver = build_filestore()
    parse_snifs_run_logs(resolver)


if __name__ == "__main__":
    parse_snifs_runs_logs()
