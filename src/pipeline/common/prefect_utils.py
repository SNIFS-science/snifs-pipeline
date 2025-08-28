import asyncio
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from prefect import Flow, flow, get_client, task
from prefect.client.schemas.objects import FlowRun, State
from prefect.deployments import run_deployment as prefect_run_deployment
from prefect.exceptions import PrefectHTTPStatusError

# from functools import wraps
# import time
# from pipeline.common.log import configure_logging
# from opentelemetry.trace import SpanKind, StatusCode
# from prefect.runtime.flow_run import get_flow_name

# from prefect.client.schemas.objects import StateType
P = ParamSpec("P")
R = TypeVar("R")


def on_finish(flow: Flow, flow_run: FlowRun, state: State):
    pass


TASK_DEFAULT_KWARGS = {
    "retries": 0,
    "retry_delay_seconds": 10,
    "log_prints": False,
    "timeout_seconds": 3600,  # An hour timeout per task
    "cache_result_in_memory": False,
}


FLOW_DEFAULT_KWARGS = {
    "timeout_seconds": 3600 * 24 * 7,  # A week timeout per flow
    # "on_crashed": [on_finish],
    # "on_failure": [on_finish],
    # "on_completion": [on_finish],
    # "on_cancellation": [on_finish],
    "log_prints": False,
    "cache_result_in_memory": False,
}


async def run_deployment(
    flow_name: str,
    deployment_name: str,
    flow_run_name: str | None = None,
    parameters: dict | None = None,
    timeout: int | None = None,
    poll_interval: int = 60,
) -> FlowRun:
    flow_run = await prefect_run_deployment(
        f"{flow_name}/{deployment_name}",
        flow_run_name=flow_run_name,
        parameters=parameters,
        timeout=0,
    )  # type: ignore

    flow_run_id = flow_run.id
    async with get_client() as client:
        async with asyncio.timeout(timeout):
            while True:
                await asyncio.sleep(poll_interval)
                try:
                    flow_run = await client.read_flow_run(flow_run_id)
                    flow_state = flow_run.state
                    if flow_state and flow_state.is_final():
                        return flow_run
                except PrefectHTTPStatusError:
                    pass

    return flow_run


def pipeline_task(**kwargs):
    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        # tracer = get_tracer(settings.service)
        # final_kwargs = {**TASK_DEFAULT_KWARGS, **kwargs}
        # name = kwargs.get("name", func.__name__)

        # @task(**final_kwargs)
        # @wraps(func)
        # def wrapper(*args, **kwargs):
        #     with tracer.start_as_current_span(name, kind=SpanKind.SERVER) as span:
        #         try:
        #             result = func(*args, **kwargs)
        #             span.set_status(StatusCode.OK)
        #             return result
        #         except Exception as e:
        #             span.record_exception(e)
        #             span.set_status(StatusCode.ERROR, description=f"{type(e).__name__}: {e}")
        #             raise

        # return wrapper

        final_kwargs = {**TASK_DEFAULT_KWARGS, **kwargs}
        return task(**final_kwargs)(func)

    return decorate


def pipeline_flow(**kwargs):
    def decorate(func: Callable[P, R]) -> Flow[P, R]:
        final_kwargs = {**FLOW_DEFAULT_KWARGS, **kwargs}
        return flow(**final_kwargs)(func)

        # tracer = get_tracer(settings.service)
        # final_kwargs = {**FLOW_DEFAULT_KWARGS, **kwargs}

        # @flow(**final_kwargs)
        # @wraps(func)
        # def wrapper(*args, **kwargs):
        #     configure_logging(settings.service)
        #     name = get_flow_name()
        #     if name is None:
        #         name = func.__name__
        #     with tracer.start_as_current_span(name, kind=SpanKind.SERVER) as span:
        #         start = time.perf_counter()
        #         observed_time = False
        #         try:
        #             FLOW_INVOCATIONS.labels(name).inc()
        #             push_metrics(initial_registry)
        #             result = func(*args, **kwargs)
        #             elapsed = time.perf_counter() - start

        #             # Note because flows can crash, we don't handle the post-execution
        #             # prometheus here
        #             if isinstance(result, State):
        #                 FLOW_PROCESSING_TIME.labels(name, result.file_type.value).observe(elapsed)
        #                 observed_time = True
        #                 if result.file_type != StateType.COMPLETED:
        #                     span.set_status(StatusCode.OK)
        #                 else:
        #                     span.set_status(StatusCode.ERROR, description=result.message)
        #             else:
        #                 FLOW_PROCESSING_TIME.labels(name, "COMPLETED").observe(elapsed)
        #                 observed_time = True
        #                 span.set_status(StatusCode.OK)
        #             push_metrics(interim_registry)
        #             return result
        #         except Exception as e:
        #             elapsed = time.perf_counter() - start
        #             if not observed_time:
        #                 FLOW_PROCESSING_TIME.labels(name, "FAILED").observe(elapsed)
        #                 push_metrics(interim_registry)
        #             span.record_exception(e)
        #             span.set_status(StatusCode.ERROR, description=f"{type(e).__name__}: {e}")
        #             raise

        # return wrapper

    return decorate
