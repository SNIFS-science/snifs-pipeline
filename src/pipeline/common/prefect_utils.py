from collections.abc import Callable

from prefect import Flow, flow, task
from prefect.client.schemas.objects import FlowRun, State

# from functools import wraps
# import time
# from pipeline.common.log import configure_logging
# from opentelemetry.trace import SpanKind, StatusCode
# from prefect.runtime.flow_run import get_flow_name

# from prefect.client.schemas.objects import StateType


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


def pipeline_task(**kwargs):
    def decorate(func: Callable) -> Callable:
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
    def decorate(func: Callable) -> Callable:
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
        #                 FLOW_PROCESSING_TIME.labels(name, result.type.value).observe(elapsed)
        #                 observed_time = True
        #                 if result.type != StateType.COMPLETED:
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
