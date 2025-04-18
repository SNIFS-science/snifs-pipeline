import json
import logging
import sys
import traceback
from collections.abc import Callable
from datetime import timezone as tz
from functools import partial, wraps
from sys import stderr
from types import FrameType
from typing import TYPE_CHECKING, TextIO, cast

from loguru import logger
from opentelemetry import trace
from opentelemetry.trace import INVALID_SPAN, INVALID_SPAN_CONTEXT

if TYPE_CHECKING:
    from loguru import Message

LOGGERS_TO_IGNORE = [
    name
    for name in logging.root.manager.loggerDict
    if any(x in name.lower() for x in ("uvicorn", "gunicorn", "pulsar"))
]
LOGGERS_TO_IGNORE += [
    "gunicorn",
    "gunicorn.access",
    "gunicorn.error",
    "uvicorn",
    "uvicorn.access",
    "uvicorn.error",
    "httpx",
    "httpx._client",
]


def sink_serializer(  # noqa: C901
    service: str,
    message: "Message",
    file: TextIO = stderr,
    json_format: bool = False,
) -> None:
    record = message.record
    level = record["level"].name
    simplified = {
        "service": service,
        "time": record["time"].astimezone(tz.utc).isoformat(timespec="milliseconds"),
        "level": level,
        "caller": f"{record['file'].name}:{record['line']}",
        "message": record["message"],
    }
    if "exception" in record:
        vals = record["exception"]
        if vals is not None:
            type, value, tb = record["exception"]  # type: ignore
            if type is not None:
                simplified["error_type"] = type.__name__  # type: ignore
            if value is not None:
                simplified["error_message"] = str(value)
            if tb is not None:
                simplified["error_traceback"] = "".join(traceback.format_tb(tb))
    if "extra" in record:
        simplified |= record["extra"]

    # ensure exceptions that have been added are jsonable
    for key, value in simplified.items():
        if isinstance(value, Exception):
            simplified[key] = str(value)

    # This logic is taken from opentelemetry-instrumentation-logging
    # opentelemetry.instrumentation.logging.__init__.py:111
    span = trace.get_current_span()
    if span != INVALID_SPAN:
        ctx = span.get_span_context()
        if ctx != INVALID_SPAN_CONTEXT:
            # simplified["otelServiceName"] = service
            simplified["trace_id"] = format(ctx.trace_id, "016x")
            simplified["span_id"] = format(ctx.span_id, "032x")

    # Ensure message is the last element
    simplified["message"] = simplified.pop("message")

    if json_format:
        serialized = json.dumps(simplified, skipkeys=True)
    else:
        serialized = ""
        for key, value in simplified.items():
            serialized += f"{key}={value} "

    if span != INVALID_SPAN:
        span.add_event("log", simplified, timestamp=int(record["time"].timestamp() * 1e9))
    print(serialized, file=file)


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if record.name in LOGGERS_TO_IGNORE:
            return
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        frame, depth = logging.currentframe(), 1
        while frame.f_code.co_filename in (logging.__file__, __file__):  # noqa: WPS609
            frame = cast(FrameType, frame.f_back)
            depth += 1
        logger_with_opts = logger.opt(depth=depth, exception=record.exc_info)
        try:
            logger_with_opts.log(level, "{}", record.getMessage())
        except Exception as e:
            safe_msg = getattr(record, "msg", None) or str(record)
            logger_with_opts.warning("Exception logging the following native logger message: {}, {!r}", safe_msg, e)


def configure_logging(service: str) -> None:
    for name in LOGGERS_TO_IGNORE:
        logga = logging.getLogger(name)
        logga.handlers = []

    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)
    logger.remove()
    logger.add(sink=partial(sink_serializer, service, file=sys.stderr))


def get_logger() -> logging.Logger:
    extra: dict[str, str] = {}
    try:
        from prefect import get_run_logger

        prefect_logger = get_run_logger()
        # TODO: remove this and use loguru later on
        return prefect_logger  # type: ignore
        extra = prefect_logger.extra

    except Exception:
        return logger  # type: ignore

    def intercept_prefect_log(func: Callable, level: str):
        @wraps(func)
        def wrapper(msg, *args, **kwargs):
            logger.bind(**extra).log(level.upper(), msg)

        return wrapper

    # Create an intercept for the warning, error, and exception functions
    for level in ["debug", "info", "warning", "error", "exception"]:
        fn = getattr(prefect_logger, level)
        setattr(prefect_logger, level, intercept_prefect_log(fn, level))

    return prefect_logger  # type: ignore
