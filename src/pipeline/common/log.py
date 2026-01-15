import logging

from loguru import logger

# Ignore log messages from certain noisy loggers
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


def get_logger() -> logging.Logger:
    try:
        from prefect import get_run_logger

        prefect_logger = get_run_logger()
        return prefect_logger  # type: ignore

    except Exception:
        return logger  # type: ignore
