FROM python:3.13-slim AS builder
ARG GIT_COMMIT_HASH

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y git curl gcc libpq-dev make

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_PROJECT_ENVIRONMENT=/usr/local/ \
    UV_PYTHON=/usr/local/bin/python \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_FROZEN=1


WORKDIR /src
ENV GIT_REPO="https://github.com/SNIFS-science/snifs-pipeline"
ENV GIT_COMMIT_HASH=$GIT_COMMIT_HASH

# Install third party dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
      uv sync --all-extras --frozen --no-dev

COPY src /src
COPY .streamlit /src/.streamlit
