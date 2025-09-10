# Prefect-Orchestrated Processing Pipeline



## Motivation

My prior experiences with data reduction pipelines and pipelines left me dissatisfied with how things are normally done, both from the persepctive of a maintainer, a developer, and a user of the end products.

Pipelines were run on unknown hardware, out-of-sight and out-of-mind... until something went wrong. Debugging the output of a pipeline meant emailing around to find who maintained the pipeline and seeing if they could access logs or any form of observability on their side. None of this was available to me, as an end user. Data lineage was non-existent, apart from sometimes being a string in a FITS header no one used, which caused chaos when people identified known issues with various verions of a pipeline. Hell, many pipelines ran unversioned, with people wantonly editing code on the fly in their institutions HPC such that at any given point in time no one could even share a well-defined version of "What the code looks like."

As a developer of pipelines, my end users had similar observability problems. Structuring code, logs, and debug output better made my life easier when tracking down problems, but I still acted as a middle man. I couldn't simply give access to all those that consumed my data products the interim steps, because they were locked down inside different systems (Getafix, Friday, Midway, NERSC, etc).

This repository serves as an example of a possible path forward in how pipelines, from observation reduction to cosmology fitting, can be engineered to address as many of these issues as possible.

## Guiding Principles

### Observability (O)

1. Anyone should be able to track metrics associated with a pipeline from one run to the next.
2. Anyone should be able to drill down into logs and events from a pipeline without needing secure access to an HPC or knowledge of a filesystem layout.
3. Debug artifacts, images, and plots, should go hand-in-hand with log output and be accessed just as easily.
4. Pipeline outputs should have a full accounting of lineage, such as what algorithms were run and summary statistics about their output.
5. Metric evolution and reprocessing should allow users to see evolution over time and pipeline versioning.

### Execution (E)

1. Pipelines can be composed of numerous steps that run on different hardware or with different images.
2. Pipelines should declare the hardware they need to run in the same place as the code they execute.
3. Pipelines can be executed with validated inputs on HPC/Cloud without requiring pre or post steps.
4. Pipelines can be executed without needing intimate knowledge of HPC/Cloud providers

### Software standards (S)

1. Code should always been provided in an image.
2. Code should always be linted to an agreed ruleset
3. Entrypoint code should always be documented
4. Code should always be type hinted
5. Applicable code should be unit tested
6. Code dependencies and project management should use best-in-industry tools.


## Implementing the principles

### Orchestration layer

Adopting Prefect as the orchestration engine fulfills many of these conditions. Prefect gathers logs and artifacts from flow runs, and allows Pydantic objects to be created a flow configuration, and then uses the OpenAPI json spec these objects product to create validated web inputs. Unlike other orchestration products like Airflow/Dagster, Prefect is closest to stock python, and decorates can be disabled via environment variables to remove Prefect completely for local execution (if wanted), which would not be possible when conforming to something like Dagster's asset-first methodology.

Specifically, adopting Prefect will help fulfill O1-4, E1-4, S1.

For advanced filtering of flow runs, cross-flow run comparison, and displaying some of the more astro-specific debug artifacts, we can easily create a streamlit dashboard which could either utilise a separate database backing (including a simple SQLite database), and/or utilise Prefect's python API to retrieve logs, artifacts, flow runs, task runs, deployment configurations, etc.

### Containerisation

Docker is industry standard, but Podman/apptainer are still common in scientific computing (being a bit closer to bare metal). All of them are able to execute standard docker images.

### Software tools

`uv` has absolutely taken off as the best tool for dependency, venv, and project management. `uv_build` may soon eclipse `hatchling` ([the recommended system from packaging.python.org](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)). Requirements.txt and setup.py approaches should be strongly discouraged given PEP 621 is over 5 years old now, not to mention we'd never run into build isolation issues (ie PEP 517 and PEP 518) if we enforce proper project management for internal libraries.

`ruff` has replaced the entirety of black, flake8, isort, pyflakes, and pylint. We will be using it.

Static type checkers are still in active development and enforcing one over another is probably not the path we want to go given the lack of maturity in many of the new tools. `ty` (by Astral) and `pyrefly` (by Facebook) are both aiming to make `mypy` redundant. Let us re-evaluate in the future if either of these tools pulls ahead such that we'd want to explicitly recommend one of them.


## Runthrough

