install_uv:
	@if [ -f "uv" ]; then echo "Downloading uv" && curl -LsSf https://astral.sh/uv/install.sh | sh; else echo "uv already installed"; fi
	uv self update

install_python:
	uv python install

install_deps:
	uv sync --all-extras

install_precommit:
	uv run pre-commit install
	uv run pre-commit gc

update_precommit:
	uv run pre-commit autoupdate
	uv run pre-commit gc

precommit:
	uv run pre-commit run --all-files

test:
	uv run pytest tests

tests: test
install: install_uv install_python install_deps install_precommit


docker_prefect:
	docker run -p 4200:4200 --env PREFECT_SERVER_API_HOST="0.0.0.0" prefecthq/prefect:3-latest prefect server start