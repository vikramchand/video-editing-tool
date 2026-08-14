.PHONY: help install install-all dev test test-fast lint format run cli check docker docker-up clean

PYTHON ?= python3.12
VENV   ?= .venv
BIN     = $(VENV)/bin

help:
	@echo "make install      - create a venv and install the package"
	@echo "make install-all  - the above plus whisper and pyscenedetect"
	@echo "make check        - verify ffmpeg and model availability"
	@echo "make test         - run the full test suite"
	@echo "make test-fast    - skip tests that shell out to ffmpeg"
	@echo "make lint         - ruff check"
	@echo "make run          - start the API on http://localhost:8000"
	@echo "make demo         - process the bundled example with mock providers"
	@echo "make docker-up    - docker compose up --build"
	@echo "make clean        - remove build artefacts and caches"

$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

install: $(VENV)
	$(BIN)/pip install -e ".[dev]"

install-all: $(VENV)
	$(BIN)/pip install -e ".[all,dev]"

check:
	$(BIN)/video-understand check

test:
	$(BIN)/python -m pytest -v

test-fast:
	$(BIN)/python -m pytest -v -m "not ffmpeg" -k "not sample_video"

lint:
	$(BIN)/ruff check video_understanding tests

format:
	$(BIN)/ruff format video_understanding tests

run:
	$(BIN)/uvicorn video_understanding.api.app:app --reload --host 0.0.0.0 --port 8000

demo:
	TRANSCRIPTION_PROVIDER=mock VISION_PROVIDER=mock LLM_PROVIDER=mock \
		$(BIN)/video-understand $(VIDEO)

docker-up:
	docker compose up --build

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
