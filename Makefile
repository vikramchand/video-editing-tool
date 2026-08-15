.PHONY: help install install-all dev test test-fast lint format run cli check docker docker-up clean reset-data mac mac-install

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
	@echo "make mac          - build the Mac app into macapp/build (macOS only)"
	@echo "make mac-install  - build it, install it to /Applications and launch it"
	@echo "make docker-up    - docker compose up --build"
	@echo "make clean        - remove build artefacts and caches"
	@echo "make reset-data   - delete the database and every stored video (asks first;"
	@echo "                    FORCE=1 skips the prompt)"

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

# The Mac app needs no virtualenv of its own: it builds one on first launch,
# inside ~/Library/Application Support. Requires the Xcode Command Line Tools.
mac:
	bash macapp/build.sh

mac-install:
	bash macapp/build.sh --install --open

# Paths come from Settings, so a relocated DATA_DIR/DATABASE_PATH is honoured.
# `clean` below only touches build artefacts; this touches your actual data.
reset-data:
	$(BIN)/video-understand reset $(if $(FORCE),--yes,)

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
