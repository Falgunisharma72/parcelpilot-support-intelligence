VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help setup run test eval eval-verbose lint clean docker docker-run verify

help:
	@echo "make setup     - create the venv and install dependencies"
	@echo "make run       - start the app on http://localhost:8000"
	@echo "make test      - run the test suite (no API key needed)"
	@echo "make verify    - check the rules registry against the document pack"
	@echo "make eval      - run the golden set against the live agent (needs a key)"
	@echo "make docker    - build the container image"

setup:
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements-dev.txt
	@echo "ready. copy .env.example to .env and add your key, then: make run"

run:
	$(VENV)/bin/uvicorn app.api:app --reload --port 8000

test:
	$(PY) -m pytest tests -q

verify:
	$(PY) -c "from app.knowledge.rules import get_rules; p=get_rules().verify(); \
	print('\n'.join(p) if p else 'rules verified: every threshold still matches its clause in the PDFs')"

eval:
	$(PY) -m evals.run_evals

eval-verbose:
	$(PY) -m evals.run_evals --verbose --json report.json

docker:
	docker build -t parcelpilot-support .

docker-run:
	docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY=$$ANTHROPIC_API_KEY parcelpilot-support

clean:
	rm -rf .build .pytest_cache report.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
