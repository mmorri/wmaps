.PHONY: venv deps api train test fmt

PY?=3.11
VENV=.venv

venv:
	uv python install $(PY)
	uv venv --python $(PY) $(VENV)
	. $(VENV)/bin/activate

deps:
	. $(VENV)/bin/activate; uv pip install -r requirements.txt

api:
	. $(VENV)/bin/activate; API_PORT=8001 python book_api.py

train:
	. $(VENV)/bin/activate; python train_and_export.py

test:
	. $(VENV)/bin/activate; python test_api.py

fmt:
	. $(VENV)/bin/activate; python -m black . || true
