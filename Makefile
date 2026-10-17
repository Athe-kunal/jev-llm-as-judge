PY := PYTHONPATH=src .venv/bin/python
DATA := data/splits/reason_long_12k

.PHONY: help download data data-view

help:
	@echo "make download   fetch the pinned Reason+Long-12k sources into data/external (skips files already present)"
	@echo "make data       build $(DATA)/{train,eval}.jsonl and manifest.json (CPU, ~10 min; see docs/data.md)"
	@echo "make data-view  browse 5 examples per dataset at http://127.0.0.1:8790"

download:
	$(PY) -m spanjudge.reason_long --download

data: download
	$(PY) -m spanjudge.reason_long

data-view:
	$(PY) -m spanjudge.data_viz --port 8790
