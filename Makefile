PY ?= python

.PHONY: install train-retriever train-planner build-index plan exec eval test

install:
	$(PY) -m pip install -r requirements.txt

train-retriever:
	$(PY) -m src.retrieval.build_dual_encoder

train-planner:
	$(PY) -m src.planner.train_planner_lora

build-index:
	$(PY) -m src.cli.demo build-index

plan:
	$(PY) -m src.cli.demo plan --q "$(Q)"

exec:
	$(PY) -m src.cli.demo exec --q "$(Q)"

eval:
	$(PY) -m src.eval.evaluate_all

test:
	pytest -q
