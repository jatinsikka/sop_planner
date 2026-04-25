"""End-to-end integration: top-1 retrieval correctness on labelled incidents."""
from __future__ import annotations

from pathlib import Path

import pytest

st = pytest.importorskip("sentence_transformers")

from src.data.schemas import IncidentEntry, load_json
from src.pipeline.plan_pipeline import run_pipeline
from src.pipeline.skills import DummyMuJoCoAdapter, execute_plan
from src.retrieval.infer_retrieve import retrieve_topk

ROOT = Path(__file__).resolve().parents[1]
SOPS = str(ROOT / "src" / "data" / "sop_examples.json")
INCIDENTS = str(ROOT / "src" / "data" / "incident_examples.json")


def _sample_incidents(n: int = 8):
    incs = load_json(INCIDENTS, IncidentEntry, key="incident_examples")
    step = max(1, len(incs) // n)
    return incs[::step][:n]


def test_top1_retrieval_hits_majority():
    incs = _sample_incidents(n=8)
    correct = 0
    for inc in incs:
        gold = (inc.labels or {}).get("sop_id", "")
        hits = retrieve_topk(inc.text, sops_path=SOPS, k=1)
        if hits and str(hits[0]["sop_id"]) == gold:
            correct += 1
    # Sentence-transformers + BGE-small should easily clear half on this dataset.
    assert correct >= len(incs) // 2, f"only {correct}/{len(incs)} top-1 correct"


def test_pipeline_returns_executable_plan():
    incs = _sample_incidents(n=3)
    api = DummyMuJoCoAdapter()
    for inc in incs:
        plan = run_pipeline(inc.text, k=5, sops_path=SOPS)
        assert plan.steps, "plan has no steps"
        assert plan.steps[-1].skill == "notify"
        result = execute_plan(plan, api)
        assert result["success"], f"execution failed: {result}"
