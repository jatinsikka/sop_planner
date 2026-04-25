"""Retrieval test that runs only when the retriever is available."""
from __future__ import annotations

import pytest

st = pytest.importorskip("sentence_transformers")


def test_retrieval_returns_top_k():
    from src.retrieval.infer_retrieve import retrieve_topk

    hits = retrieve_topk("Machine A red light; pressure low", k=5)
    assert isinstance(hits, list)
    assert len(hits) == 5
    assert {"sop_id", "score", "text"}.issubset(hits[0].keys())
    assert hits[0]["score"] >= hits[-1]["score"]
