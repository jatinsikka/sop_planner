"""Cross-encoder reranker over (incident, SOP) pairs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from rich.console import Console

console = Console()

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class Pair:
    incident: str
    sop_text: str
    sop_id: str


_model = None


def _load(model_name: str = DEFAULT_RERANKER):
    global _model
    if _model is not None:
        return _model
    from sentence_transformers import CrossEncoder

    _model = CrossEncoder(model_name)
    return _model


def _word_overlap(incident: str, sop: str) -> float:
    a = set(incident.lower().split())
    b = set(sop.lower().split())
    return float(len(a & b)) / (1.0 + len(a))


def predict(pairs: Sequence[Pair], model_name: str = DEFAULT_RERANKER) -> List[Tuple[str, float]]:
    """Return [(sop_id, score)] sorted by descending relevance."""
    if not pairs:
        return []
    try:
        model = _load(model_name)
        scores = model.predict([(p.incident, p.sop_text) for p in pairs]).tolist()
    except Exception as exc:
        console.log(f"[yellow]Cross-encoder unavailable, using word-overlap fallback: {exc}[/yellow]")
        scores = [_word_overlap(p.incident, p.sop_text) for p in pairs]

    ranked = sorted(zip([p.sop_id for p in pairs], scores), key=lambda x: -x[1])
    return [(sid, float(s)) for sid, s in ranked]
