"""Top-k SOP retrieval with auto-build of the index on first use."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schemas import SOPEntry, load_json
from src.retrieval.index_utils import build_and_save_index, load_index, search

console = Console()

DEFAULT_SOPS_PATH = PROJECT_ROOT / "src" / "data" / "sop_examples.json"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "artifacts" / "retriever" / "index"
DEFAULT_ENCODER_DIR = PROJECT_ROOT / "artifacts" / "retriever" / "encoder"


def _synthesize_text(sop: SOPEntry) -> str:
    return f"{sop.title}. {sop.condition}. Steps: " + " ; ".join(sop.steps)


def retrieve_topk(
    incident_text: str,
    sops_path: Optional[str] = None,
    index_dir: Optional[str] = None,
    encoder_dir: Optional[str] = None,
    k: int = 5,
) -> List[Dict[str, str | float]]:
    sops_path = sops_path or str(DEFAULT_SOPS_PATH)
    index_dir = index_dir or str(DEFAULT_INDEX_DIR)
    encoder_dir = encoder_dir or str(DEFAULT_ENCODER_DIR)

    sops = load_json(sops_path, SOPEntry, key="sop_examples")
    texts = [_synthesize_text(s) for s in sops]
    ids = [s.sop_id for s in sops]

    out = Path(index_dir)
    if not (out / "embeddings.npy").exists():
        console.log("[yellow]Index not found. Building now...[/yellow]")
        encoder_arg = encoder_dir if Path(encoder_dir).exists() else None
        build_and_save_index(texts, ids, out, encoder_path=encoder_arg)

    embeddings, id_list, faiss_index, meta, vectorizer = load_index(out)
    encoder_arg = encoder_dir if Path(encoder_dir).exists() else None
    results = search(
        [incident_text],
        embeddings,
        id_list,
        faiss_index,
        top_k=k,
        meta=meta,
        vectorizer=vectorizer,
        encoder_path=encoder_arg,
    )[0]

    id_to_text = {s.sop_id: _synthesize_text(s) for s in sops}
    return [{"sop_id": sid, "score": score, "text": id_to_text.get(sid, "")} for sid, score in results]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrieve top-k SOPs for an incident.")
    p.add_argument("--q", type=str, default="Machine A red light; pressure low")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--sops", type=str, default=str(DEFAULT_SOPS_PATH))
    p.add_argument("--index_dir", type=str, default=str(DEFAULT_INDEX_DIR))
    p.add_argument("--encoder_dir", type=str, default=str(DEFAULT_ENCODER_DIR))
    p.add_argument("--rebuild_index", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.rebuild_index:
        sops = load_json(args.sops, SOPEntry, key="sop_examples")
        texts = [_synthesize_text(s) for s in sops]
        ids = [s.sop_id for s in sops]
        encoder_arg = args.encoder_dir if Path(args.encoder_dir).exists() else None
        build_and_save_index(texts, ids, args.index_dir, encoder_path=encoder_arg)
    hits = retrieve_topk(args.q, sops_path=args.sops, index_dir=args.index_dir, encoder_dir=args.encoder_dir, k=args.k)
    for h in hits:
        console.print(h)
