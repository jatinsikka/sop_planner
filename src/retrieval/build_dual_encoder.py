"""Fine-tune a sentence-transformers encoder on (incident, SOP) pairs with InfoNCE.

Defaults to BAAI/bge-small-en-v1.5 — a 33M-param model that runs comfortably on
CPU. MultipleNegativesRankingLoss is the contrastive (InfoNCE) objective: every
in-batch SOP that isn't the labelled match is treated as a negative.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schemas import IncidentEntry, SOPEntry, load_json
from src.retrieval.index_utils import DEFAULT_MODEL, build_and_save_index

console = Console()


def synthesize_text(sop: SOPEntry) -> str:
    return f"{sop.title}. {sop.condition}. Steps: " + " ; ".join(sop.steps)


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text()) or {}
    except ImportError:
        console.log("[yellow]PyYAML not installed; using defaults[/yellow]")
        return {}


def build_pairs(sops: List[SOPEntry], incidents: List[IncidentEntry]) -> List[Tuple[str, str]]:
    sop_text_by_id = {s.sop_id: synthesize_text(s) for s in sops}
    pairs: List[Tuple[str, str]] = []
    for inc in incidents:
        sop_id = (inc.labels or {}).get("sop_id")
        if sop_id and sop_id in sop_text_by_id:
            pairs.append((inc.text, sop_text_by_id[sop_id]))
    return pairs


def train(args: argparse.Namespace) -> None:
    sops = load_json(args.sops_path, SOPEntry, key="sop_examples")
    incident_path = Path(args.sops_path).parent / "incident_examples.json"
    incidents: List[IncidentEntry] = []
    if incident_path.exists():
        incidents = load_json(str(incident_path), IncidentEntry, key="incident_examples")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir = out_dir / "encoder"

    pairs = build_pairs(sops, incidents)
    if not pairs:
        console.log("[yellow]No labelled (incident, SOP) pairs found — skipping fine-tune; will index with zero-shot encoder.[/yellow]")
    else:
        try:
            from sentence_transformers import InputExample, SentenceTransformer, losses
            from torch.utils.data import DataLoader

            console.log(f"[bold]Fine-tuning {args.model_name} on {len(pairs)} pairs[/bold]")
            model = SentenceTransformer(args.model_name)
            examples = [InputExample(texts=[q, p]) for q, p in pairs]
            loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
            loss = losses.MultipleNegativesRankingLoss(model)
            warmup = max(1, int(0.1 * len(loader) * args.epochs))
            model.fit(
                train_objectives=[(loader, loss)],
                epochs=args.epochs,
                warmup_steps=warmup,
                show_progress_bar=True,
                optimizer_params={"lr": args.lr},
                output_path=str(encoder_dir),
            )
            console.log(f"[green]✓ Saved fine-tuned encoder to {encoder_dir}[/green]")
        except Exception as exc:
            console.log(f"[yellow]Fine-tune failed ({exc}); falling back to zero-shot encoder for indexing.[/yellow]")

    texts = [synthesize_text(s) for s in sops]
    ids = [s.sop_id for s in sops]
    encoder_arg = str(encoder_dir) if encoder_dir.exists() else None
    build_and_save_index(texts, ids, out_dir / "index", encoder_path=encoder_arg)
    console.log(f"[bold green]Artifacts saved under {out_dir}[/bold green]")


def parse_args() -> argparse.Namespace:
    cfg = load_config(PROJECT_ROOT / "config" / "retriever_config.yaml")
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    out_cfg = cfg.get("output", {})

    p = argparse.ArgumentParser(description="Train sentence-transformers encoder with InfoNCE.")
    p.add_argument("--model_name", type=str, default=model_cfg.get("name", DEFAULT_MODEL))
    p.add_argument("--sops_path", type=str, default=str(PROJECT_ROOT / data_cfg.get("train_json", "src/data/sop_examples.json")))
    p.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / out_cfg.get("out_dir", "artifacts/retriever")))
    p.add_argument("--epochs", type=int, default=train_cfg.get("epochs", 3))
    p.add_argument("--batch_size", type=int, default=train_cfg.get("batch_size", 16))
    p.add_argument("--lr", type=float, default=train_cfg.get("learning_rate", 2e-5))
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
