"""End-to-end pipeline: incident text → top SOP → structured plan."""
from __future__ import annotations

from pathlib import Path
from typing import List

from rich.console import Console

from src.data.schemas import Plan, SKILLS, SOPEntry, load_json
from src.planner.infer_plan import plan as planner_plan
from src.retrieval.infer_retrieve import retrieve_topk

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOPS_PATH = PROJECT_ROOT / "src" / "data" / "sop_examples.json"


def _get_sop_by_id(sops: List[SOPEntry], sop_id: str) -> dict:
    for s in sops:
        if s.sop_id == sop_id:
            return {
                "sop_id": s.sop_id,
                "title": s.title,
                "condition": s.condition,
                "steps": s.steps,
                "equipment": s.equipment,
            }
    raise KeyError(f"SOP not found: {sop_id}")


def run_pipeline(incident_text: str, k: int = 5, sops_path: str | Path | None = None) -> Plan:
    """Retrieve top-k SOPs for `incident_text` and generate a Plan from the best."""
    sops_path = str(sops_path or DEFAULT_SOPS_PATH)
    console.log("[bold]Pipeline: retrieve → plan[/bold]")

    sops = load_json(sops_path, SOPEntry, key="sop_examples")
    hits = retrieve_topk(incident_text, sops_path=sops_path, k=k)
    console.log(f"[dim]Top-{len(hits)} retrieved[/dim]")
    for i, h in enumerate(hits[:3], 1):
        console.log(f"  {i}. {h['sop_id']}: {h['text'][:60]}... (score={h['score']:.4f})")

    best_id = str(hits[0]["sop_id"])
    sop = _get_sop_by_id(sops, best_id)
    console.log(f"[bold green]Selected SOP: {best_id} — {sop['title']}[/bold green]")

    return planner_plan(incident_text, sop, skills=SKILLS)
