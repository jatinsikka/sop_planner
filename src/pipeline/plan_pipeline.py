from __future__ import annotations

from typing import List

from rich.console import Console

from src.data.schemas import Plan, SKILLS, SOPEntry, load_jsonl
from src.retrieval.infer_retrieve import retrieve_topk
from src.retrieval.reranker_cross_encoder import Pair, predict
from src.planner.infer_plan import plan as planner_plan

console = Console()


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


def _repair_text_to_json(text: str) -> str:
    # Simple fix: ensure braces are balanced
    opens = text.count("{")
    closes = text.count("}")
    if opens > closes:
        text += "}" * (opens - closes)
    return text


def run_pipeline(incident_text: str, k: int = 5) -> Plan:
    console.log("[bold]Running pipeline: retrieve → plan[/bold]")
    sops = load_jsonl("src/data/sop_examples.jsonl", SOPEntry)
    
    # Retrieve top-k from dual-encoder (most reliable)
    hits = retrieve_topk(incident_text, k=k)
    console.log(f"[dim]Retrieved top-{len(hits)} SOPs by dual-encoder[/dim]")
    for i, h in enumerate(hits[:3]):
        console.log(f"  {i+1}. {h['sop_id']}: {h['text'][:60]}... (score: {h['score']:.4f})")
    
    # Use top result directly (dual-encoder is more reliable than reranker for this task)
    best_id = hits[0]["sop_id"]
    sop = _get_sop_by_id(sops, str(best_id))
    console.log(f"[bold green]Selected SOP: {best_id} - {sop['title']}[/bold green]")
    console.log(f"[dim]Condition: {sop['condition']}[/dim]")
    
    plan = planner_plan(incident_text, sop, skills=SKILLS)
    return plan


