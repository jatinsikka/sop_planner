"""Evaluate retrieval and planning end-to-end on all incidents."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schemas import IncidentEntry, Plan, SOPEntry, load_json
from src.eval.metrics import execution_success_rate, json_plan_f1, mrr_score, retrieval_recall_at_k
from src.pipeline.plan_pipeline import run_pipeline
from src.pipeline.skills import DummyMuJoCoAdapter, execute_plan
from src.planner.train_planner_lora import synthesize_target
from src.retrieval.infer_retrieve import retrieve_topk

console = Console()


def _reference_plans(sops: List[SOPEntry]) -> Dict[str, Plan]:
    """Treat the heuristic plan derived from the gold SOP as the reference."""
    return {sop.sop_id: Plan.model_validate(synthesize_target(sop)) for sop in sops}


def main() -> None:
    sops_path = str(PROJECT_ROOT / "src" / "data" / "sop_examples.json")
    incidents_path = str(PROJECT_ROOT / "src" / "data" / "incident_examples.json")

    sops = load_json(sops_path, SOPEntry, key="sop_examples")
    incidents = load_json(incidents_path, IncidentEntry, key="incident_examples")
    refs_by_sop = _reference_plans(sops)

    console.log(f"[bold]Loaded {len(sops)} SOPs and {len(incidents)} incidents[/bold]")

    console.log("[bold]Evaluating retrieval...[/bold]")
    golds: List[str] = []
    ranked_ids: List[List[str]] = []
    plans: List[Plan] = []
    for inc in incidents:
        golds.append((inc.labels or {}).get("sop_id", ""))
        hits = retrieve_topk(inc.text, sops_path=sops_path, k=5)
        ranked_ids.append([str(h["sop_id"]) for h in hits])

    r1 = retrieval_recall_at_k(golds, ranked_ids, k=1)
    r5 = retrieval_recall_at_k(golds, ranked_ids, k=5)
    mrr = mrr_score(golds, ranked_ids)

    console.log("[bold]Evaluating planner...[/bold]")
    f1s: List[float] = []
    exec_summaries = []
    api = DummyMuJoCoAdapter()
    for inc, gold in zip(incidents, golds):
        plan_obj = run_pipeline(inc.text, k=5, sops_path=sops_path)
        plans.append(plan_obj)
        if gold in refs_by_sop:
            f1s.append(json_plan_f1(refs_by_sop[gold], plan_obj))
        exec_summaries.append(execute_plan(plan_obj, api))

    avg_f1 = sum(f1s) / max(1, len(f1s))
    success = execution_success_rate(exec_summaries)
    valid_json = sum(1 for p in plans if p.steps) / max(1, len(plans))

    table = Table(title="Evaluation results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Retrieval Recall@1", f"{r1:.3f}")
    table.add_row("Retrieval Recall@5", f"{r5:.3f}")
    table.add_row("Retrieval MRR", f"{mrr:.3f}")
    table.add_row(f"Plan F1 (vs reference, n={len(f1s)})", f"{avg_f1:.3f}")
    table.add_row("Execution success rate", f"{success:.3f}")
    table.add_row("Valid plan rate", f"{valid_json:.3f}")
    console.print(table)


if __name__ == "__main__":
    main()
