from __future__ import annotations

from typing import List

from rich.console import Console

from src.data.schemas import IncidentEntry, SOPEntry, load_jsonl, Plan
from src.eval.metrics import execution_success_rate, json_plan_f1, mrr_score, retrieval_recall_at_k
from src.pipeline.plan_pipeline import run_pipeline
from src.pipeline.skills import DummyMuJoCoAdapter, execute_plan

console = Console()


def main() -> None:
    sops = load_jsonl("src/data/sop_examples.jsonl", SOPEntry)
    incidents = load_jsonl("src/data/incident_examples.jsonl", IncidentEntry)

    # Retrieval evaluation (simple)
    golds = [i.labels.get("sop_id", "") if i.labels else "" for i in incidents]
    ranked_ids: List[List[str]] = []
    from src.retrieval.infer_retrieve import retrieve_topk

    for inc in incidents:
        hits = retrieve_topk(inc.text, k=5)
        ranked_ids.append([str(h["sop_id"]) for h in hits])

    r1 = retrieval_recall_at_k(golds, ranked_ids, k=1)
    r5 = retrieval_recall_at_k(golds, ranked_ids, k=5)
    mrr = mrr_score(golds, ranked_ids)
    console.log(f"Retrieval: Recall@1={r1:.2f}, Recall@5={r5:.2f}, MRR={mrr:.2f}")

    # Plan F1 vs tiny refs (inline)
    refs = {
        "INC-001": Plan(goal="Pressure Low in Machine A", steps=[{"skill": "walk_to", "args": {"target": "Machine A"}}], fallback=[]),
        "INC-002": Plan(goal="Reset Machine A Alarm", steps=[{"skill": "press_button", "args": {"button": "green_reset"}}], fallback=[]),
    }
    f1s: List[float] = []
    exec_summaries = []
    api = DummyMuJoCoAdapter()
    for inc in incidents:
        p = run_pipeline(inc.text, k=5)
        if inc.incident_id in refs:
            f1s.append(json_plan_f1(refs[inc.incident_id], p))
        exec_summaries.append(execute_plan(p, api))
    console.log(f"Plan JSON F1 (on tiny refs): {sum(f1s)/max(1,len(f1s)):.2f}")
    console.log(f"Execution success rate: {execution_success_rate(exec_summaries):.2f}")


if __name__ == "__main__":
    main()


