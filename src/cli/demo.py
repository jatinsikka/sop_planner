from __future__ import annotations

import json
import sys
from pathlib import Path
import typer
from rich.console import Console

# Add parent directory to path to enable imports from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.plan_pipeline import run_pipeline
from src.pipeline.skills import DummyMuJoCoAdapter, execute_plan
from src.retrieval.infer_retrieve import retrieve_topk
from src.data.schemas import SOPEntry, load_jsonl
from src.retrieval.index_utils import build_and_save_index

app = typer.Typer(help="SOP → Planner → MuJoCo Executor demo CLI")
console = Console()


@app.command()
def build_index():
    """Embed SOPs and build FAISS (or fallback) index."""
    project_root = Path(__file__).parent.parent.parent
    sops = load_jsonl(str(project_root / "src" / "data" / "sop_examples.jsonl"), SOPEntry)
    texts = [f"{s.title}. {s.condition}. Steps: " + " ; ".join(s.steps) for s in sops]
    ids = [s.sop_id for s in sops]
    build_and_save_index(texts, ids, str(project_root / "artifacts" / "retriever_bert" / "index"))
    console.log(f"[green]✓ Index built with {len(sops)} SOPs[/green]")


@app.command()
def retrieve(q: str = typer.Option(..., "--q", help="Incident text"), k: int = 5):
    """Retrieve top-k SOPs."""
    hits = retrieve_topk(q, k=k)
    for h in hits:
        console.print(h)


@app.command()
def plan(q: str = typer.Option(..., "--q", help="Incident text")):
    """Run full pipeline and print JSON plan."""
    p = run_pipeline(q, k=5)
    console.print_json(data=json.loads(p.model_dump_json()))


@app.command()
def exec(q: str = typer.Option(..., "--q", help="Incident text")):
    """Run plan and mock execute with Dummy adapter."""
    p = run_pipeline(q, k=5)
    api = DummyMuJoCoAdapter()
    summary = execute_plan(p, api)
    console.print_json(data={"plan": json.loads(p.model_dump_json()), "summary": summary})


if __name__ == "__main__":
    app()


