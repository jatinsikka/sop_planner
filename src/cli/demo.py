"""CLI: build the index, retrieve SOPs, generate plans, or run mock execution."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schemas import SOPEntry, load_json
from src.pipeline.plan_pipeline import run_pipeline
from src.pipeline.skills import DummyMuJoCoAdapter, execute_plan
from src.retrieval.index_utils import build_and_save_index
from src.retrieval.infer_retrieve import (
    DEFAULT_ENCODER_DIR,
    DEFAULT_INDEX_DIR,
    DEFAULT_SOPS_PATH,
    retrieve_topk,
)

app = typer.Typer(help="SOP → Planner → MuJoCo executor demo CLI", no_args_is_help=True)
console = Console()


def _synthesize(s: SOPEntry) -> str:
    return f"{s.title}. {s.condition}. Steps: " + " ; ".join(s.steps)


@app.command("build-index")
def build_index() -> None:
    """Embed all SOPs and write the FAISS index to artifacts/retriever/index."""
    sops = load_json(str(DEFAULT_SOPS_PATH), SOPEntry, key="sop_examples")
    texts = [_synthesize(s) for s in sops]
    ids = [s.sop_id for s in sops]
    encoder_arg = str(DEFAULT_ENCODER_DIR) if DEFAULT_ENCODER_DIR.exists() else None
    build_and_save_index(texts, ids, str(DEFAULT_INDEX_DIR), encoder_path=encoder_arg)
    console.log(f"[green]✓ Index built with {len(sops)} SOPs[/green]")


@app.command()
def retrieve(q: str = typer.Option(..., "--q", help="Incident text"), k: int = 5) -> None:
    """Print the top-k SOPs for `q`."""
    for h in retrieve_topk(q, k=k):
        console.print(h)


@app.command()
def plan(q: str = typer.Option(..., "--q", help="Incident text"), k: int = 5) -> None:
    """Run retrieval + planning and print the JSON plan."""
    p = run_pipeline(q, k=k)
    console.print_json(data=json.loads(p.model_dump_json()))


@app.command("exec")
def exec_cmd(q: str = typer.Option(..., "--q", help="Incident text"), k: int = 5) -> None:
    """Run the pipeline and dispatch the plan to the dummy MuJoCo adapter."""
    p = run_pipeline(q, k=k)
    summary = execute_plan(p, DummyMuJoCoAdapter())
    console.print_json(data={"plan": json.loads(p.model_dump_json()), "summary": summary})


if __name__ == "__main__":
    app()
