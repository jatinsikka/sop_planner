# SOP-Guided Robotic Task Planning

End-to-end system that retrieves Standard Operating Procedures (SOPs) from natural-language incident descriptions and generates structured robot plans. Built as the "brain" of a larger autonomous robotics project — this repo handles perception-to-plan; physical execution runs on a [Unitree G1 humanoid via MuJoCo](https://github.com/jatinsikka/mujoco_G1_AMO).

## Pipeline

```
Incident text ──► BGE-small dual-encoder ──► Top-K SOPs ──► Qwen2.5 + LoRA ──► JSON plan ──► Robot
                  (sentence-transformers)     (FAISS IP)     (causal LM)        (7 skills)
```

1. **Retrieval** — `BAAI/bge-small-en-v1.5` (33M params) is fine-tuned on labelled (incident, SOP) pairs with `MultipleNegativesRankingLoss` (the InfoNCE objective). FAISS holds normalized embeddings; cosine = inner product.
2. **Planning** — `Qwen/Qwen2.5-0.5B-Instruct` (494M params) with a LoRA adapter (`r=16`, attention `q/k/v/o_proj`) is trained to emit a JSON plan. JSON is parsed with `json-repair` so truncated or quote-mangled output still validates.
3. **Execution** — Plans are validated against the seven-skill whitelist (`walk_to`, `press_button`, `wait`, `read_sensor`, `pick`, `place`, `notify`) and dispatched to a MuJoCo skill API (mocked in `DummyMuJoCoAdapter`).

## Why this stack

- **Zero-cost / offline.** Both models run on CPU. No paid APIs, no GPU required for inference.
- **Modern small models.** BGE-small and Qwen2.5-0.5B are state-of-the-art among models in their size class as of 2025-26 — and small enough to ship.
- **Heuristic fallback.** If the LoRA adapter is missing or generation fails, a deterministic SOP-step → skill mapper still produces an executable plan. The pipeline never crashes.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\Activate.ps1
make install

make train-retriever                # fine-tune BGE-small on incident↔SOP pairs (~2 min CPU)
make train-planner                  # LoRA-tune Qwen2.5-0.5B (~10 min CPU, faster on GPU)
make build-index                    # embed SOPs + write FAISS index

make plan Q="Machine A pressure is low"
make exec Q="Machine A pressure is low"
make eval                           # full retrieval + planning eval over 100 incidents
make test                           # pytest
```

You can run `make plan` without training first — the heuristic planner runs and the index is built on demand from the zero-shot BGE encoder.

## Results

Measured on all 100 labelled incidents (`make eval`), zero-shot encoder + heuristic planner (no training required):

| Metric | Value |
|--------|------:|
| Retrieval Recall@1 | 0.960 |
| Retrieval Recall@5 | 0.990 |
| Retrieval MRR | 0.975 |
| Plan F1 (vs SOP-derived reference, n=100) | 0.995 |
| Execution success rate | 1.000 |
| Valid plan rate | 1.000 |

Reference plans are derived from each gold SOP by the same heuristic mapper used as the planner's fallback, so this measures *retrieval correctness × plan well-formedness* end-to-end. Training the LoRA planner is optional and only matters once SOPs are diverse enough that the heuristic cannot cover them.

## Dataset

100 synthetic manufacturing SOPs across three categories:
- **Machine Control** (SOP-001 — SOP-020): pressure / temperature alerts, startup, shutdown
- **Table Manipulation** (SOP-021 — SOP-060): object handling, tool / part organization
- **Complex Workflows** (SOP-061 — SOP-100): multi-step emergencies, conditional logic

Plus 100 incident examples with ground-truth SOP labels for evaluation.

## Project structure

```
src/
├── retrieval/          # BGE-small training, FAISS index, optional Ollama RAG baseline
├── planner/            # Qwen2.5 + LoRA training, inference, JSON repair
├── pipeline/           # Plan orchestration + skill API
├── eval/               # Retrieval / planning metrics, baselines
├── data/               # 100 SOPs + 100 incidents (JSON)
└── cli/                # Typer CLI: build-index, retrieve, plan, exec
config/                 # Training hyperparameters (YAML)
artifacts/              # Trained encoders, LoRA adapter, FAISS index (gitignored)
tests/                  # Schema, JSON repair, retrieval, integration, execution
```

## Architecture details

| Component | Model | Notes |
|-----------|-------|-------|
| Retriever | `BAAI/bge-small-en-v1.5` (33M) | Fine-tuned with `MultipleNegativesRankingLoss` on labelled pairs |
| Planner   | `Qwen/Qwen2.5-0.5B-Instruct` (494M) | LoRA adapter, attention-only targets, ~0.5% of params trained |
| Reranker  | `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M) | Optional, off by default |
| Index     | FAISS `IndexFlatIP` | Cosine via normalized embeddings; numpy fallback if FAISS missing |
| JSON parse | `json-repair` | Recovers from truncated / single-quoted / fenced output |

Configuration lives in `config/retriever_config.yaml` and `config/planner_config.yaml`. CLI flags override config values.

## Related repo

- **This repo (`sop_planner`)** — SOP retrieval + plan generation
- **[`mujoco_G1_AMO`](https://github.com/jatinsikka/mujoco_G1_AMO)** — Unitree G1 locomotion + manipulation in MuJoCo

## Paper

See [`DL_Project_report.pdf`](DL_Project_report.pdf) for the writeup. Note: numbers in the original PDF reference an earlier BERT/Flan-T5 stack; the codebase has since been modernized to BGE-small + Qwen2.5.
