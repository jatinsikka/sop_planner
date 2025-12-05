# 1. Train retriever (dual-encoder) on 100 SOPs
# Recommended: 50 epochs with batch_size 8 for stable convergence
python src/retrieval/build_dual_encoder.py

# 2. Train planner (Flan-T5 + LoRA) on 100 SOPs
# Recommended: 10 epochs with batch_size 4 for good generalization
python src/planner/train_planner_lora.py

# 3. Build FAISS index with trained encoder (run after training retriever)
python src/cli/demo.py build-index

# 4. Test retrieval on 100 SOPs
python src/cli/demo.py retrieve --q "Machine A pressure is low"

# 5. Test full pipeline (retrieval + planning)
python src/cli/demo.py plan --q "Machine A pressure is low"

# 6. Test full pipeline with execution
python src/cli/demo.py exec --q "Machine A pressure is low"

# 7. Comprehensive evaluation
python src/eval/evaluate_all.py



SOP → High-Level Planner → MuJoCo Executor
========================================

This repository provides a minimal, fully-local skeleton to:
- Train on SOPs so the system understands incidents, retrieves the right SOP (retrieval baseline = BERT with a robust TF-IDF/FAISS fallback), and generates a structured JSON plan using an instruction model (Flan-T5 with LoRA via PEFT).
- Execute the plan via a small MuJoCo skill API (here mocked with a dummy adapter).

Everything is designed to run locally with tiny dummy datasets and safe fallbacks where pretrained checkpoints may not yet be available.

Quickstart
----------

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
make build-index
make plan Q="Machine A red light; pressure low"
make exec Q="Machine A red light; pressure low"
```

Dataset
-------

**100 Standard Operating Procedures (SOPs)** covering:
- **Machine control** (SOP-001 to SOP-020): Pressure warnings, alarms, startups, shutdowns, diagnostics, sensor reads
- **Table 1 manipulation** (SOP-021 to SOP-040): Object handling, sorting, delivery, retrieval
- **Table 2 manipulation** (SOP-041 to SOP-060): Tool management, part organization, cross-table transfers
- **Complex workflows** (SOP-061 to SOP-100): Multi-step procedures, conditional logic, error handling, dual-object operations

**Data format** (`src/data/sop_examples.jsonl`):
```json
{
  "sop_id": "SOP-001",
  "title": "Low Pressure Warning",
  "condition": "Blue indicator is flashing",
  "steps": ["Walk to Machine", "Read pressure_sensor", ...],
  "equipment": ["machine", "pressure_sensor", ...]
}
```

**100 Incident Examples** (`src/data/incident_examples.jsonl`):
- Diverse incidents sampled from SOP conditions and contextual variations
- Used for retrieval evaluation and pipeline testing



Training
--------

### With 100 SOPs and 100 Incidents

**Optimal training hyperparameters (already configured in YAML):**

```bash
# Train retriever (dual-encoder BERT) - 50 epochs, batch 8
python src/retrieval/build_dual_encoder.py
# Saves: artifacts/retriever_bert/model_q, model_p, tokenizer, index/

# Train planner (Flan-T5 + LoRA) - 10 epochs, batch 4, r=32, alpha=64
python src/planner/train_planner_lora.py
# Saves: artifacts/planner_lora/adapter, tokenizer/

# Rebuild index after training (critical step!)
python src/cli/demo.py build-index
```

**Override defaults with CLI arguments:**

```bash
# Custom retriever training (override config)
python src/retrieval/build_dual_encoder.py --config config/retriever_config.yaml --epochs 100 --batch-size 16

# Custom planner training
python src/planner/train_planner_lora.py --epochs 20 --batch-size 8 --lora-r 64 --learning-rate 0.00005
```

**Configuration files control training:**
- `config/retriever_config.yaml`: Epochs (50), batch_size (8), LR (0.00005), warmup_steps (100)
- `config/planner_config.yaml`: Epochs (10), batch_size (4), LoRA r=32/alpha=64, learning_rate (0.0001)

**Before Training (First Time):**
```bash
# Build FAISS index from 100 SOPs (optional on first run)
python src/cli/demo.py build-index
```

**Typical Training Times (single GPU):**
- Retriever: ~50 epochs on 100 SOPs ≈ 2-5 minutes (batch_size=8)
- Planner: ~10 epochs on 100 pairs ≈ 1-2 minutes (batch_size=4, LoRA is efficient)
- Index building: <1 minute

**Monitoring Training Loss:**
- Retriever loss should decrease from ~1.5 → ~0.2-0.5
- Planner loss should decrease from ~5.0 → ~1.0-2.0
- Both use gradient checkpointing for memory efficiency



Architecture
------------

| Component  | Model/Method                        | Dataset | Notes                                                       |
|------------|-------------------------------------|---------|-------------------------------------------------------------|
| Retrieval  | Dual-encoder BERT (InfoNCE)         | 100 SOPs| Contrastive learning on 100 SOP examples; 50 epoch default  |
| Reranker   | DeBERTa-v3-base cross-encoder       | Static  | Simple [CLS] pooled + linear head for precision reranking    |
| Planning   | Flan-T5 with LoRA (PEFT)            | 100 SOPs| 100 synthetic pairs (1 per SOP); 10 epoch default; r=32      |
| Execution  | MuJoCo skill API (Dummy adapter)    | N/A     | Replace adapter with real MuJoCo bindings later             |

**Config-Driven Training:**
- `config/retriever_config.yaml`: 50 epochs, batch=8, warmup_steps=100, top_k reranking
- `config/planner_config.yaml`: 10 epochs, batch=4, LoRA r=32/alpha=64, max_plan_length=30
- All hyperparameters tunable via CLI (CLI args override config file)



Model Swaps
-----------
- Retrieval: swap to E5 or BGE by changing the model name in `build_dual_encoder.py`.
- Reranker: use a stronger cross-encoder such as DeBERTa-v3-large.
- Planner: swap to Llama-3-8B and train via QLoRA (PEFT supports this).

MuJoCo Integration
------------------
Replace the dummy adapter with a real MuJoCo interface in `src/env/mujoco_stub.py`. Map SOP names to your MJCF actuator/sensor names (e.g., button `green_reset` → `act_green_reset`; sensor `pressure_gauge_A`).

CLI
---
Typer CLI: `src/cli/demo.py`
- build-index: embed SOPs and build FAISS (or fallback)
- retrieve: `--q "incident text"`
- plan: `--q "incident text"` (full pipeline to JSON plan)
- exec: `--q "incident text"` (pipeline + dummy execution)

Makefile Targets
----------------
- build-index: builds retrieval model and index
- plan: runs full pipeline and prints JSON
- exec: runs pipeline + dummy execution
- test: runs pytest

Notes
-----

**With 100 SOPs and Incidents:**
- Config files now tuned for larger dataset: retriever (50 epochs, batch_size=8), planner (10 epochs, batch_size=4, LoRA r=32)
- Dual-encoder BERT trains on contrastive pairs from 100 SOPs using InfoNCE loss with batch negatives
- LoRA adapter scales to r=32 for better capacity across 100 diverse SOP scenarios
- FAISS index stores 100 SOP embeddings for fast retrieval (~500ms for top-5)
- Reranker (DeBERTa) provides precision by re-scoring top-k candidates

**Recommended Workflow:**
1. Train retriever: `python src/retrieval/build_dual_encoder.py` (config controls hyperparameters)
2. Build index: `python src/cli/demo.py build-index` (critical after training!)
3. Train planner: `python src/planner/train_planner_lora.py`
4. Evaluate: `python src/eval/evaluate_all.py` (runs on all 100 incidents)

**Troubleshooting:**
- If retrieval quality is poor, increase retriever epochs (config/retriever_config.yaml)
- If planner generates invalid plans, increase LoRA r or training epochs
- If memory issues, decrease batch_size in config files
- If pretrained checkpoints aren't downloaded yet, fallback to TF-IDF or heuristic generation

**Key Hyperparameters by Dataset Size:**
- **Small** (~10 SOPs): epochs=5, batch_size=2, LoRA r=8
- **Medium** (~100 SOPs): epochs=50, batch_size=8, LoRA r=32 [CURRENT]
- **Large** (~1000+ SOPs): epochs=100+, batch_size=32, LoRA r=64, with validation set

**Default Fallbacks:**
- If pretrained checkpoints missing → uses TF-IDF + heuristic fallback
- If model inference fails → heuristic planner generates backup plan
- All plans validated against safety whitelist (8 allowed skills)




