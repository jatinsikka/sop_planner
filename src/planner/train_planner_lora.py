from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

from rich.console import Console

# Add parent directory to path to enable imports from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.schemas import SOPEntry, SKILLS, load_jsonl

console = Console()


def load_config(config_path: str | Path) -> dict:
    """Load configuration from YAML file."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        console.log("[yellow]PyYAML not installed, using default config[/yellow]")
        return {}
    except FileNotFoundError:
        console.log(f"[yellow]Config file not found: {config_path}, using defaults[/yellow]")
        return {}


def build_instruction(incident: str, sop: SOPEntry) -> str:
    steps = " ; ".join(sop.steps)
    return (
        "[INCIDENT]\n"
        f"{incident}\n"
        "[SOP]\n"
        f"{sop.title}\n{sop.condition}\n{steps}\n"
        "[ALLOWED_SKILLS]\n"
        f"{','.join(SKILLS)}"
    )


def synthesize_pairs(sops: List[SOPEntry]) -> List[Tuple[str, str]]:
    """Build training pairs with realistic multi-step plans from SOP steps."""
    pairs: List[Tuple[str, str]] = []
    for sop in sops:
        incident = f"{sop.title}. {sop.condition}"
        instr = build_instruction(incident, sop)
        
        # Map SOP steps to realistic multi-step plan
        steps = []
        step_text = " ".join(sop.steps).lower()
        
        # Add walk_to if any "walk to" in steps
        if "walk to" in step_text:
            steps.append({"skill": "walk_to", "args": {"target": "machine"}})
        
        # Add read_sensor if any "read" in steps
        if "read" in step_text:
            steps.append({"skill": "read_sensor", "args": {"sensor": "indicator"}})
        
        # Add press_button if any "press" in steps
        if "press" in step_text:
            steps.append({"skill": "press_button", "args": {"button": "reset"}})
        
        # Add toggle_valve if any "valve" or "toggle" in steps
        if "valve" in step_text or "toggle" in step_text:
            steps.append({"skill": "toggle_valve", "args": {"valve": "A_inlet", "position": "open"}})
        
        # Add pick/place if any "pick" or "place" in steps
        if "pick" in step_text:
            steps.append({"skill": "pick", "args": {"object": "tool"}})
        if "place" in step_text:
            steps.append({"skill": "place", "args": {"object": "tool", "location": "shelf"}})
        
        # Add wait if any "wait" in steps
        if "wait" in step_text:
            steps.append({"skill": "wait", "args": {"seconds": 3}})
        
        # Always end with notify
        steps.append({"skill": "notify", "args": {"level": "tech"}})
        
        # Ensure we have at least 2 steps
        if len(steps) < 2:
            steps = [
                {"skill": "walk_to", "args": {"target": "machine"}},
                {"skill": "notify", "args": {"level": "tech"}},
            ]
        
        target = {
            "goal": sop.title,
            "steps": steps,
            "fallback": [],
        }
        pairs.append((instr, json.dumps(target)))
    return pairs


def train_lora(args: argparse.Namespace) -> None:
    sops = load_jsonl(args.train_jsonl, SOPEntry)
    pairs = synthesize_pairs(sops)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import Dataset
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType
        import torch

        base = args.base
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForSeq2SeqLM.from_pretrained(base)
        lora = LoraConfig(
            r=args.r,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules=["q", "v", "o"],
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora)
        ds = Dataset.from_dict({"input_text": [x for x, _ in pairs], "labels": [y for _, y in pairs]})

        def preprocess(batch):
            model_inputs = tok(batch["input_text"], truncation=True, padding=True, max_length=512)
            with tok.as_target_tokenizer():
                labels = tok(batch["labels"], truncation=True, padding=True, max_length=256)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        ds = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
        collator = DataCollatorForSeq2Seq(tok, model=model)
        tr_args = TrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            save_strategy="no",
            logging_steps=1,
        )
        trainer = Trainer(model=model, args=tr_args, train_dataset=ds, data_collator=collator, tokenizer=tok)
        console.log(f"[green]Starting training with config:[/green]")
        console.log(f"  Model: {base}")
        console.log(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.learning_rate}")
        console.log(f"  LoRA: r={args.r}, alpha={args.alpha}, dropout={args.dropout}")
        trainer.train()
        # Save adapter weights and tokenizer
        model.save_pretrained(str(out_dir / "adapter"))
        tok.save_pretrained(str(out_dir / "tokenizer"))
        console.log(f"[green]✓ Saved LoRA adapter to {out_dir}[/green]")
    except Exception as e:
        console.log(f"[yellow]Skipping LoRA training due to: {e}[/yellow]")
        # Produce a tiny placeholder so inference can proceed
        (out_dir / "adapter").mkdir(parents=True, exist_ok=True)
        (out_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
        with (out_dir / "adapter" / "placeholder.json").open("w", encoding="utf-8") as f:
            json.dump({"note": "placeholder adapter"}, f)


def build_parser():
    """Build argument parser with config file support."""
    # Get the project root directory (three levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    
    p = argparse.ArgumentParser(description="Train planner (Flan-T5) with LoRA on synthesized pairs.")
    
    # Config file argument (takes precedence)
    p.add_argument(
        "--config",
        type=str,
        default=str(project_root / "config" / "planner_config.yaml"),
        help="Path to YAML config file"
    )
    
    # CLI arguments (override config file values)
    p.add_argument("--base", type=str, help="Base model name (overrides config)")
    p.add_argument("--train_jsonl", type=str, help="Path to training data (overrides config)")
    p.add_argument("--out_dir", type=str, help="Output directory (overrides config)")
    p.add_argument("--r", type=int, help="LoRA rank (overrides config)")
    p.add_argument("--alpha", type=int, help="LoRA alpha (overrides config)")
    p.add_argument("--dropout", type=float, help="LoRA dropout (overrides config)")
    p.add_argument("--epochs", type=int, help="Number of training epochs (overrides config)")
    p.add_argument("--learning-rate", type=float, dest="learning_rate", help="Learning rate (overrides config)")
    p.add_argument("--batch-size", type=int, dest="batch_size", help="Batch size (overrides config)")
    
    return p


def merge_config(cli_args: argparse.Namespace, config_dict: dict, project_root: Path) -> argparse.Namespace:
    """Merge CLI arguments with config file, prioritizing CLI args."""
    # Extract values in priority order: CLI args → config → hardcoded defaults
    merged = argparse.Namespace()
    
    # Get nested config values
    training_cfg = config_dict.get('training', {})
    lora_cfg = config_dict.get('lora', {})
    model_cfg = config_dict.get('model', {})
    paths_cfg = config_dict.get('paths', {})
    
    # Base model
    merged.base = (
        cli_args.base or
        model_cfg.get('base') or
        "google/flan-t5-base"
    )
    
    # Training hyperparameters
    merged.epochs = (
        cli_args.epochs or
        training_cfg.get('epochs') or
        1
    )
    merged.learning_rate = (
        cli_args.learning_rate or
        training_cfg.get('learning_rate') or
        2e-4
    )
    merged.batch_size = (
        cli_args.batch_size or
        training_cfg.get('batch_size') or
        2
    )
    
    # LoRA hyperparameters
    merged.r = (
        cli_args.r or
        lora_cfg.get('r') or
        16
    )
    merged.alpha = (
        cli_args.alpha or
        lora_cfg.get('alpha') or
        32
    )
    merged.dropout = (
        cli_args.dropout or
        lora_cfg.get('dropout') or
        0.05
    )
    
    # Paths (resolve relative paths)
    merged.train_jsonl = (
        cli_args.train_jsonl or
        paths_cfg.get('train_data') or
        str(project_root / "src" / "data" / "sop_examples.jsonl")
    )
    merged.out_dir = (
        cli_args.out_dir or
        paths_cfg.get('output_dir') or
        str(project_root / "artifacts" / "planner_lora")
    )
    
    # Resolve relative paths
    if not Path(merged.train_jsonl).is_absolute():
        merged.train_jsonl = str(project_root / merged.train_jsonl)
    if not Path(merged.out_dir).is_absolute():
        merged.out_dir = str(project_root / merged.out_dir)
    
    return merged


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    cli_args = build_parser().parse_args()
    
    # Load config from YAML file
    config_dict = load_config(cli_args.config)
    
    # Merge CLI args with config file (CLI takes precedence)
    args = merge_config(cli_args, config_dict, project_root)
    
    console.log(f"[cyan]Loaded config from: {cli_args.config}[/cyan]")
    train_lora(args)


