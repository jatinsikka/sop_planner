"""Fine-tune Qwen2.5-0.5B-Instruct with LoRA to emit structured plan JSON.

Synthesises one (instruction, plan_json) pair per SOP and trains a causal-LM
LoRA adapter. The base model has 494M params; LoRA trains <1% of that.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schemas import SKILLS, SOPEntry, load_json

console = Console()

DEFAULT_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = (
    "You are a robot task planner. Given an incident description and a Standard "
    "Operating Procedure, output a single JSON object with keys 'goal' (string), "
    "'steps' (list of {skill, args}), and 'fallback' (list, may be empty). The "
    f"only allowed skill values are: {', '.join(SKILLS)}. Output JSON only — no "
    "prose, no Markdown."
)


def build_user_prompt(incident: str, sop: SOPEntry) -> str:
    steps = " ; ".join(sop.steps)
    return (
        f"[INCIDENT]\n{incident}\n\n"
        f"[SOP]\nTitle: {sop.title}\nCondition: {sop.condition}\nSteps: {steps}\n\n"
        f"[ALLOWED_SKILLS]\n{','.join(SKILLS)}"
    )


def _extract_arg(skill: str, step_text: str) -> dict:
    s = step_text.lower()
    if skill == "walk_to":
        for tgt in ("table", "shelf", "machine"):
            if tgt in s:
                return {"target": tgt}
        return {"target": "machine"}
    if skill == "read_sensor":
        for kind in ("temperature", "vibration", "light", "pressure"):
            if kind in s:
                return {"sensor": f"{kind}_sensor"}
        return {"sensor": "pressure_sensor"}
    if skill == "press_button":
        for color in ("red", "green", "blue", "yellow"):
            if color in s:
                return {"button": f"{color}_button"}
        return {"button": "green_button"}
    if skill == "pick":
        for obj in (
            "wrench", "screwdriver", "battery", "lubricant", "cloth",
            "fuse", "brush", "goggles", "cable", "container",
        ):
            if obj in s:
                return {"object": obj}
        return {"object": "tool"}
    if skill == "place":
        loc = "machine_shelf" if "shelf" in s else "table"
        return {"object": "tool", "location": loc}
    if skill == "wait":
        return {"seconds": 3}
    if skill == "notify":
        level = "manager" if any(k in s for k in ("manager", "supervisor")) else "tech"
        return {"level": level}
    return {}


def synthesize_target(sop: SOPEntry) -> dict:
    """Heuristic mapping from SOP steps to a structured plan (used as labels)."""
    steps = []
    for raw_step in sop.steps:
        s = raw_step.lower()
        if any(k in s for k in ("walk", "go", "navigate", "proceed", "move")):
            steps.append({"skill": "walk_to", "args": _extract_arg("walk_to", raw_step)})
        elif any(k in s for k in ("press", "button", "push")):
            steps.append({"skill": "press_button", "args": _extract_arg("press_button", raw_step)})
        elif any(k in s for k in ("read", "sensor", "check", "monitor", "measure", "verify")):
            steps.append({"skill": "read_sensor", "args": _extract_arg("read_sensor", raw_step)})
        elif "pick" in s or "grab" in s or "take" in s:
            steps.append({"skill": "pick", "args": _extract_arg("pick", raw_step)})
        elif "place" in s or "return" in s or "put" in s:
            steps.append({"skill": "place", "args": _extract_arg("place", raw_step)})
        elif "wait" in s:
            steps.append({"skill": "wait", "args": _extract_arg("wait", raw_step)})
        elif "notify" in s or "alert" in s or "report" in s:
            steps.append({"skill": "notify", "args": _extract_arg("notify", raw_step)})
    if not steps or steps[-1]["skill"] != "notify":
        steps.append({"skill": "notify", "args": {"level": "tech"}})
    return {"goal": sop.title, "steps": steps, "fallback": []}


def synthesize_pairs(sops: List[SOPEntry]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for sop in sops:
        incident = f"{sop.title}. {sop.condition}"
        prompt = build_user_prompt(incident, sop)
        target = json.dumps(synthesize_target(sop), separators=(",", ":"))
        pairs.append((prompt, target))
    return pairs


def _format_chat(tokenizer, prompt: str, target: str | None) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=target is None,
    )


def train(args: argparse.Namespace) -> None:
    sops = load_json(args.sops_path, SOPEntry, key="sop_examples")
    pairs = synthesize_pairs(sops)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    console.log(f"[bold]Loading {args.base}[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base)

    lora = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def to_text(batch):
        return {"text": [_format_chat(tokenizer, p, t) for p, t in zip(batch["prompt"], batch["target"])]}

    ds = Dataset.from_dict({"prompt": [p for p, _ in pairs], "target": [t for _, t in pairs]})
    ds = ds.map(to_text, batched=True, remove_columns=ds.column_names)

    def tokenize(batch):
        out = tokenizer(batch["text"], truncation=True, padding=False, max_length=args.max_length)
        return out

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        save_strategy="no",
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
    )

    console.log(f"[green]Training: {len(pairs)} pairs, {args.epochs} epochs, batch {args.batch_size}, lr {args.lr}[/green]")
    trainer.train()

    adapter_dir = out_dir / "adapter"
    tokenizer_dir = out_dir / "tokenizer"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(tokenizer_dir))

    meta = {"base": args.base, "skills": SKILLS}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    console.log(f"[bold green]✓ Saved adapter to {adapter_dir}[/bold green]")


def load_config() -> dict:
    cfg_path = PROJECT_ROOT / "config" / "planner_config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(cfg_path.read_text()) or {}
    except ImportError:
        return {}


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    lora_cfg = cfg.get("lora", {})
    paths_cfg = cfg.get("paths", {})

    p = argparse.ArgumentParser(description="Train Qwen2.5 + LoRA planner.")
    p.add_argument("--base", type=str, default=model_cfg.get("base", DEFAULT_BASE))
    p.add_argument("--sops_path", type=str, default=str(PROJECT_ROOT / paths_cfg.get("train_data", "src/data/sop_examples.json")))
    p.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / paths_cfg.get("output_dir", "artifacts/planner_lora")))
    p.add_argument("--epochs", type=int, default=train_cfg.get("epochs", 3))
    p.add_argument("--batch_size", type=int, default=train_cfg.get("batch_size", 2))
    p.add_argument("--lr", type=float, default=train_cfg.get("learning_rate", 2e-4))
    p.add_argument("--max_length", type=int, default=train_cfg.get("max_input_length", 768))
    p.add_argument("--r", type=int, default=lora_cfg.get("r", 16))
    p.add_argument("--alpha", type=int, default=lora_cfg.get("alpha", 32))
    p.add_argument("--dropout", type=float, default=lora_cfg.get("dropout", 0.05))
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
