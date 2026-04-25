"""Inference for the SOP planner.

Loads the Qwen2.5 + LoRA adapter when present and generates a JSON plan;
falls back to a deterministic heuristic plan derived from the SOP steps when
the model is unavailable. JSON repair uses the `json_repair` library.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from rich.console import Console

from src.data.schemas import Plan, PlanStep, SKILLS

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "artifacts" / "planner_lora" / "adapter"
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "artifacts" / "planner_lora" / "tokenizer"
DEFAULT_META_PATH = PROJECT_ROOT / "artifacts" / "planner_lora" / "meta.json"
DEFAULT_BASE = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a robot task planner. Given an incident description and a Standard "
    "Operating Procedure, output a single JSON object with keys 'goal' (string), "
    "'steps' (list of {skill, args}), and 'fallback' (list, may be empty). The "
    f"only allowed skill values are: {', '.join(SKILLS)}. Output JSON only — no "
    "prose, no Markdown."
)

_model = None
_tokenizer = None


def _build_user_prompt(incident: str, sop: dict) -> str:
    steps = " ; ".join(sop.get("steps", []))
    return (
        f"[INCIDENT]\n{incident}\n\n"
        f"[SOP]\nTitle: {sop.get('title', '')}\nCondition: {sop.get('condition', '')}\nSteps: {steps}\n\n"
        f"[ALLOWED_SKILLS]\n{','.join(SKILLS)}"
    )


def _try_repair(text: str) -> Optional[dict]:
    """Recover a JSON object from messy LLM output."""
    try:
        from json_repair import repair_json

        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
        if isinstance(repaired, list) and repaired and isinstance(repaired[0], dict):
            return repaired[0]
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def parse_plan_from_model_output(text: str) -> Plan:
    """Parse model text into a validated Plan; never raises."""
    obj: Optional[dict]
    try:
        obj = json.loads(text)
    except Exception:
        obj = _try_repair(text)

    if not isinstance(obj, dict):
        obj = {"goal": "Resolve incident", "steps": [{"skill": "notify", "args": {"level": "tech"}}], "fallback": []}

    obj.setdefault("goal", "Resolve incident")
    obj.setdefault("steps", [])
    obj.setdefault("fallback", [])

    cleaned_steps = []
    for raw in obj.get("steps", []):
        if not isinstance(raw, dict):
            continue
        skill = raw.get("skill")
        args = raw.get("args") if isinstance(raw.get("args"), dict) else {}
        if skill not in SKILLS:
            skill = "notify"
            args = {"level": "tech"}
        cleaned_steps.append({"skill": skill, "args": args})
    if not cleaned_steps:
        cleaned_steps.append({"skill": "notify", "args": {"level": "tech"}})
    obj["steps"] = cleaned_steps

    cleaned_fallback = []
    for raw in obj.get("fallback", []):
        if isinstance(raw, dict) and raw.get("skill") in SKILLS:
            cleaned_fallback.append({"skill": raw["skill"], "args": raw.get("args", {}) if isinstance(raw.get("args"), dict) else {}})
    obj["fallback"] = cleaned_fallback

    return Plan.model_validate(obj)


def _heuristic_plan(sop: dict) -> Plan:
    """Deterministic baseline that maps SOP step text to robot skills."""
    from src.planner.train_planner_lora import synthesize_target
    from src.data.schemas import SOPEntry

    sop_entry = SOPEntry.model_validate(
        {
            "sop_id": sop.get("sop_id", ""),
            "title": sop.get("title", "Resolve incident"),
            "condition": sop.get("condition", ""),
            "steps": sop.get("steps", []),
            "equipment": sop.get("equipment", []),
        }
    )
    target = synthesize_target(sop_entry)
    return Plan.model_validate(target)


def _adapter_available(adapter_dir: Path) -> bool:
    return adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists()


def _load_model(adapter_dir: Path = DEFAULT_ADAPTER_DIR, tokenizer_dir: Path = DEFAULT_TOKENIZER_DIR) -> Tuple[object, object]:
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = DEFAULT_BASE
    if DEFAULT_META_PATH.exists():
        try:
            base = json.loads(DEFAULT_META_PATH.read_text()).get("base", DEFAULT_BASE)
        except Exception:
            pass

    console.log(f"[cyan]Loading planner: base={base}, adapter={adapter_dir}[/cyan]")
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(base)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    _model, _tokenizer = model, tok
    return model, tok


def _generate(model, tokenizer, incident: str, sop: dict, max_new_tokens: int = 256) -> str:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(incident, sop)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def plan(
    incident: str,
    sop: dict,
    skills: List[str] = SKILLS,
    adapter_dir: Optional[Path] = None,
    tokenizer_dir: Optional[Path] = None,
) -> Plan:
    """Return a Plan for (incident, sop). Uses LoRA model if available, else heuristic."""
    adapter_dir = Path(adapter_dir) if adapter_dir else DEFAULT_ADAPTER_DIR
    tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else DEFAULT_TOKENIZER_DIR

    if _adapter_available(adapter_dir) and tokenizer_dir.exists():
        try:
            model, tokenizer = _load_model(adapter_dir, tokenizer_dir)
            text = _generate(model, tokenizer, incident, sop)
            console.log(f"[dim]Generated: {text[:160]}...[/dim]")
            plan_obj = parse_plan_from_model_output(text)
        except Exception as exc:
            console.log(f"[yellow]LoRA generation failed ({exc}); falling back to heuristic[/yellow]")
            plan_obj = _heuristic_plan(sop)
    else:
        console.log("[yellow]No trained adapter; using heuristic planner[/yellow]")
        plan_obj = _heuristic_plan(sop)

    if plan_obj.steps and plan_obj.steps[-1].skill != "notify":
        plan_obj.steps.append(PlanStep(skill="notify", args={"level": "tech"}))

    console.log(f"[green]✓ Plan: {len(plan_obj.steps)} steps[/green]")
    for i, step in enumerate(plan_obj.steps, 1):
        console.log(f"[dim]  {i}. {step.skill} → {step.args}[/dim]")
    return plan_obj
