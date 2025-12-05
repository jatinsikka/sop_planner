from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from rich.console import Console

from src.data.schemas import Plan, SKILLS

console = Console()


def _repair_json(text: str) -> Optional[dict]:
    """Attempt minimal brace/quote repairs to recover a JSON object."""
    # Extract first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    cand = m.group(0)
    # Replace single quotes with double quotes conservatively
    cand = re.sub(r"'", '"', cand)
    # Remove trailing commas
    cand = re.sub(r",\s*([}\]])", r"\1", cand)
    try:
        return json.loads(cand)
    except Exception:
        return None


def parse_plan_from_model_output(text: str) -> Plan:
    """Parse model text into Plan. Repair once if invalid."""
    try:
        obj = json.loads(text)
        console.log(f"[green]✓ Direct JSON parse succeeded[/green]")
    except Exception as e1:
        console.log(f"[yellow]Direct parse failed: {e1}[/yellow]")
        console.log(f"[dim]Raw text: {text[:200]}...[/dim]")
        obj = _repair_json(text)
        if obj:
            console.log(f"[green]✓ Repair succeeded[/green]")
        else:
            console.log(f"[yellow]Repair failed, obj is None[/yellow]")
    
    if not isinstance(obj, dict):
        console.log(f"[red]✗ obj is not dict: {type(obj)} = {obj}[/red]")
        # Fallback heuristic minimal plan
        obj = {"goal": "Resolve incident", "steps": [{"skill": "notify", "args": {"level": "tech"}}], "fallback": []}
    
    try:
        plan = Plan.model_validate(obj)
    except Exception as e2:
        console.log(f"[red]✗ Plan.model_validate failed: {e2}[/red]")
        console.log(f"[dim]obj was: {obj}[/dim]")
        raise
    
    # Enforce skill whitelist
    for step in plan.steps:
        if step.skill not in SKILLS:
            step.skill = "notify"
            step.args = {"level": "tech"}
    return plan


def _build_prompt(incident: str, sop_title: str, sop_cond: str, sop_steps: List[str], skills: List[str]) -> str:
    steps = " ; ".join(sop_steps)
    return (
        "[INCIDENT]\n"
        f"{incident}\n"
        "[SOP]\n"
        f"{sop_title}\n{sop_cond}\n{steps}\n"
        "[ALLOWED_SKILLS]\n"
        f"{','.join(skills)}\n\n"
        "[INSTRUCTIONS]\n"
        "Return ONLY a single JSON object (no explanation) with the following keys:\n"
        "  - goal: string\n"
        "  - steps: list of objects with keys 'skill' (string) and 'args' (object)\n"
        "  - fallback: list\n"
        "The JSON must be strictly parseable by a JSON parser (use double quotes, no trailing commas)."
    )


def _extract_skills_from_sop(sop_steps: List[str], skills: List[str]) -> List[dict]:
    """Extract and map SOP steps to robot skills using semantic keywords."""
    extracted = []
    step_text = " ".join(sop_steps).lower()
    
    # Keywords for each skill (semantic matching)
    skill_keywords = {
        "walk_to": ["walk", "go", "move", "proceed", "head to", "navigate", "travel"],
        "read_sensor": ["read", "check", "monitor", "observe", "measure", "indicator", "sensor", "display", "gauge"],
        "press_button": ["press", "push", "click", "activate", "button", "switch", "trigger"],
        "toggle_valve": ["toggle", "valve", "turn", "open", "close", "flow", "release"],
        "pick": ["pick", "grab", "take", "collect", "retrieve", "hold"],
        "place": ["place", "put", "set", "deposit", "position"],
        "wait": ["wait", "pause", "delay", "hold"],
        "notify": ["notify", "report", "alert", "inform", "communicate"],
    }
    
    # Track which skills we've already added (avoid duplicates)
    added_skills = set()
    
    # Scan through steps and match keywords
    for step in sop_steps:
        step_lower = step.lower()
        
        for skill, keywords in skill_keywords.items():
            # Only add if skill exists, not already added, and keyword matches
            if skill in skills and skill not in added_skills:
                if any(kw in step_lower for kw in keywords):
                    # Extract contextual args
                    args = _extract_args_for_skill(skill, step)
                    extracted.append({"skill": skill, "args": args})
                    added_skills.add(skill)
                    break  # Move to next step after matching one skill
    
    return extracted


def _extract_args_for_skill(skill: str, step_text: str) -> dict:
    """Extract arguments/parameters for a skill from the step description."""
    step_lower = step_text.lower()
    
    if skill == "walk_to":
        # Try to extract target location
        if "table" in step_lower:
            return {"target": "table"}
        elif "machine" in step_lower:
            return {"target": "machine"}
        elif "shelf" in step_lower:
            return {"target": "shelf"}
        else:
            return {"target": "machine"}  # Default
    
    elif skill == "read_sensor":
        # Try to extract sensor type
        if "pressure" in step_lower:
            return {"sensor": "pressure_sensor"}
        elif "temperature" in step_lower:
            return {"sensor": "temperature_sensor"}
        elif "indicator" in step_lower or "display" in step_lower:
            return {"sensor": "indicator"}
        else:
            return {"sensor": "indicator"}  # Default
    
    elif skill == "press_button":
        # Try to extract button type
        if "blue" in step_lower:
            return {"button": "blue_diagnostic_button"}
        elif "green" in step_lower:
            return {"button": "green_reset"}
        elif "red" in step_lower:
            return {"button": "red_emergency"}
        else:
            return {"button": "reset"}  # Default
    
    elif skill == "toggle_valve":
        # Try to extract valve and position
        position = "open" if "open" in step_lower else ("close" if "close" in step_lower else "toggle")
        valve = "A_inlet" if "inlet" in step_lower else ("B_outlet" if "outlet" in step_lower else "A_inlet")
        return {"valve": valve, "position": position}
    
    elif skill == "pick":
        # Try to extract object
        if "screw" in step_lower:
            return {"object": "screwdriver"}
        elif "wrench" in step_lower:
            return {"object": "wrench"}
        elif "tool" in step_lower:
            return {"object": "tool"}
        else:
            return {"object": "tool"}  # Default
    
    elif skill == "place":
        # Try to extract location
        if "shelf" in step_lower:
            return {"object": "tool", "location": "machine_shelf"}
        elif "table" in step_lower:
            return {"object": "tool", "location": "table"}
        else:
            return {"object": "tool", "location": "shelf"}  # Default
    
    elif skill == "wait":
        # Extract wait duration
        return {"seconds": 3}
    
    elif skill == "notify":
        # Extract notification level
        if "manager" in step_lower or "supervisor" in step_lower:
            return {"level": "manager"}
        elif "tech" in step_lower or "technician" in step_lower:
            return {"level": "tech"}
        else:
            return {"level": "tech"}  # Default
    
    return {}


def plan(incident: str, sop: dict, skills: List[str] = SKILLS) -> Plan:
    """Generate a plan from (incident, sop) using semantic skill extraction."""
    console.log(f"[dim]Planning for: {sop['title']}[/dim]")
    
    goal = sop["title"]
    
    # Extract skills from SOP steps using semantic matching
    steps = _extract_skills_from_sop(sop.get("steps", []), skills)
    
    # Always end with notify if not already present
    if not steps or steps[-1]["skill"] != "notify":
        steps.append({"skill": "notify", "args": {"level": "tech"}})
    
    # Ensure at least one action before notify
    if len(steps) == 1:
        steps.insert(0, {"skill": "walk_to", "args": {"target": "machine"}})
    
    plan_obj = Plan(goal=goal, steps=steps, fallback=[])
    console.log(f"[green]✓ Generated plan: {len(plan_obj.steps)} steps[/green]")
    for i, step in enumerate(plan_obj.steps, 1):
        console.log(f"[dim]  {i}. {step.skill} → {step.args}[/dim]")
    
    return plan_obj


