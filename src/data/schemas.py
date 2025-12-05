from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError

# Allowed skills across the project
SKILLS: List[str] = [
    "walk_to",
    "toggle_valve",
    "press_button",
    "wait",
    "read_sensor",
    "pick",
    "place",
    "notify",
]


class SOPEntry(BaseModel):
    """Standard Operating Procedure entry."""

    sop_id: str
    title: str
    condition: str
    steps: List[str]
    equipment: List[str] = Field(default_factory=list)


class IncidentEntry(BaseModel):
    """Incident/Query entry."""

    incident_id: str
    text: str
    labels: Optional[Dict[str, str]] = None


class PlanStep(BaseModel):
    """One step in the high-level plan."""

    skill: str
    args: Dict[str, Any]


class Plan(BaseModel):
    """Structured plan for execution."""

    goal: str
    steps: List[PlanStep]
    fallback: List[PlanStep] = Field(default_factory=list)


T = TypeVar("T", bound=BaseModel)


def load_jsonl(path: str | Path, model: Type[T]) -> List[T]:
    """Load a JSONL file and validate each line with the given Pydantic model.

    Args:
        path: Path to JSONL file.
        model: Pydantic model class to validate each record.
    Returns:
        List of validated model instances.
    """
    import json

    data: List[T] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            try:
                data.append(model.model_validate(obj))
            except ValidationError as e:
                raise ValueError(f"Invalid record in {p}: {e}") from e
    return data


