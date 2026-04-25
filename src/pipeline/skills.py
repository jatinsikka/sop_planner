"""Skill API surface for the planner output.

Method signatures match the keys produced by `train_planner_lora.synthesize_target`
so `execute_plan` can dispatch via `method(**args)` with no shimming.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from rich.console import Console

from src.data.schemas import Plan

console = Console()


class SkillAPI(ABC):
    @abstractmethod
    def walk_to(self, target: str) -> bool: ...

    @abstractmethod
    def press_button(self, button: str) -> bool: ...

    @abstractmethod
    def wait(self, seconds: float) -> bool: ...

    @abstractmethod
    def read_sensor(self, sensor: str, expect: str | None = None) -> dict: ...

    @abstractmethod
    def pick(self, object: str) -> bool: ...  # noqa: A002 — matches plan arg key

    @abstractmethod
    def place(self, object: str, location: str) -> bool: ...  # noqa: A002

    @abstractmethod
    def notify(self, level: str = "tech") -> bool: ...


class DummyMuJoCoAdapter(SkillAPI):
    """In-process stub used for tests and the `demo exec` command."""

    def walk_to(self, target: str) -> bool:
        console.log(f"walk_to: {target}")
        return True

    def press_button(self, button: str) -> bool:
        console.log(f"press_button: {button}")
        return True

    def wait(self, seconds: float) -> bool:
        console.log(f"wait: {seconds}s")
        return True

    def read_sensor(self, sensor: str, expect: str | None = None) -> dict:
        console.log(f"read_sensor: {sensor}, expect={expect}")
        return {"sensor": sensor, "value": "ok"}

    def pick(self, object: str) -> bool:  # noqa: A002
        console.log(f"pick: {object}")
        return True

    def place(self, object: str, location: str) -> bool:  # noqa: A002
        console.log(f"place: {object} → {location}")
        return True

    def notify(self, level: str = "tech") -> bool:
        console.log(f"notify: {level}")
        return True


def execute_plan(plan: Plan, api: SkillAPI) -> Dict[str, object]:
    """Run plan steps against `api`, returning {success, failed_step}."""
    for i, step in enumerate(plan.steps):
        method = getattr(api, step.skill, None)
        if method is None:
            console.log(f"[red]Unknown skill: {step.skill}[/red]")
            return {"success": False, "failed_step": i}
        try:
            ok = method(**step.args)
        except TypeError as exc:
            console.log(f"[red]Bad args for {step.skill}: {step.args} ({exc})[/red]")
            return {"success": False, "failed_step": i}
        if ok is False:
            return {"success": False, "failed_step": i}
    return {"success": True, "failed_step": None}
