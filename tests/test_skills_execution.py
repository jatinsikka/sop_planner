"""Verify the dummy skill API accepts the JSON arg shapes the planner produces."""
from __future__ import annotations

from src.data.schemas import Plan, PlanStep
from src.pipeline.skills import DummyMuJoCoAdapter, execute_plan


def test_execute_plan_full_skill_set():
    plan = Plan(
        goal="exercise every skill",
        steps=[
            PlanStep(skill="walk_to", args={"target": "machine"}),
            PlanStep(skill="read_sensor", args={"sensor": "pressure_sensor"}),
            PlanStep(skill="press_button", args={"button": "green_button"}),
            PlanStep(skill="wait", args={"seconds": 2}),
            PlanStep(skill="pick", args={"object": "wrench"}),
            PlanStep(skill="place", args={"object": "wrench", "location": "table"}),
            PlanStep(skill="notify", args={"level": "tech"}),
        ],
    )
    result = execute_plan(plan, DummyMuJoCoAdapter())
    assert result == {"success": True, "failed_step": None}


def test_execute_plan_reports_bad_args():
    plan = Plan(
        goal="bad",
        steps=[PlanStep(skill="press_button", args={"colour": "blue"})],
    )
    result = execute_plan(plan, DummyMuJoCoAdapter())
    assert result["success"] is False
    assert result["failed_step"] == 0
