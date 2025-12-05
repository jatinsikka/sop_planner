from __future__ import annotations

from src.data.schemas import IncidentEntry, Plan, PlanStep, SOPEntry, load_jsonl, SKILLS


def test_load_schemas():
    sops = load_jsonl("src/data/sop_examples.jsonl", SOPEntry)
    incs = load_jsonl("src/data/incident_examples.jsonl", IncidentEntry)
    assert len(sops) == 10
    assert len(incs) == 10
    assert isinstance(sops[0].steps, list)
    assert isinstance(incs[0].text, str)


def test_plan_roundtrip():
    p = Plan(goal="Test", steps=[PlanStep(skill="notify", args={"level": "tech"})], fallback=[])
    s = p.model_dump_json()
    p2 = Plan.model_validate_json(s)
    assert p2.goal == "Test"
    assert p2.steps[0].skill in SKILLS


