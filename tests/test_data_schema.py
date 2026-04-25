from __future__ import annotations

from pathlib import Path

from src.data.schemas import SKILLS, IncidentEntry, Plan, PlanStep, SOPEntry, load_json

ROOT = Path(__file__).resolve().parents[1]
SOPS_PATH = ROOT / "src" / "data" / "sop_examples.json"
INCIDENTS_PATH = ROOT / "src" / "data" / "incident_examples.json"


def test_load_sops_validates_records():
    sops = load_json(str(SOPS_PATH), SOPEntry, key="sop_examples")
    assert sops, "no SOPs loaded"
    assert all(s.sop_id and s.title and s.steps for s in sops)


def test_load_incidents_validates_records():
    incs = load_json(str(INCIDENTS_PATH), IncidentEntry, key="incident_examples")
    assert incs
    assert all(i.incident_id and i.text for i in incs)


def test_incidents_reference_existing_sops():
    sops = load_json(str(SOPS_PATH), SOPEntry, key="sop_examples")
    incs = load_json(str(INCIDENTS_PATH), IncidentEntry, key="incident_examples")
    sop_ids = {s.sop_id for s in sops}
    missing = [i.incident_id for i in incs if (i.labels or {}).get("sop_id") not in sop_ids]
    assert not missing, f"incidents reference unknown SOPs: {missing[:5]}"


def test_plan_roundtrip():
    p = Plan(goal="Test", steps=[PlanStep(skill="notify", args={"level": "tech"})])
    p2 = Plan.model_validate_json(p.model_dump_json())
    assert p2.steps[0].skill in SKILLS
