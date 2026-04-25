from __future__ import annotations

from src.data.schemas import SKILLS, Plan
from src.planner.infer_plan import parse_plan_from_model_output


def test_parses_clean_json():
    text = '{"goal":"Reset","steps":[{"skill":"press_button","args":{"button":"green_button"}}],"fallback":[]}'
    plan = parse_plan_from_model_output(text)
    assert isinstance(plan, Plan)
    assert plan.steps[0].skill == "press_button"


def test_repairs_single_quotes_and_trailing_commas():
    text = "{'goal':'Reset','steps':[{'skill':'press_button','args':{'button':'green_button'}},],}"
    plan = parse_plan_from_model_output(text)
    assert plan.steps[0].skill == "press_button"


def test_strips_leading_prose_and_code_fence():
    text = "Here you go:\n```json\n{\"goal\":\"x\",\"steps\":[{\"skill\":\"notify\",\"args\":{\"level\":\"tech\"}}]}\n```"
    plan = parse_plan_from_model_output(text)
    assert plan.steps[0].skill == "notify"


def test_unknown_skill_is_replaced_with_notify():
    text = '{"goal":"x","steps":[{"skill":"frobnicate","args":{}}]}'
    plan = parse_plan_from_model_output(text)
    assert plan.steps[0].skill == "notify"


def test_garbage_input_returns_notify_only_plan():
    plan = parse_plan_from_model_output("model went off the rails")
    assert plan.steps[0].skill in SKILLS
    assert plan.steps[0].skill == "notify"


def test_truncated_json_is_repaired():
    # Cut off mid-object; json-repair should still recover.
    text = '{"goal":"Reset","steps":[{"skill":"walk_to","args":{"target":"machine"}'
    plan = parse_plan_from_model_output(text)
    assert plan.steps[0].skill == "walk_to"
