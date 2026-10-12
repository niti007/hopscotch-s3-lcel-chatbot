import os
import sys
from unittest.mock import patch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chains import build_full_chain_jev, build_signals_jev, decide_route, explain_route_jev
from jev import (
    DEFAULT_JEV_MODEL,
    calculate_jev_cost,
    calculate_llm_cost,
    estimate_tokens,
    parse_jev_response,
)
from tests.test_pipeline import FakeLLM


def test_cost_calculations():
    # 1,000,000 tokens of Jev input
    assert calculate_jev_cost(1_000_000) == pytest.approx(0.042)
    # 100 tokens of Jev input
    assert calculate_jev_cost(100) == pytest.approx(0.0000042)

    # 1,000,000 tokens in + 1,000,000 tokens out for GPT-4o-mini
    assert calculate_llm_cost(1_000_000, 1_000_000) == pytest.approx(0.75)


def test_parse_jev_response():
    mock_api_data = {
        "decisions": {
            "category": {
                "answer": "genuine_defect",
                "probabilities": {"genuine_defect": 0.95, "policy_violation": 0.03, "unclear": 0.02},
            },
            "priority": {
                "answer": "High",
                "probabilities": {"Low": 0.05, "Medium": 0.15, "High": 0.80},
            },
            "is_safety_hazard": {
                "answer": False,
                "p_yes": 0.02,
            },
        },
        "_elapsed_s": 0.12,
    }
    input_text = "The sole peeled off after 3 wears."
    parsed = parse_jev_response(mock_api_data, input_text)

    assert parsed["triage"]["category"] == "genuine_defect"
    assert parsed["triage"]["priority"] == "High"
    assert parsed["safety"]["flag"] is False
    assert parsed["probabilities"]["category"]["genuine_defect"] == 0.95
    assert parsed["probabilities"]["priority"]["High"] == 0.80
    assert parsed["latency_s"] == 0.12
    assert parsed["cost"] > 0


def test_parse_openrouter_native_answers():
    """Verify parsing of OpenRouter's exact native Decisions API response structure."""
    native_data = {
        "answers": {
            "category": {
                "type": "choice",
                "choice": "genuine_defect",
                "probabilities": {"genuine_defect": 0.92, "policy_violation": 0.05, "unclear": 0.03},
            },
            "priority": {
                "type": "score",
                "score": 1.9,
                "probabilities": {"Low": 0.05, "Medium": 0.2, "High": 0.75},
            },
            "is_safety_hazard": {
                "type": "noul",
                "noul": 0.015,
            },
        },
        "usage": {
            "input_tokens": 45,
            "output_tokens": 0,
            "cost": 0.00000189,
        },
        "_elapsed_s": 0.08,
    }
    parsed = parse_jev_response(native_data, "The sole of my shoe peeled off.")
    assert parsed["triage"]["category"] == "genuine_defect"
    assert parsed["triage"]["priority"] == "High"
    assert parsed["safety"]["flag"] is False
    assert parsed["safety"]["p_yes"] == 0.015
    assert parsed["cost"] == 0.00000189


def test_jev_safety_flag_triggers_human_review():
    mock_api_data = {
        "decisions": {
            "category": {"answer": "other"},
            "priority": {"answer": "High"},
            "is_safety_hazard": {"answer": True, "p_yes": 0.98},
        },
        "_elapsed_s": 0.15,
    }
    with patch("jev.call_jev_decisions", return_value=mock_api_data):
        signals = build_signals_jev().invoke("My baby choked on a small button.")
        assert signals["safety"]["flag"] is True
        decision = decide_route({"input": "My baby choked on a small button.", "signals": signals})
        assert decision["route"] == "human_review"
        assert decision["needs_human"] is True


def test_build_full_chain_jev_with_fake_llm():
    mock_api_data = {
        "decisions": {
            "category": {"answer": "genuine_defect"},
            "priority": {"answer": "High"},
            "is_safety_hazard": {"answer": False, "p_yes": 0.01},
        },
        "_elapsed_s": 0.10,
    }
    fake_llm = FakeLLM()
    with patch("jev.call_jev_decisions", return_value=mock_api_data):
        chain = build_full_chain_jev(fake_llm)
        reply = chain.invoke({"input": "The zipper broke on day 2."})
        assert reply == "REPLY"
        # Verify FakeLLM received the DEFECT route prompt
        assert "[ROUTE: GENUINE DEFECT]" in fake_llm.seen_system[-1]


def test_explain_route_jev():
    mock_api_data = {
        "decisions": {
            "category": {"answer": "policy_violation"},
            "priority": {"answer": "Low"},
            "is_safety_hazard": {"answer": False, "p_yes": 0.01},
        },
        "_elapsed_s": 0.09,
    }
    with patch("jev.call_jev_decisions", return_value=mock_api_data):
        res = explain_route_jev("I wore it for 2 months and want a return.")
        assert res["route"] == "policy"
        assert res["category"] == "policy_violation"
        assert res["priority"] == "Low"
        assert "jev_metadata" in res
