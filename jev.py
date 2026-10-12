"""
jev.py - TypeSafe Jev decision model integration via OpenRouter Decisions API.

Jev is a System One structured decision model designed specifically for routing,
classification, filtering, and structured decision making without generative text overhead.
Uses OpenRouter's Decisions endpoint: POST https://openrouter.ai/api/alpha/decisions
"""
import os
import time
from typing import Any, Dict, Optional
import requests
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

POLICY_FILE = Path(__file__).resolve().parent / "hopscotch_policy.md"
POLICY = POLICY_FILE.read_text(encoding="utf-8") if POLICY_FILE.exists() else ""

OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_JEV_MODEL = "typesafe/jev-1.13"

# Pricing per 1M tokens ($)
JEV_INPUT_COST_PER_M = 0.042
JEV_OUTPUT_COST_PER_M = 0.0

GPT4O_MINI_INPUT_COST_PER_M = 0.15
GPT4O_MINI_OUTPUT_COST_PER_M = 0.60

JEV_TRIAGE_QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "Classify the customer request described in `ticket`.",
        "criteria": {
            "genuine_defect": (
                "Manufacturing fault under normal use (covered for up to 90 days from delivery, even if worn or washed as per care label). "
                "Examples: sole peeling or separating, seams/stitching coming apart, zip or button failing, colour bleeding, print cracking, flaws on arrival."
            ),
            "policy_violation": (
                "Standard return or refund request that is outside standard return rules (item has been worn, washed, had tags removed, or is after 30 days, "
                "including wrong size or change of mind after wear), OR damage caused by misuse, accidents, rough use, or ignoring care label."
            ),
            "unclear": (
                "Facts in `ticket` are unclear, missing, or conflicting, including cases where it is uncertain whether damage is a manufacturing defect or customer misuse "
                "(e.g., customer is unsure how it was washed or what caused the issue)."
            ),
            "other": (
                "General customer service inquiries that do not claim a product defect or request a return/refund for a purchased item "
                "(e.g., delivery areas/shipping queries, pre-purchase sizing advice, stock checks, or greetings)."
            ),
        },
    },
    "priority": {
        "type": "score",
        "instructions": "Determine the urgency and priority level of the customer request.",
        "criteria": [
            "Low: General inquiry, shipping query, pre-purchase sizing questions, or non-urgent general requests",
            "Medium: Standard return requests, general defect inquiries, or return questions without child safety risk",
            "High: Urgent safety issues: child safety concerns, small parts/choking, skin reactions, or severe manufacturing defects",
        ],
    },
    "is_safety_hazard": {
        "type": "noul",
        "instructions": (
            "Does the customer's `ticket` describe: "
            "1) Any child-safety concern (choking hazard, small parts coming loose/swallowed, physical injury, cuts, bleeding)? OR "
            "2) Any skin reaction or allergy complaint (rash, hives, itching, blister)?"
        ),
        "criteria": {
            "true": "Ticket explicitly reports a child safety concern (choking, swallowed small parts, injury) or skin reaction/allergy",
            "false": "Ticket does not report any child safety concern, injury, or skin reaction/allergy",
        },
    },
}


def estimate_tokens(text: str) -> int:
    """Heuristic approximation: ~4 characters per token."""
    return max(1, len(text) // 4)


def calculate_jev_cost(input_tokens: int) -> float:
    """Calculate cost for Jev request ($0.042 / 1M input, $0 output)."""
    return (input_tokens / 1_000_000) * JEV_INPUT_COST_PER_M


def calculate_llm_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for GPT-4o-mini request ($0.15 / 1M in, $0.60 / 1M out)."""
    in_cost = (input_tokens / 1_000_000) * GPT4O_MINI_INPUT_COST_PER_M
    out_cost = (output_tokens / 1_000_000) * GPT4O_MINI_OUTPUT_COST_PER_M
    return in_cost + out_cost


def call_jev_decisions(
    text: str,
    questions: Optional[Dict[str, Any]] = None,
    model: str = DEFAULT_JEV_MODEL,
    api_key: Optional[str] = None,
    timeout: int = 15,
) -> Dict[str, Any]:
    """Execute decision queries against OpenRouter's Decisions API."""
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    qs = questions or JEV_TRIAGE_QUESTIONS
    payload = {
        "model": model,
        "state": {
            "policy": POLICY,
            "ticket": text,
        },
        "questions": qs,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://hopscotch.in",
        "X-Title": "Hopscotch Support Decision Engine",
    }

    t0 = time.perf_counter()
    resp = requests.post(OPENROUTER_DECISIONS_URL, json=payload, headers=headers, timeout=timeout)
    elapsed = time.perf_counter() - t0

    if resp.status_code != 200:
        err_msg = f"OpenRouter Decisions API returned status {resp.status_code}: {resp.text}"
        raise RuntimeError(err_msg)

    data = resp.json()
    data["_elapsed_s"] = elapsed
    return data


PRIORITY_LEVELS = ["Low", "Medium", "High"]


def _normalize_priority(raw_val: Any) -> str:
    if isinstance(raw_val, (int, float)):
        idx = int(round(raw_val))
        if 0 <= idx < len(PRIORITY_LEVELS):
            return PRIORITY_LEVELS[idx]
        elif idx == len(PRIORITY_LEVELS):
            return PRIORITY_LEVELS[idx - 1]
    text = str(raw_val or "").lower()
    for level in ("high", "medium", "low"):
        if level in text:
            return level.capitalize()
    return "Medium"


def _normalize_category(cat_info: Any) -> tuple:
    probs = {}
    if isinstance(cat_info, dict):
        raw_ans = (
            cat_info.get("choice")
            or cat_info.get("answer")
            or cat_info.get("value")
            or ""
        )
        probs = cat_info.get("probabilities", {})
    else:
        raw_ans = str(cat_info or "")
    text = str(raw_ans).lower()
    for cat in ("genuine_defect", "policy_violation", "unclear", "other"):
        if cat in text:
            return cat, probs
    return "unclear", probs


def _normalize_safety(safety_info: Any, threshold: float = 0.5) -> tuple:
    if isinstance(safety_info, dict):
        # OpenRouter returns noul probability directly under key 'noul'
        raw_p = (
            safety_info.get("noul")
            if safety_info.get("noul") is not None
            else safety_info.get("p_yes")
            or safety_info.get("probability", 0.0)
        )
        try:
            p_yes = float(raw_p)
        except (ValueError, TypeError):
            p_yes = 0.0

        raw_ans = safety_info.get("answer")
        if isinstance(raw_ans, bool):
            is_safety = raw_ans
        elif isinstance(raw_ans, str):
            is_safety = raw_ans.lower() in ("true", "yes", "1")
        else:
            is_safety = p_yes >= threshold
        return is_safety, p_yes
    elif isinstance(safety_info, bool):
        return safety_info, 1.0 if safety_info else 0.0
    return False, 0.0


def parse_jev_response(data: Dict[str, Any], input_text: str) -> Dict[str, Any]:
    """Extract category, priority, safety flag, probabilities, and cost from Decisions response."""
    decisions = data.get("answers") or data.get("decisions") or data.get("results") or {}

    # Category extraction (Choice: key 'choice')
    cat_info = decisions.get("category", {})
    category, cat_probs = _normalize_category(cat_info)

    # Priority extraction (Score: key 'score')
    pri_info = decisions.get("priority", {})
    if isinstance(pri_info, dict):
        raw_pri = pri_info.get("score") if pri_info.get("score") is not None else (pri_info.get("answer") or pri_info.get("value"))
        pri_probs = pri_info.get("probabilities", {})
    else:
        raw_pri = pri_info
        pri_probs = {}
    priority = _normalize_priority(raw_pri)

    # Safety hazard extraction (Noul: key 'noul')
    safety_info = decisions.get("is_safety_hazard", {})
    is_safety, p_yes = _normalize_safety(safety_info)

    in_tokens = estimate_tokens((POLICY + "\n" if POLICY else "") + input_text)
    cost = calculate_jev_cost(in_tokens)

    # Prefer OpenRouter's exact usage reporting if provided
    usage = data.get("usage", {})
    if usage and usage.get("cost") is not None:
        try:
            cost = float(usage["cost"])
            in_tokens = int(usage.get("input_tokens", in_tokens))
        except (ValueError, TypeError):
            pass

    return {
        "triage": {
            "category": category,
            "priority": priority,
        },
        "safety": {
            "flag": is_safety,
            "reason": "child-safety hazard identified by Jev" if is_safety else None,
            "p_yes": p_yes,
        },
        "probabilities": {
            "category": cat_probs,
            "priority": pri_probs,
            "safety_p_yes": p_yes,
        },
        "latency_s": round(data.get("_elapsed_s", 0.0), 3),
        "tokens": in_tokens,
        "cost": cost,
        "raw": data,
    }


def build_jev_triage_runnable(model: str = DEFAULT_JEV_MODEL) -> RunnableLambda:
    """Construct an LCEL Runnable that calls TypeSafe Jev for triage decisions."""
    def _run_jev(x: Any) -> Dict[str, Any]:
        text = x if isinstance(x, str) else (x.get("input") or x.get("ticket_text") or "")
        raw = call_jev_decisions(text, model=model)
        return parse_jev_response(raw, text)

    return RunnableLambda(_run_jev)
