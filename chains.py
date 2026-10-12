"""
chains.py - the Session 3 LCEL pipeline for the Hopscotch support bot.

Everything here is a *Runnable*: it has .invoke() / .batch() / .stream() and can be
piped with `|`.  The big idea is that the whole triage-and-reply flow is ONE declarative object
instead of a Python function calling chains in order.

                      {"input": str, "chat_history": [...]}
                                     |
                                     v
        RunnablePassthrough.assign(signals = text | RunnableParallel)
                                     |
            +------------+-----------+-----------+------------+
            | triage     | safety    | order_    | order_id   |   <- run CONCURRENTLY
            | (LLM call) | (keywords)| value     | (regex)    |
            +------------+-----------+-----------+------------+
                                     |   adds key "signals"
                                     v
                       RunnableLambda(decide_route)
                       adds needs_human, human_reasons,
                       route, triage_summary
                                     |
                                     v
                             RunnableBranch
        +--------------+--------------+----------------+-------------+
        | human_review | defect       | policy         | general     |
        | ESCALATION   | DEFECT       | POLICY_REPLY   | GENERAL     |
        | | llm | str  | | llm | str  | | llm | str    | | llm | str |
        +--------------+--------------+----------------+-------------+
                                     |
                                     v
                                reply (str)
"""
import re
import time
from typing import Optional

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from prompts import (
    DEFECT_PROMPT,
    ESCALATION_PROMPT,
    GENERAL_PROMPT,
    POLICY_REPLY_PROMPT,
    TRIAGE_PROMPT,
)
from jev import DEFAULT_JEV_MODEL, build_jev_triage_runnable

ORDER_VALUE_LIMIT = 5000  # policy Section 4: orders above INR 5,000 need a human

# =========================================================================== #
# 1. Core primitives: RunnableLambda and RunnablePassthrough.assign
# =========================================================================== #


def classify_ticket(ticket_text: str) -> str:
    """A tiny heuristic pre-filter (no LLM) - the classic RunnableLambda example."""
    text = ticket_text.lower()
    if any(w in text for w in ["worn", "used it", "already wore", "outside", "expired"]):
        return "likely_policy_violation"
    if any(w in text for w in ["broke", "defect", "peeling", "peeled", "torn", "damaged", "faulty"]):
        return "likely_genuine_defect"
    return "needs_llm_classification"


# A plain function becomes a Runnable: invoke / batch / stream / `|` all work.
quick_filter = RunnableLambda(classify_ticket)

_ORDER_ID_RE = re.compile(r"#?\b(HS\d+)\b", re.IGNORECASE)


def extract_order_id(source) -> Optional[str]:
    """Find an order id like #HS88213. Accepts a string or a dict with 'ticket_text'/'input'."""
    text = source if isinstance(source, str) else source.get("ticket_text") or source.get("input", "")
    m = _ORDER_ID_RE.search(text)
    return m.group(1).upper() if m else None


# .assign() keeps every key of the input dict and ADDS "order_id".
enrich_with_order_id = RunnablePassthrough.assign(order_id=RunnableLambda(extract_order_id))


# =========================================================================== #
# 2. The triage sub-chain (one LLM call, parsed defensively)
# =========================================================================== #

_CATEGORIES = ("genuine_defect", "policy_violation", "unclear", "other")
_PRIORITIES = {"high": "High", "medium": "Medium", "low": "Low"}


def parse_triage(raw: str) -> dict:
    """Turn the model's 'category,priority' text into a dict, tolerating messy output.

    LLMs add spaces, capitals, quotes or labels ("category: Genuine_Defect, priority: high").
    Anything we cannot understand becomes category='unclear', priority='Medium' - the safe
    default, because 'unclear' sends the case to a human.
    """
    text = str(raw or "").strip().lower().replace("-", "_")
    text = re.sub(r"(?<=[a-z])\s+(?=[a-z])", "_", text)  # "genuine defect" -> "genuine_defect"
    category = next((c for c in _CATEGORIES if c in text), "unclear")
    pm = re.search(r"(?<![a-z])(high|medium|low)(?![a-z])", text)
    priority = _PRIORITIES[pm.group(1)] if pm else "Medium"
    return {"category": category, "priority": priority}


def build_triage_subchain(llm):
    """One LLM call -> {'category': ..., 'priority': ...}."""
    return TRIAGE_PROMPT | llm | StrOutputParser() | RunnableLambda(parse_triage)


# =========================================================================== #
# 3. Deterministic signals (pure functions: no LLM, unit-testable)
# =========================================================================== #

_CHILD_WORDS = r"(child|children|kid|kids|baby|toddler|infant|son|daughter|mouth)"
_SAFETY_PATTERNS = [
    # (regex, reason).  Strong signals count on their own...
    (r"\b(chok\w*|swallow\w*|small parts?|injur\w+|hurt (him|her|them)|bleed\w*)",
     "possible child-safety concern (choking hazard / small parts / injury)"),
    (r"\bcut (him|her|them|my (son|daughter|child|baby))\b",
     "possible child-safety concern (injury)"),
    (r"\b(rash\w*|allerg\w+|itch\w*|skin reaction|hives|blister\w*|swollen)",
     "skin reaction or allergy complaint"),
]
# ...while "came loose / came off" only matters when a child is involved.
_WEAK_SAFETY = re.compile(r"\b(came loose|came off|fell off|come loose|coming loose)\b")
_CHILD = re.compile(r"\b" + _CHILD_WORDS + r"(?:'s)?\b")


def detect_safety_concern(text: str) -> dict:
    """Policy Section 4: child-safety and skin/allergy issues ALWAYS go to a human."""
    low = (text or "").lower()
    for pattern, reason in _SAFETY_PATTERNS:
        if re.search(pattern, low):
            return {"flag": True, "reason": reason}
    if _WEAK_SAFETY.search(low) and _CHILD.search(low):
        return {"flag": True, "reason": "possible child-safety concern (small part came loose)"}
    return {"flag": False, "reason": None}


_MONEY_RE = re.compile(r"(?:₹|(?<![a-z])rs\.?|(?<![a-z])inr)\s*(\d[\d,]*)", re.IGNORECASE)


def extract_order_value(text: str) -> Optional[int]:
    """Largest rupee amount mentioned (₹7,200 / Rs. 6500 / INR 5,500 / rs 12000), else None."""
    amounts = [int(m.group(1).replace(",", "").rstrip(",") or 0) for m in _MONEY_RE.finditer(text or "")]
    return max(amounts) if amounts else None


# =========================================================================== #
# 4. RunnableParallel: gather every signal at once
# =========================================================================== #


def build_signals(llm):
    """Run the LLM triage AND three deterministic checks side by side.

    RunnableParallel feeds the SAME input (the customer text) to every branch and
    LangChain runs the branches concurrently (threads for .invoke, asyncio for .ainvoke),
    so the keyword/regex checks cost nothing extra while the LLM call is in flight.
    Output: {"triage": {...}, "safety": {...}, "order_value": int|None, "order_id": str|None}
    """
    return RunnableParallel(
        triage=build_triage_subchain(llm),
        safety=RunnableLambda(detect_safety_concern),
        order_value=RunnableLambda(extract_order_value),
        order_id=RunnableLambda(extract_order_id),
    )


def build_enrichment_pipeline(llm):
    """The SEQUENTIAL way to gather the same signals: .assign().assign().assign()...

    Each .assign() keeps every existing key and adds one more, but the steps run one
    after another - so the LLM triage finishes before the cheap regex checks even start.
    Compare with build_signals above, where RunnableParallel runs them all at once.
    Output: {"input": ..., "triage": ..., "safety": ..., "order_value": ..., "order_id": ...}
    """
    text = lambda x: x["input"]  # noqa: E731 - each step reads the customer text from the dict
    return (
        RunnablePassthrough.assign(triage=text | build_triage_subchain(llm))
        .assign(safety=lambda x: detect_safety_concern(text(x)))
        .assign(order_value=lambda x: extract_order_value(text(x)))
        .assign(order_id=lambda x: extract_order_id(text(x)))
    )


# =========================================================================== #
# 5. decide_route: policy Section 4 as plain, testable code
# =========================================================================== #

ROUTES = ("human_review", "defect", "policy", "general")


def decide_route(x: dict) -> dict:
    """Read x['signals'] and add needs_human, human_reasons, route, triage_summary.

    The mandatory-human-review rules are enforced HERE, deterministically, so a model
    that is polite but wrong cannot auto-approve a child-safety or high-value case.
    """
    sig = x["signals"]
    triage = sig.get("triage") or {}
    category = triage.get("category", "unclear")
    priority = triage.get("priority", "Medium")
    safety = sig.get("safety") or {}
    value = sig.get("order_value")

    reasons = []
    if safety.get("flag"):
        reasons.append(safety.get("reason") or "safety concern")
        priority = "High"
    if value is not None and value > ORDER_VALUE_LIMIT:
        reasons.append(f"order value INR {value:,} is above INR {ORDER_VALUE_LIMIT:,}")
    if category == "unclear":
        reasons.append("facts are unclear, missing or conflicting (possible defect vs misuse)")

    needs_human = bool(reasons)
    if needs_human:
        route = "human_review"
    elif category == "genuine_defect":
        route = "defect"
    elif category == "policy_violation":
        route = "policy"
    else:
        route = "general"

    summary = f"category={category}; priority={priority}; needs_human={needs_human}"
    if reasons:
        summary += "; reasons=" + " | ".join(reasons)
    if sig.get("order_id"):
        summary += f"; order_id={sig['order_id']}"
    if value is not None:
        summary += f"; order_value=INR {value}"

    return {
        "chat_history": [],
        **x,  # keep input, chat_history, signals for the branch prompts
        "category": category,
        "priority": priority,
        "needs_human": needs_human,
        "human_reasons": reasons,
        "route": route,
        "triage_summary": summary,
    }


# =========================================================================== #
# 6. RunnableBranch + the full chain
# =========================================================================== #


def build_route_branch(llm):
    """Pick the reply prompt: (condition, runnable) pairs checked in order, then a default."""
    parser = StrOutputParser()
    return RunnableBranch(
        (lambda x: x["needs_human"], ESCALATION_PROMPT | llm | parser),
        (lambda x: x["category"] == "genuine_defect", DEFECT_PROMPT | llm | parser),
        (lambda x: x["category"] == "policy_violation", POLICY_REPLY_PROMPT | llm | parser),
        GENERAL_PROMPT | llm | parser,  # default
    )


def build_full_chain(llm):
    """The whole declarative pipeline (see the diagram at the top of this file)."""
    text_only = RunnableLambda(lambda x: x["input"])  # the parallel step needs just the text
    return (
        RunnablePassthrough.assign(signals=text_only | build_signals(llm))
        | RunnableLambda(decide_route)
        | build_route_branch(llm)
    )


# --- chat memory ---------------------------------- #
store: dict = {}  # session_id -> InMemoryChatMessageHistory


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def build_bot(llm):
    """The full chain + per-session memory. History is injected as 'chat_history'."""
    return RunnableWithMessageHistory(
        build_full_chain(llm),
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def explain_route(text: str, llm) -> dict:
    """Run only the signals + decide_route (no reply) - for the facilitator debug panel."""
    start = time.perf_counter()
    signals = build_signals(llm).invoke(text)
    decision = decide_route({"input": text, "signals": signals})
    return {
        "route": decision["route"],
        "needs_human": decision["needs_human"],
        "human_reasons": decision["human_reasons"],
        "category": decision["category"],
        "priority": decision["priority"],
        "triage_summary": decision["triage_summary"],
        "signals": signals,
        "latency_s": round(time.perf_counter() - start, 2),
    }


def build_greeting_chains(llm):
    """Demo: the same chain built with `|` and with explicit RunnableSequence."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You work for Hopscotch, a kids' fashion retailer. Reply in one short, warm sentence."),
        ("human", "{input}"),
    ])
    piped = prompt | llm | StrOutputParser()
    explicit = RunnableSequence(first=prompt, middle=[llm], last=StrOutputParser())
    return piped, explicit


# =========================================================================== #
# 6b. Jev-powered Decision Pipelines (System One)
# =========================================================================== #


def build_signals_jev(jev_model: str = DEFAULT_JEV_MODEL):
    """Signals pipeline using TypeSafe Jev for category, priority, and safety decisions."""
    jev_runnable = build_jev_triage_runnable(model=jev_model)

    def _combine_signals(inputs: dict) -> dict:
        jev_res = inputs["jev"]
        return {
            "triage": jev_res.get("triage", {}),
            "safety": jev_res.get("safety", {}),
            "order_value": inputs.get("order_value"),
            "order_id": inputs.get("order_id"),
            "jev_metadata": {
                "probabilities": jev_res.get("probabilities", {}),
                "latency_s": jev_res.get("latency_s", 0.0),
                "cost": jev_res.get("cost", 0.0),
                "tokens": jev_res.get("tokens", 0),
            },
        }

    return (
        RunnableParallel(
            jev=jev_runnable,
            order_value=RunnableLambda(extract_order_value),
            order_id=RunnableLambda(extract_order_id),
        )
        | RunnableLambda(_combine_signals)
    )


def build_full_chain_jev(llm, jev_model: str = DEFAULT_JEV_MODEL):
    """Full pipeline using Jev for decision nodes, LLM for downstream text generation only."""
    text_only = RunnableLambda(lambda x: x["input"] if isinstance(x, dict) else str(x))
    ensure_history = RunnableLambda(
        lambda x: {**x, "chat_history": x.get("chat_history", [])}
        if isinstance(x, dict)
        else {"input": str(x), "chat_history": []}
    )
    return (
        ensure_history
        | RunnablePassthrough.assign(signals=text_only | build_signals_jev(jev_model))
        | RunnableLambda(decide_route)
        | build_route_branch(llm)
    )


def explain_route_jev(text: str, jev_model: str = DEFAULT_JEV_MODEL) -> dict:
    """Run Jev signals + decide_route (no LLM reply generation) for benchmark comparison."""
    start = time.perf_counter()
    signals = build_signals_jev(jev_model).invoke(text)
    decision = decide_route({"input": text, "signals": signals})
    return {
        "route": decision["route"],
        "needs_human": decision["needs_human"],
        "human_reasons": decision["human_reasons"],
        "category": decision["category"],
        "priority": decision["priority"],
        "triage_summary": decision["triage_summary"],
        "signals": signals,
        "latency_s": round(time.perf_counter() - start, 3),
        "jev_metadata": signals.get("jev_metadata", {}),
    }


# =========================================================================== #
# 7. Sample tickets
# =========================================================================== #
SAMPLE_TICKETS = {
    "Defect: Dino-Print sneakers": (
        "Hi, I'm Ananya. I bought the Hopscotch Dino-Print Sneakers (Order #HS88213) for my son "
        "two weeks ago. The sole started peeling off after just three wears, even though he only "
        "wore them to school. This seems like a manufacturing issue. Can I get a replacement pair?"
    ),
    "Policy: worn jacket, wrong size": (
        "I bought a winter jacket last month, wore it a few times and it turns out it's the wrong "
        "size. Can I return it for a refund?"
    ),
    "Safety: button, toddler": "The button came off my toddler's shirt and he almost swallowed it.",
    "Value: INR 7,200 sneakers": "My order was ₹7,200 and the sneakers are peeling at the sole.",
    "Unclear: faded dress": "My dress colour faded, not sure if I washed it hot. Can I return it?",
    "General: Pune shipping": "Do you ship to Pune?",
}
SAMPLE_EXPECTED = dict(zip(SAMPLE_TICKETS, ["defect", "policy", "human_review", "human_review",
                                            "human_review", "general"]))
