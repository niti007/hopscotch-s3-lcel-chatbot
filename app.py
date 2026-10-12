"""Hopscotch Support (Session 3) - LCEL pipeline with RunnableParallel + RunnableBranch."""
import inspect
import os
import time
import uuid

import altair as alt
import pandas as pd
import streamlit as st

import chains
from chains import (
    SAMPLE_EXPECTED, SAMPLE_TICKETS, build_bot, build_enrichment_pipeline, build_full_chain,
    build_full_chain_jev, build_greeting_chains, build_route_branch, build_signals,
    build_signals_jev, decide_route, enrich_with_order_id, explain_route, explain_route_jev,
    quick_filter,
)
from jev import (
    DEFAULT_JEV_MODEL,
    calculate_jev_cost,
    calculate_llm_cost,
    estimate_tokens,
)
from llm import DEFAULT_MODEL, get_llm
from prompts import DEFECT_PROMPT, ESCALATION_PROMPT, GENERAL_PROMPT, POLICY_REPLY_PROMPT

st.set_page_config(page_title="Hopscotch Support (Session 3)", page_icon="🧵", layout="wide")

if not os.getenv("OPENROUTER_API_KEY"):
    st.error("OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key "
             "(or set it in the Streamlit Cloud app Secrets).")
    st.stop()

ROUTE_BADGES = {
    "human_review": "🚨 Human review",
    "defect": "🧵 Defect",
    "policy": "📋 Policy",
    "general": "💬 General",
}


@st.cache_resource
def get_store() -> dict:
    """Chat histories live here so they survive Streamlit reruns (same dict as chains.store)."""
    return chains.store


@st.cache_resource
def get_bot():
    """One bot for the whole app; get_session_history reads the shared store."""
    return build_bot(get_llm())


store = get_store()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("🧵 Hopscotch")
    st.caption(f"Model: {DEFAULT_MODEL}")
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:8]
    st.caption(f"Session id: `{st.session_state.session_id}`")
    if st.button("New session"):
        st.session_state.session_id = uuid.uuid4().hex[:8]
        st.rerun()
    facilitator = st.toggle("🛠️ Facilitator mode: show internal triage", value=False,
                            help="Adds one extra LLM call per turn to show the routing decision.")
    st.divider()
    st.header("⚡ Decision Engine")
    jev_model_id = st.text_input(
        "TypeSafe Jev Model",
        value=DEFAULT_JEV_MODEL,
        help="System 1 structured decision model via OpenRouter Decisions API",
    )
    st.caption("Jev Pricing: **$0.042 / 1M in**, **$0 / out**")

sid = st.session_state.session_id
try:
    llm = get_llm()
except Exception as e:
    st.error(f"Could not create the model client: {e}")
    st.stop()
st.session_state.setdefault("debug_log", {})  # (session, message index) -> explain_route output


def render_debug(info: dict):
    """Facilitator panel: route badge, reasons, the 4 parallel signals, latency."""
    with st.expander(f"🛠️ Internal triage - {ROUTE_BADGES[info['route']]}", expanded=False):
        st.markdown(f"**Route:** {ROUTE_BADGES[info['route']]}  |  **Latency (signals):** {info['latency_s']}s")
        if info["human_reasons"]:
            st.markdown("**Why a human:**")
            for r in info["human_reasons"]:
                st.markdown(f"- {r}")
        s = info["signals"]
        st.table(pd.DataFrame([
            {"signal": "triage (LLM)", "value": f"{s['triage']['category']} / {s['triage']['priority']}"},
            {"signal": "safety (keywords)", "value": str(s["safety"])},
            {"signal": "order_value (regex)", "value": str(s["order_value"])},
            {"signal": "order_id (regex)", "value": str(s["order_id"])},
        ]))


tab_chat, tab_pipe, tab_lab, tab_jev = st.tabs([
    "💬 Chat", "🔗 Pipeline", "🧪 Runnables lab", "⚡ Jev Decision Lab"
])

# ------------------------------------------------------------------ tab 1: chat
with tab_chat:
    st.subheader("Hopscotch Support")
    history = chains.get_session_history(sid).messages
    if not history:
        st.caption("Try: 'The sole of my son's sneakers is peeling after 3 wears' or "
                   "'The button came off my toddler's shirt'.")
    for i, m in enumerate(history):
        with st.chat_message("user" if m.type == "human" else "assistant"):
            st.markdown(m.content)
            if facilitator and (sid, i) in st.session_state.debug_log:
                render_debug(st.session_state.debug_log[(sid, i)])

    msg = st.chat_input("How can we help?")
    if msg:
        with st.chat_message("user"):
            st.markdown(msg)
        with st.chat_message("assistant"):
            try:
                t0 = time.perf_counter()
                bot = get_bot()
                # .stream() yields string chunks; RunnableWithMessageHistory saves the
                # complete reply to the store once the stream finishes.
                st.write_stream(bot.stream({"input": msg}, config={"configurable": {"session_id": sid}}))
                st.caption(f"⏱️ {time.perf_counter() - t0:.1f}s total")
                if facilitator:
                    info = explain_route(msg, llm)  # one extra LLM call, debug only
                    idx = len(chains.get_session_history(sid).messages) - 1
                    st.session_state.debug_log[(sid, idx)] = info
                    render_debug(info)
            except Exception as e:  # noqa: BLE001
                st.error(f"Something went wrong calling the model: {e}")

# ------------------------------------------------------------------ tab 2: pipeline
DOT = """
digraph G {
  rankdir=TB; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#7c6ae6", fillcolor="#efeafe", fontcolor="#222"];
  edge [color="#888"];
  input [label="input + chat_history", fillcolor="#e6f7f5", color="#1fa89a"];
  assign [label="RunnablePassthrough.assign(signals = ...)"];
  subgraph cluster_par { label="RunnableParallel (concurrent)"; fontname="Helvetica"; fontsize=10; color="#aaa"; style="dashed";
    triage [label="triage\\n(LLM)"]; safety [label="safety\\n(keywords)"]; value [label="order_value\\n(regex)"]; oid [label="order_id\\n(regex)"]; }
  decide [label="RunnableLambda(decide_route)\\nneeds_human, route, triage_summary"];
  subgraph cluster_br { label="RunnableBranch"; fontname="Helvetica"; fontsize=10; color="#aaa"; style="dashed";
    human [label="human_review\\nESCALATION_PROMPT | llm", fillcolor="#fde8e8", color="#d9534f"];
    defect [label="defect\\nDEFECT_PROMPT | llm"]; policy [label="policy\\nPOLICY_REPLY_PROMPT | llm"]; general [label="general (default)\\nGENERAL_PROMPT | llm"]; }
  parser [label="StrOutputParser"];
  reply [label="reply (streamed)", fillcolor="#e6f7f5", color="#1fa89a"];
  input -> assign; assign -> triage; assign -> safety; assign -> value; assign -> oid;
  triage -> decide; safety -> decide; value -> decide; oid -> decide;
  decide -> human; decide -> defect; decide -> policy; decide -> general;
  human -> parser; defect -> parser; policy -> parser; general -> parser; parser -> reply;
}
"""

with tab_pipe:
    st.subheader("The LCEL pipeline")
    st.graphviz_chart(DOT, width="stretch")
    st.markdown("**`build_full_chain` (from `chains.py`)**")
    st.code(inspect.getsource(build_full_chain), language="python")
    st.code(inspect.getsource(build_route_branch), language="python")
    with st.expander("LangChain's own graph: full_chain.get_graph().draw_ascii()"):
        try:
            st.code(build_full_chain(llm).get_graph().draw_ascii(), language="text")
        except Exception as e:  # noqa: BLE001
            st.info(f"Could not draw the graph: {e}")

    with st.expander("Branch prompts: the system prompt each RunnableBranch route uses"):
        for route, branch_prompt in [("human_review", ESCALATION_PROMPT), ("defect", DEFECT_PROMPT),
                                     ("policy", POLICY_REPLY_PROMPT), ("general (default)", GENERAL_PROMPT)]:
            st.markdown(f"**{route}**")
            # messages[0] is the system message; {policy} is filled in by .partial(), so show the rules only
            st.code(branch_prompt.messages[0].prompt.template.split("\n\n--- POLICY ---")[0], language="text")

    st.markdown("### Run step-by-step")
    key = st.selectbox("Sample ticket", list(SAMPLE_TICKETS), key="pipe_sample")
    text = st.text_area("Customer message", SAMPLE_TICKETS[key], key=f"pipe_text_{key}")
    if st.button("Run step-by-step"):
        try:
            payload = {"input": text, "chat_history": []}
            st.markdown("**Stage 0 - input**")
            st.json(payload)
            with st.spinner("Stage 1: RunnableParallel signals..."):
                signals = build_signals(llm).invoke(text)
            st.markdown("**Stage 1 - `RunnablePassthrough.assign(signals=...)`** (adds the `signals` key)")
            st.json(signals)
            routed = decide_route({**payload, "signals": signals})
            st.markdown("**Stage 2 - `decide_route`**")
            st.json({k: routed[k] for k in ("needs_human", "human_reasons", "category", "priority",
                                            "route", "triage_summary")})
            st.markdown(f"**Stage 3 - `RunnableBranch` chose:** {ROUTE_BADGES[routed['route']]}")
            with st.spinner("Stage 4: reply prompt | llm | parser..."):
                reply = build_route_branch(llm).invoke(routed)
            st.markdown("**Stage 4 - `StrOutputParser` output (the reply)**")
            st.info(reply)
        except Exception as e:  # noqa: BLE001
            st.error(f"Pipeline failed: {e}")

# ------------------------------------------------------------------ tab 3: lab
with tab_lab:
    st.subheader("Runnables lab")

    st.markdown("#### (a) `|` pipe vs explicit `RunnableSequence`")
    c1, c2 = st.columns(2)
    c1.code("chain = prompt | llm | StrOutputParser()", language="python")
    c2.code("chain = RunnableSequence(\n    first=prompt,\n    middle=[llm],\n    last=StrOutputParser(),\n)",
            language="python")
    q = st.text_input("Question", "Do you have raincoats for toddlers?", key="lab_a")
    if st.button("Run both", key="btn_a"):
        try:
            piped, explicit = build_greeting_chains(llm)
            c1.write(piped.invoke({"input": q}))
            c2.write(explicit.invoke({"input": q}))
        except Exception as e:  # noqa: BLE001
            st.error(str(e))

    st.markdown("#### (b) `quick_filter.invoke(...)` - a RunnableLambda, no LLM")
    t_b = st.text_input("Ticket text", "The sole peeled off after one wear.", key="lab_b")
    st.code(f"quick_filter.invoke({t_b!r})  ->  {quick_filter.invoke(t_b)!r}", language="python")

    st.markdown("#### (c) `RunnablePassthrough.assign` keeps the original keys")
    t_c = st.text_input("Ticket text", "My order #HS88213 arrived with a broken zipper.", key="lab_c")
    st.json(enrich_with_order_id.invoke({"ticket_text": t_c, "customer_name": "Ananya"}))

    st.markdown("#### (d) `.batch()` over all sample tickets")
    if st.button("Triage all samples (batch)", key="btn_d"):
        try:
            names, texts = list(SAMPLE_TICKETS), list(SAMPLE_TICKETS.values())
            t0 = time.perf_counter()
            all_signals = build_signals(llm).batch(texts, config={"max_concurrency": 3})
            took = time.perf_counter() - t0
            rows = []
            for n, tx, s in zip(names, texts, all_signals):
                d = decide_route({"input": tx, "signals": s})
                rows.append({"ticket": n, "category": d["category"], "priority": d["priority"],
                             "safety": s["safety"]["flag"], "order_value": s["order_value"],
                             "route": ROUTE_BADGES[d["route"]],
                             "expected": ROUTE_BADGES[SAMPLE_EXPECTED[n]]})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption(f"{len(texts)} tickets in {took:.1f}s with max_concurrency=3")
        except Exception as e:  # noqa: BLE001
            st.error(str(e))

    st.markdown("#### (e) `.stream()` token by token")
    q_e = st.text_input("Question", "Tell me about your return policy.", key="lab_e")
    if st.button("Stream", key="btn_e"):
        try:
            piped, _ = build_greeting_chains(llm)
            st.write_stream(piped.stream({"input": q_e}))
        except Exception as e:  # noqa: BLE001
            st.error(str(e))

    st.markdown("#### (f) `.assign().assign()` vs `RunnableParallel`")
    st.code("""# sequential: each .assign() waits for the previous one
build_enrichment_pipeline = (
    RunnablePassthrough.assign(triage=text | triage_subchain)   # LLM call first...
    .assign(safety=...).assign(order_value=...).assign(order_id=...)  # ...then the cheap checks
)
# concurrent: all four start together
build_signals = RunnableParallel(triage=..., safety=..., order_value=..., order_id=...)""", language="python")
    t_f = st.text_area("Ticket", SAMPLE_TICKETS["Defect: Dino-Print sneakers"], key="lab_f")
    if st.button("Run both and compare", key="btn_f"):
        try:
            t0 = time.perf_counter()
            seq_out = build_enrichment_pipeline(llm).invoke({"input": t_f})
            seq_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            par_out = build_signals(llm).invoke(t_f)
            par_s = time.perf_counter() - t0
            f1, f2 = st.columns(2)
            f1.markdown(f"**`.assign()` sequence** - {seq_s:.2f}s")
            f1.json(seq_out)
            f2.markdown(f"**`RunnableParallel`** - {par_s:.2f}s")
            f2.json(par_out)
            st.caption("Same signals either way. The difference: the assign steps run one at a time, "
                       "RunnableParallel runs its steps concurrently. Only one step here is a slow LLM "
                       "call, so the gap is small - with several LLM steps it would be large.")
        except Exception as e:  # noqa: BLE001
            st.error(str(e))

# ------------------------------------------------------------------ tab 4: jev decision lab
with tab_jev:
    st.subheader("⚡ Jev Decision Lab - LLM vs Decision Model Comparison")
    st.markdown(
        "Decision models (**System 1**) evaluate typed questions over application state without "
        "generating free-form text tokens. This eliminates output token latency and costs while providing "
        "calibrated probability distributions. Downstream customer replies are still handled by the LLM (**System 2**)."
    )

    info_c1, info_c2 = st.columns(2)
    with info_c1:
        st.info(
            f"**LLM Triage (`{DEFAULT_MODEL}`)**\n\n"
            "- Generative completion parsed defensively\n"
            "- Pricing: $0.15 / 1M input | $0.60 / 1M output\n"
            "- Deterministic keyword regex for safety checks"
        )
    with info_c2:
        st.success(
            f"**TypeSafe Jev (`{jev_model_id}`)**\n\n"
            "- Structured decisions: `choice` (category), `score` (priority), `noul` (safety)\n"
            "- Pricing: **$0.042 / 1M input | $0.00 output** (Zero generation cost)\n"
            "- Returns calibrated probability distributions"
        )

    st.markdown("### 1. Interactive Single-Ticket Benchmark")
    sample_key = st.selectbox("Choose sample ticket or type below:", list(SAMPLE_TICKETS), key="jev_sample_sel")
    custom_ticket = st.text_area("Ticket Text", SAMPLE_TICKETS[sample_key], height=110, key="jev_ticket_text")

    if st.button("Compare Decision Nodes (LLM vs Jev)", key="btn_compare_single"):
        with st.spinner("Running LLM triage and TypeSafe Jev decisions..."):
            try:
                # 1. LLM run
                t0_llm = time.perf_counter()
                llm_info = explain_route(custom_ticket, llm)
                llm_time = time.perf_counter() - t0_llm
                llm_in_tok = estimate_tokens(custom_ticket) + 120  # prompt overhead ~120 tokens
                llm_out_tok = 15  # "category,priority" ~15 tokens
                llm_cost = calculate_llm_cost(llm_in_tok, llm_out_tok)

                # 2. Jev run
                t0_jev = time.perf_counter()
                jev_info = explain_route_jev(custom_ticket, jev_model=jev_model_id)
                jev_time = time.perf_counter() - t0_jev
                jev_in_tok = estimate_tokens(custom_ticket)
                jev_cost = calculate_jev_cost(jev_in_tok)

                # Summary metrics
                m1, m2, m3, m4 = st.columns(4)
                speedup = f"{llm_time / max(jev_time, 0.001):.1f}x" if jev_time > 0 else "N/A"
                cost_saving = f"{(1 - (jev_cost / max(llm_cost, 0.0000001))) * 100:.1f}%"
                match_str = "✅ Match" if llm_info["route"] == jev_info["route"] else "⚠️ Differ"

                m1.metric("LLM Latency", f"{llm_time:.2f}s")
                m2.metric(
                    "Jev Latency",
                    f"{jev_time:.2f}s",
                    delta=f"{speedup} faster" if jev_time < llm_time else "slower",
                )
                m3.metric("LLM Cost", f"${llm_cost:.6f}")
                m4.metric("Jev Cost", f"${jev_cost:.6f}", delta=f"-{cost_saving}", delta_color="inverse")

                col_llm, col_jev = st.columns(2)
                with col_llm:
                    st.markdown(f"#### LLM Route: {ROUTE_BADGES.get(llm_info['route'], llm_info['route'])}")
                    st.write(f"- **Category:** `{llm_info['category']}`")
                    st.write(f"- **Priority:** `{llm_info['priority']}`")
                    st.write(f"- **Safety Flag (Regex):** `{llm_info['signals']['safety']['flag']}`")
                    st.write(f"- **Estimated Cost:** `${llm_cost:.6f}` ({llm_in_tok} in / {llm_out_tok} out tokens)")
                    if llm_info["human_reasons"]:
                        st.caption(f"Reasons: {', '.join(llm_info['human_reasons'])}")

                with col_jev:
                    st.markdown(f"#### Jev Route: {ROUTE_BADGES.get(jev_info['route'], jev_info['route'])}")
                    st.write(f"- **Category (Choice):** `{jev_info['category']}`")
                    st.write(f"- **Priority (Score):** `{jev_info['priority']}`")
                    st.write(f"- **Safety (Noul):** `{jev_info['signals']['safety']['flag']}`")
                    st.write(f"- **Estimated Cost:** `${jev_cost:.6f}` ({jev_in_tok} in / 0 out tokens)")

                    probs = jev_info["signals"].get("jev_metadata", {}).get("probabilities", {})
                    if probs.get("category"):
                        st.caption("Category Probabilities:")
                        st.json(probs["category"])
                    if probs.get("priority"):
                        st.caption("Priority Probabilities:")
                        st.json(probs["priority"])

                with st.expander("Downstream reply generation test (LLM generates reply based on Jev route)"):
                    st.caption("Notice: Jev strictly decided the route; the actual reply text is generated by the LLM:")
                    branch_llm = build_route_branch(llm)
                    routed_jev = decide_route({"input": custom_ticket, "chat_history": [], "signals": jev_info["signals"]})
                    reply_text = branch_llm.invoke(routed_jev)
                    st.info(reply_text)

            except Exception as err:
                st.error(f"Comparison failed: {err}")

    st.divider()
    st.markdown("### 2. Batch Benchmark Across All Sample Tickets")
    st.caption("Compare speed, cost, and routing consistency across all predefined test cases.")

    if st.button("Run Batch Benchmark (All 6 Samples)", key="btn_batch_jev"):
        names, texts = list(SAMPLE_TICKETS), list(SAMPLE_TICKETS.values())
        progress = st.progress(0)
        rows = []
        total_llm_time = 0.0
        total_jev_time = 0.0
        total_llm_cost = 0.0
        total_jev_cost = 0.0

        for idx, (name, text) in enumerate(zip(names, texts)):
            progress.progress((idx + 1) / len(names))
            try:
                # LLM
                t0 = time.perf_counter()
                l_info = explain_route(text, llm)
                l_time = time.perf_counter() - t0
                l_in = estimate_tokens(text) + 120
                l_cost = calculate_llm_cost(l_in, 15)

                # Jev
                t0 = time.perf_counter()
                j_info = explain_route_jev(text, jev_model=jev_model_id)
                j_time = time.perf_counter() - t0
                j_in = estimate_tokens(text)
                j_cost = calculate_jev_cost(j_in)

                total_llm_time += l_time
                total_jev_time += j_time
                total_llm_cost += l_cost
                total_jev_cost += j_cost

                exp_route = SAMPLE_EXPECTED.get(name, "unknown")
                rows.append({
                    "Ticket": name,
                    "Expected": ROUTE_BADGES.get(exp_route, exp_route),
                    "LLM Route": ROUTE_BADGES.get(l_info["route"], l_info["route"]),
                    "Jev Route": ROUTE_BADGES.get(j_info["route"], j_info["route"]),
                    "Agree?": "✅ Yes" if l_info["route"] == j_info["route"] else "⚠️ No",
                    "LLM Time (s)": round(l_time, 2),
                    "Jev Time (s)": round(j_time, 2),
                    "LLM Cost ($)": f"${l_cost:.6f}",
                    "Jev Cost ($)": f"${j_cost:.6f}",
                })
            except Exception as e:
                rows.append({
                    "Ticket": name,
                    "Expected": ROUTE_BADGES.get(SAMPLE_EXPECTED.get(name, ""), ""),
                    "LLM Route": "Error",
                    "Jev Route": f"Error: {e}",
                    "Agree?": "❌",
                    "LLM Time (s)": 0,
                    "Jev Time (s)": 0,
                    "LLM Cost ($)": "$0",
                    "Jev Cost ($)": "$0",
                })

        st.session_state["batch_results"] = {
            "rows": rows,
            "total_llm_time": total_llm_time,
            "total_jev_time": total_jev_time,
            "total_llm_cost": total_llm_cost,
            "total_jev_cost": total_jev_cost,
            "avg_llm_cost": total_llm_cost / max(len(names), 1),
            "avg_jev_cost": total_jev_cost / max(len(names), 1),
            "avg_llm_time": total_llm_time / max(len(names), 1),
            "avg_jev_time": total_jev_time / max(len(names), 1),
        }

    if "batch_results" in st.session_state:
        bdata = st.session_state["batch_results"]
        rows = bdata["rows"]
        total_llm_time = bdata["total_llm_time"]
        total_jev_time = bdata["total_jev_time"]
        total_llm_cost = bdata["total_llm_cost"]
        total_jev_cost = bdata["total_jev_cost"]
        avg_llm_cost = bdata["avg_llm_cost"]
        avg_jev_cost = bdata["avg_jev_cost"]
        avg_llm_time = bdata["avg_llm_time"]
        avg_jev_time = bdata["avg_jev_time"]

        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        b1, b2, b3 = st.columns(3)
        latency_diff = total_jev_time - total_llm_time
        speedup_factor = (
            f"{total_llm_time / max(total_jev_time, 0.001):.1f}x faster"
            if total_jev_time <= total_llm_time
            else f"{total_jev_time / max(total_llm_time, 0.001):.1f}x slower"
        )
        b1.metric(
            "Total Latency",
            f"Jev: {total_jev_time:.2f}s",
            delta=f"{latency_diff:+.2f}s ({speedup_factor}) vs LLM ({total_llm_time:.2f}s)",
            delta_color="inverse",
        )
        b2.metric(
            "Total Decision Cost",
            f"Jev: ${total_jev_cost:.6f}",
            delta=f"-{(1 - total_jev_cost / max(total_llm_cost, 1e-9)) * 100:.1f}% vs LLM (${total_llm_cost:.6f})",
            delta_color="inverse",
        )
        agreed_count = sum(1 for r in rows if "✅" in r.get("Agree?", ""))
        b3.metric("Decision Agreement", f"{agreed_count}/{len(rows)} ({agreed_count / len(rows) * 100:.0f}%)")

        st.divider()
        st.markdown("### 3. Projected Scale & Growing Comparison")
        st.caption("Extrapolate growing costs and processing time from 1,000 to 1,000,000 production tickets.")

        if st.button("📈 Project Cost & Time at Scale", key="btn_project_scale"):
            vol_steps = [int(10_000 + (1_000_000 - 10_000) * i / 34) for i in range(35)]
            max_cost_llm = max(1_000_000 * avg_llm_cost * 1.05, 1.0)
            max_time_llm = max(((1_000_000 * avg_llm_time) / 3600) * 1.05, 0.1)

            chart_c1, chart_c2 = st.columns(2)
            with chart_c1:
                st.markdown("#### Cumulative Cost Projection ($)")
                cost_ph = st.empty()
            with chart_c2:
                st.markdown("#### Cumulative Time Projection (Hours)")
                time_ph = st.empty()

            cost_data = []
            time_data = []

            for v in vol_steps:
                cost_data.extend([
                    {"Tickets": v, "Cost": round(v * avg_llm_cost, 2), "Model": f"LLM ({DEFAULT_MODEL})"},
                    {"Tickets": v, "Cost": round(v * avg_jev_cost, 2), "Model": f"Jev ({jev_model_id})"},
                ])
                time_data.extend([
                    {"Tickets": v, "Hours": round((v * avg_llm_time) / 3600, 2), "Model": f"LLM ({DEFAULT_MODEL})"},
                    {"Tickets": v, "Hours": round((v * avg_jev_time) / 3600, 2), "Model": f"Jev ({jev_model_id})"},
                ])

                chart_cost = (
                    alt.Chart(pd.DataFrame(cost_data))
                    .mark_line(interpolate="monotone", strokeWidth=3.5)
                    .encode(
                        x=alt.X("Tickets:Q", scale=alt.Scale(domain=[0, 1_000_000]), title="Ticket Volume", axis=alt.Axis(format="~s")),
                        y=alt.Y("Cost:Q", scale=alt.Scale(domain=[0, max_cost_llm]), title="Total Cost ($)", axis=alt.Axis(format="$,.0f")),
                        color=alt.Color(
                            "Model:N",
                            scale=alt.Scale(
                                domain=[f"LLM ({DEFAULT_MODEL})", f"Jev ({jev_model_id})"],
                                range=["#ff4b4b", "#00c08b"],
                            ),
                            legend=alt.Legend(orient="bottom", title=None),
                        ),
                    )
                    .properties(height=320)
                )

                chart_time = (
                    alt.Chart(pd.DataFrame(time_data))
                    .mark_line(interpolate="monotone", strokeWidth=3.5)
                    .encode(
                        x=alt.X("Tickets:Q", scale=alt.Scale(domain=[0, 1_000_000]), title="Ticket Volume", axis=alt.Axis(format="~s")),
                        y=alt.Y("Hours:Q", scale=alt.Scale(domain=[0, max_time_llm]), title="Compute Time (Hours)", axis=alt.Axis(format=",.0f")),
                        color=alt.Color(
                            "Model:N",
                            scale=alt.Scale(
                                domain=[f"LLM ({DEFAULT_MODEL})", f"Jev ({jev_model_id})"],
                                range=["#ff4b4b", "#00c08b"],
                            ),
                            legend=alt.Legend(orient="bottom", title=None),
                        ),
                    )
                    .properties(height=320)
                )

                cost_ph.altair_chart(chart_cost, use_container_width=True)
                time_ph.altair_chart(chart_time, use_container_width=True)
                time.sleep(0.02)

            p100_llm_cost = 100_000 * avg_llm_cost
            p100_jev_cost = 100_000 * avg_jev_cost
            p100_saved = p100_llm_cost - p100_jev_cost
            p100_llm_hrs = (100_000 * avg_llm_time) / 3600
            p100_jev_hrs = (100_000 * avg_jev_time) / 3600
            p100_hrs_saved = p100_llm_hrs - p100_jev_hrs

            st.success(
                f"**At 100,000 Tickets Scale:**\n\n"
                f"- **Cost:** LLM ${p100_llm_cost:,.2f} ➔ Jev **${p100_jev_cost:,.2f}** "
                f"(Saves **${p100_saved:,.2f}**, {((1 - p100_jev_cost / max(p100_llm_cost, 1e-9)) * 100):.1f}% reduction)\n"
                f"- **Compute Time:** LLM {p100_llm_hrs:.1f} hrs ➔ Jev **{p100_jev_hrs:.1f} hrs** "
                f"(Saves **{p100_hrs_saved:.1f} hours** of compute)"
            )

