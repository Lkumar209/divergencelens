"""Streamlit dashboard: inspect runs, view provenance graph, visualize divergences."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

st.set_page_config(
    page_title="DivergenceLens",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🔍 DivergenceLens")
st.sidebar.caption("Silent divergence auditing for Deep Agents")

mode = st.sidebar.radio("Mode", ["Audit a run", "Browse benchmark results", "About"])

# ---------------------------------------------------------------------------
# Helper: load audit result
# ---------------------------------------------------------------------------


def _load_result(source: str | dict) -> dict[str, Any] | None:
    try:
        if isinstance(source, dict):
            return source
        p = Path(source)
        if p.exists():
            return json.loads(p.read_text())
        return None
    except Exception as exc:
        st.error(f"Failed to load: {exc}")
        return None


def _severity_color(sev: str) -> str:
    return {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(sev, "⚪")


def _category_emoji(cat: str) -> str:
    return {
        "phantom_completion": "👻",
        "silent_failure_masking": "🔇",
        "claim_write_mismatch": "✍️",
        "summary_inflation": "🎈",
        "plan_drift": "🧭",
        "orphaned_evidence": "🗂️",
    }.get(cat, "❓")


# ---------------------------------------------------------------------------
# Mode 1: Audit a run
# ---------------------------------------------------------------------------

if mode == "Audit a run":
    st.title("Audit a Run")

    col1, col2 = st.columns([2, 1])
    with col1:
        run_id = st.text_input("LangSmith Run ID or path to trace JSON")
    with col2:
        enable_judge = st.checkbox("Enable LLM judge", value=False)
        judge_model = st.selectbox("Judge model", ["gpt-4o-mini", "gpt-4o", "claude-haiku-4-5-20251001"])

    uploaded = st.file_uploader("Or upload a trace JSON", type=["json"])

    if st.button("Audit", type="primary") and (run_id or uploaded):
        with st.spinner("Running DivergenceLens audit..."):
            try:
                from divergencelens.core.config import DivergenceLensConfig, DetectionConfig, JudgeConfig
                from divergencelens.sdk.client import DivergenceLens

                config = DivergenceLensConfig(
                    detection=DetectionConfig(
                        enable_judge=enable_judge,
                        judge=JudgeConfig(model=judge_model),
                    )
                )
                lens = DivergenceLens(config)

                if uploaded:
                    trace_data = json.loads(uploaded.read())
                    from divergencelens.ingest.trace_normalizer import TraceNormalizer
                    run = TraceNormalizer().normalize_from_langsmith(trace_data)
                    result = lens.audit_run(run)
                elif run_id and Path(run_id).exists():
                    result = lens.audit_json(run_id)
                else:
                    result = lens.audit_langsmith_run(run_id)

                st.session_state["last_result"] = result.model_dump()
                st.session_state["last_run"] = None
            except Exception as exc:
                st.error(f"Audit failed: {exc}")
                st.stop()

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        divs = result.get("divergences", [])
        cells = result.get("cells", [])
        summary = result.get("summary", {})

        st.markdown("---")

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Run ID", result["run_id"][:12] + "…")
        c2.metric("Divergences", len(divs))
        c3.metric("Cells scored", len(cells))
        c4.metric("Audit time", f"{result.get('duration_ms', 0):.0f} ms")

        status = "✅ Clean" if len(divs) == 0 else f"⚠️ {len(divs)} divergence(s)"
        st.markdown(f"### Status: {status}")

        if divs:
            st.markdown("### Divergence Timeline")
            sorted_divs = sorted(divs, key=lambda d: d.get("step_index") or 0)
            for i, div in enumerate(sorted_divs, 1):
                sev = div.get("severity", "low")
                cat = div.get("category", "unknown")
                conf = div.get("confidence", 0.0)
                with st.expander(
                    f"{_severity_color(sev)} {i}. `{cat}` — step {div.get('step_index', '?')} (confidence {conf:.2f})"
                ):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Stated:**")
                        st.info(div.get("stated_excerpt", "—"))
                    with col_b:
                        st.markdown("**Enacted:**")
                        st.warning(div.get("enacted_excerpt", "—"))
                    st.markdown(f"**Rationale:** {div.get('rationale', '')}")
                    if div.get("evidence_path"):
                        st.markdown(f"**Evidence path:** `{' → '.join(div['evidence_path'][:5])}`")

            # Provenance graph (text-based since we can't render networkx in Streamlit easily)
            st.markdown("### Category Breakdown")
            by_cat = summary.get("by_category", {})
            if by_cat:
                import pandas as pd
                df = pd.DataFrame(
                    [{"Category": f"{_category_emoji(k)} {k}", "Count": v} for k, v in by_cat.items()]
                )
                st.bar_chart(df.set_index("Category"))

        # Raw JSON
        with st.expander("Raw audit result JSON"):
            st.json(result)

# ---------------------------------------------------------------------------
# Mode 2: Browse benchmark results
# ---------------------------------------------------------------------------

elif mode == "Browse benchmark results":
    st.title("DivergenceBench Results")

    results_path = Path("results/results.json")
    if not results_path.exists():
        st.warning("No benchmark results found. Run `make bench` first.")
    else:
        data = json.loads(results_path.read_text())
        import pandas as pd

        st.markdown(f"**Split:** {data['split']} | **Seeds:** {data['n_seeds']}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean F1", f"{data['mean_f1']:.4f}")
        c2.metric("Std", f"{data['std_f1']:.4f}")
        c3.metric("95% CI", f"({data['ci_95'][0]:.4f}, {data['ci_95'][1]:.4f})")

        st.markdown("### Per-Seed Results")
        rows = []
        for r in data["per_seed"]:
            rows.append({
                "Seed": r["seed"],
                "F1": f"{r['f1']:.4f}",
                "Precision": f"{r['precision']:.4f}",
                "Recall": f"{r['recall']:.4f}",
                "FP Rate": f"{r['fp_rate']:.4f}",
                "Localization": f"{r['localization_acc']:.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("### Per-Category F1 (Seed 0)")
        if data["per_seed"]:
            cat_f1 = data["per_seed"][0].get("per_category_f1", {})
            if cat_f1:
                df_cat = pd.DataFrame([
                    {"Category": f"{_category_emoji(k)} {k}", "F1": v}
                    for k, v in cat_f1.items()
                ])
                st.bar_chart(df_cat.set_index("Category"))

        with st.expander("Full results.json"):
            st.json(data)

# ---------------------------------------------------------------------------
# Mode 3: About
# ---------------------------------------------------------------------------

elif mode == "About":
    st.title("About DivergenceLens")
    st.markdown("""
**DivergenceLens** audits LangChain Deep Agents for *silent divergence* — gaps between
what an agent **states** (plan, claims, subagent summaries) and what it **actually enacts**
(tool calls, file mutations, real subagent trajectories).

### Divergence taxonomy

| Category | Description |
|---|---|
| 👻 Phantom completion | Todo marked done with no supporting action |
| 🔇 Silent failure masking | Tool errored; agent claimed success |
| ✍️ Claim–write mismatch | Agent asserts it wrote a file; no mutation exists |
| 🎈 Summary inflation | Async subagent summary overstates vs. real trajectory |
| 🧭 Plan drift | Consequential actions with no corresponding todo |
| 🗂️ Orphaned evidence | Retrieved content never used or contradicted |

### Architecture

```
serve / sdk / cli
    reporting · dashboard
    runtime: middleware · monitor · policy · interrupt · rollback
    detection: consistency matrix · taxonomy · severity
    alignment: deterministic rules · judge · calibration
    provenance: causal / data-flow graph
    ingest: LangSmith · LangGraph · OTEL · stream
    core: event schema · types · config · registries
```

**GitHub:** https://github.com/Lkumar209/divergencelens
    """)
