"""
MACI Sentinel — Production AI Governance Dashboard
Maqasid AI | Beenish Fatima | maqasidai.org | Superior University Lahore

Backend API: SyedaScientist72/macimlops  →  Space: SyedaScientist72/MLOps-MACI
Run locally:  streamlit run app.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MACI Sentinel — Production AI Governance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = "https://syedascientist72-mlops-maci.hf.space"

CUSTOM_CSS = """
<style>
.hero {
    background: linear-gradient(135deg, #0B1F3A 0%, #103A5E 50%, #0B1F3A 100%);
    padding: 2.2rem 2.5rem;
    border-radius: 14px;
    margin-bottom: 1.4rem;
    border: 1px solid #1d4a73;
}
.hero h1 { color: #ffffff; font-size: 2.0rem; margin-bottom: 0.2rem; }
.hero p  { color: #9fc4e8; font-size: 1.0rem; margin: 0; }
.diff-badge {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    color: #cfe6ff;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    margin: 4px 6px 0 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR — connection
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ MACI Sentinel")
    st.caption("Production AI Governance · Maqasid AI")
    st.markdown("---")
    st.markdown("### 🔌 Connection")

    api_url = st.text_input(
        "API URL",
        value=st.session_state.get("api_url", DEFAULT_API_URL),
        help="Your HuggingFace Space URL, no trailing slash",
    )
    api_key = st.text_input(
        "API Key", type="password",
        value=st.session_state.get("api_key", ""),
        help="Bearer token — request access at maqasidai.org",
    )
    st.session_state["api_url"] = api_url.rstrip("/")
    st.session_state["api_key"] = api_key
    api_url = st.session_state["api_url"]
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    if st.button("🔄 Check Connection", use_container_width=True):
        try:
            r = requests.get(f"{api_url}/health", timeout=10)
            if r.status_code == 200:
                st.success("✅ Connected")
                st.session_state["health"] = r.json()
            else:
                st.error(f"❌ HTTP {r.status_code}")
        except Exception as e:
            st.error(f"❌ {e}")

    st.markdown("---")
    st.markdown(
        "**Beenish Fatima**  \n"
        "[maqasidai.org](https://maqasidai.org)  \n"
        "Superior University Lahore"
    )
    st.caption("© 2026 Maqasid AI")


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def api_get(path: str, params=None):
    try:
        r = requests.get(f"{api_url}{path}", headers=headers, params=params, timeout=15)
        if r.status_code == 200:
            return r.json(), None
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, str(e)


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{api_url}{path}", headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            return r.json(), None
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, str(e)


def status_colour(level: str) -> str:
    return {"HEALTHY": "🟢", "OK": "🟢", "INFO": "🔵",
            "WARNING": "🟡", "CRITICAL": "🔴"}.get(level, "⚪")


# ─────────────────────────────────────────────────────────────
# HERO — billboard section, always visible at top
# ─────────────────────────────────────────────────────────────

health_data, _ = api_get("/health")
dash_data, _   = api_get("/dashboard/summary")

st.markdown(f"""
<div class="hero">
  <h1>🛡️ MACI Sentinel</h1>
  <p>Know your fraud model still works — before your business finds out the hard way.</p>
  <div style="margin-top:14px;">
    <span class="diff-badge">⚡ Cost-aware thresholding, not just accuracy</span>
    <span class="diff-badge">📊 Real feature-level drift (PSI + KS), not prediction-only proxies</span>
    <span class="diff-badge">🌍 Built for emerging-market fintech — lightweight, no Kubernetes</span>
    <span class="diff-badge">🕌 Shariah-aware governance layer via MaqasidAI / MACI</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Live status strip — this is the "billboard" proof-of-life
status_cols = st.columns(5)
if health_data:
    model_ready = health_data.get("model_ready", False)
    status_cols[0].metric("System", "🟢 LIVE" if model_ready else "🔴 DOWN")
    status_cols[1].metric("Model Source", health_data.get("model_source", "—"))
    roc = health_data.get("roc_auc")
    status_cols[2].metric("ROC-AUC", f"{roc:.4f}" if roc else "—")
    pr = health_data.get("pr_auc")
    status_cols[3].metric("PR-AUC", f"{pr:.4f}" if pr else "—")
    status_cols[4].metric("Threshold", health_data.get("current_threshold", "—"))
else:
    status_cols[0].metric("System", "⚪ Connecting…")
    for c in status_cols[1:]:
        c.metric("—", "—")
    st.info("Enter your API key in the sidebar, or click **Check Connection**, to activate live monitoring.")


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", "🔍 Score Transaction",
    "📁 Batch Scoring", "📈 Drift Monitor", "⚙️ Settings",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Production Snapshot")

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            data, err = api_get("/dashboard/summary")
            if data:
                st.session_state["dashboard"] = data
            elif err:
                st.warning(f"Could not load dashboard: {err}")

    if "dashboard" not in st.session_state and dash_data:
        st.session_state["dashboard"] = dash_data

    d = st.session_state.get("dashboard", {})
    if d:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("System Status",
                  status_colour(d.get("system_status", "—")) + " " + d.get("system_status", "—"))
        c2.metric("Active Threshold", d.get("current_threshold", "—"))
        c3.metric("Total Alerts", d.get("total_alerts", 0))
        c4.metric("Reference Rows", f"{d.get('reference_rows', 0):,}")

        m = d.get("model_metrics", {})
        if any(v for v in m.values() if v is not None):
            st.markdown("#### Model Performance")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("ROC-AUC", m.get("roc_auc", "—"))
            mc2.metric("PR-AUC", m.get("pr_auc", "—"))
            mc3.metric("Trained Threshold", m.get("optimal_threshold", "—"))
            mc4.metric("Avg Production Recall", m.get("avg_production_recall", "—"))

        crit = d.get("recent_critical", 0)
        warn = d.get("recent_warnings", 0)
        if crit:
            st.error(f"🔴 {crit} critical alert(s) in recent activity")
        elif warn:
            st.warning(f"🟡 {warn} warning(s) in recent activity")
        else:
            st.success("🟢 All systems nominal")
    else:
        st.info("No dashboard data yet — check your connection in the sidebar.")

    st.markdown("#### Recent Alerts")
    alerts_data, alerts_err = api_get("/alerts", params={"unresolved_only": False})
    if alerts_data and alerts_data.get("alerts"):
        df_alerts = pd.DataFrame(alerts_data["alerts"])
        cols = [c for c in ["id", "timestamp", "alert_level", "event",
                            "mean_psi", "recall", "resolved"]
                if c in df_alerts.columns]
        st.dataframe(df_alerts[cols].head(20), use_container_width=True)
    elif alerts_err:
        st.warning(f"Could not load alerts: {alerts_err}")
    else:
        st.info("No alerts yet. Run a drift report or batch score to generate alerts.")


# ══════════════════════════════════════════════════════════════
# TAB 2 — SCORE A SINGLE TRANSACTION
# ══════════════════════════════════════════════════════════════

with tab2:
    st.header("🔍 Score a Single Transaction")
    st.markdown(
        "Enter V1–V28 + Amount. Engineered features (interaction terms, "
        "log-amount, summary stats) are derived automatically server-side."
    )

    input_method = st.radio("Input method", ["JSON", "Sliders"], horizontal=True)

    if input_method == "JSON":
        example = {f"V{i}": round(float(np.random.normal(0, 1)), 4) for i in range(1, 29)}
        example["Amount"] = 149.62
        default_json = json.dumps({"features": example}, indent=2)
        raw = st.text_area("Transaction JSON", value=default_json, height=250)

        if st.button("🚀 Score", use_container_width=True):
            try:
                payload = json.loads(raw)
                result, err = api_post("/predict", payload)
                if result:
                    prob = result["fraud_probability"]
                    col1, col2 = st.columns(2)
                    col1.metric("Fraud Probability", f"{prob:.4f}")
                    if result["decision"] == "FRAUD":
                        col2.error(f"🔴 {result['decision']}")
                    else:
                        col2.success(f"🟢 {result['decision']}")
                    st.caption(
                        f"Threshold used: {result['threshold_used']}  |  "
                        f"Time: {result['timestamp']}"
                    )
                else:
                    st.error(err)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    else:
        feat_vals = {}
        v_feats = [f"V{i}" for i in range(1, 29)]
        for row_start in range(0, len(v_feats), 4):
            row_feats = v_feats[row_start:row_start + 4]
            cols = st.columns(4)
            for col, f in zip(cols, row_feats):
                feat_vals[f] = col.slider(f, -5.0, 5.0, 0.0, 0.01, key=f)
        feat_vals["Amount"] = st.slider("Amount ($)", 0.0, 5000.0, 100.0, 1.0)

        if st.button("🚀 Score Transaction", use_container_width=True):
            result, err = api_post("/predict", {"features": feat_vals})
            if result:
                prob = result["fraud_probability"]
                c1, c2 = st.columns(2)
                c1.metric("Fraud Probability", f"{prob:.4f}")
                if result["decision"] == "FRAUD":
                    c2.error(f"🔴 {result['decision']}")
                else:
                    c2.success(f"🟢 {result['decision']}")
            else:
                st.error(err)


# ══════════════════════════════════════════════════════════════
# TAB 3 — BATCH SCORING
# ══════════════════════════════════════════════════════════════

with tab3:
    st.header("📁 Batch Scoring")
    st.markdown(
        "Upload a CSV with columns V1–V28 + Amount (optionally `Class` for recall)."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_preview = pd.read_csv(uploaded)
        st.write(f"**Preview** — {len(df_preview):,} rows · {len(df_preview.columns)} columns")
        st.dataframe(df_preview.head(5), use_container_width=True)

        if st.button("🚀 Score All Rows", use_container_width=True):
            with st.spinner("Scoring..."):
                uploaded.seek(0)
                try:
                    r = requests.post(
                        f"{api_url}/predict/csv",
                        headers=headers,
                        files={"file": (uploaded.name, uploaded, "text/csv")},
                        timeout=60,
                    )
                    if r.status_code == 200:
                        result = r.json()
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Rows Scored", f"{result['rows_scored']:,}")
                        c2.metric("Fraud Detected", result["fraud_count"])
                        c3.metric("Fraud Rate", f"{result['fraud_rate']:.2%}")

                        if result.get("recall") is not None:
                            recall = result["recall"]
                            if recall < 0.70:
                                st.warning(f"⚠️ Recall: {recall:.1%} — below 70% target.")
                            else:
                                st.success(f"✅ Recall: {recall:.1%}")

                        res_df = pd.DataFrame(result["results"])
                        csv_out = res_df.to_csv(index=False)
                        st.download_button(
                            "⬇️ Download Scored CSV", csv_out,
                            file_name="sentinel_scored.csv", mime="text/csv",
                        )
                        st.dataframe(res_df.head(20), use_container_width=True)
                    else:
                        st.error(f"HTTP {r.status_code}: {r.text[:300]}")
                except Exception as e:
                    st.error(str(e))

    st.markdown("---")
    st.markdown("#### Generate & score a demo batch")
    n_demo = st.slider("Number of demo transactions", 10, 500, 50)
    if st.button("Generate & Score Demo Batch"):
        demo_txns = [
            {f"V{i}": round(float(np.random.normal(0, 1)), 4) for i in range(1, 29)}
            | {"Amount": round(abs(float(np.random.exponential(100))), 2)}
            for _ in range(n_demo)
        ]
        result, err = api_post("/predict/batch", {"transactions": demo_txns})
        if result:
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows Scored", result["count"])
            c2.metric("Fraud Detected", result["fraud_count"])
            c3.metric("Fraud Rate", f"{result['fraud_rate']:.2%}")
        else:
            st.error(err)


# ══════════════════════════════════════════════════════════════
# TAB 4 — DRIFT MONITOR
# ══════════════════════════════════════════════════════════════

with tab4:
    st.header("📈 Drift Monitor")
    st.markdown(
        "Upload production data (CSV) to check feature distribution drift "
        "against the reference baseline. PSI + KS run across all 38 model features "
        "(28 PCA components + Amount + 9 engineered features) — not just on "
        "prediction scores, which is what most lightweight monitors check."
    )

    drift_file = st.file_uploader("Upload production CSV", type=["csv"], key="drift_upload")
    if drift_file:
        df_drift = pd.read_csv(drift_file)
        st.write(f"Uploaded: {len(df_drift):,} rows")
        feat_cols    = [c for c in df_drift.columns if c != "Class"]
        transactions = df_drift[feat_cols].fillna(0).to_dict(orient="records")

        if st.button("🔍 Run Drift Report", use_container_width=True):
            with st.spinner("Analysing drift..."):
                result, err = api_post("/drift/report", {"transactions": transactions})

            if result:
                level = result["alert_level"]
                psi   = result["mean_psi"]
                if level == "CRITICAL":
                    st.error(f"🔴 CRITICAL — Mean PSI: {psi:.4f}")
                elif level == "WARNING":
                    st.warning(f"🟡 WARNING — Mean PSI: {psi:.4f}")
                elif level == "INFO":
                    st.info(f"🔵 INFO — Mean PSI: {psi:.4f}")
                else:
                    st.success(f"🟢 OK — Mean PSI: {psi:.4f}")

                st.markdown(f"**Recommended action:** {result['recommended_action']}")

                if result.get("critical_features"):
                    st.error("Critical features: " + ", ".join(result["critical_features"]))
                if result.get("warning_features"):
                    st.warning("Warning features: " + ", ".join(result["warning_features"]))

                if result.get("per_feature"):
                    df_feat = (
                        pd.DataFrame(result["per_feature"])
                        .T.reset_index()
                        .rename(columns={"index": "feature"})
                        .sort_values("psi", ascending=False)
                    )
                    st.dataframe(df_feat, use_container_width=True)
            else:
                st.error(err)

    st.markdown("---")
    st.subheader("⚙️ Threshold Optimisation")
    st.markdown(
        "Re-optimise the decision threshold using production labels.  \n"
        "θ* = argmax Recall(θ) − λ·FPR(θ)"
    )

    thresh_file = st.file_uploader(
        "Upload CSV with `probability` and `label` columns",
        type=["csv"], key="thresh_upload"
    )
    lam = st.slider("λ — FP cost weight (0.1 = standard fraud)", 0.01, 0.50, 0.10, 0.01)

    if thresh_file and st.button("Optimise Threshold"):
        df_t = pd.read_csv(thresh_file)
        if "probability" not in df_t.columns or "label" not in df_t.columns:
            st.error("CSV must have `probability` and `label` columns.")
        else:
            result, err = api_post("/threshold/optimize", {
                "lambda_cost":   lam,
                "labels":        df_t["label"].tolist(),
                "probabilities": df_t["probability"].tolist(),
            })
            if result:
                c1, c2, c3 = st.columns(3)
                c1.metric("New Threshold", result["optimal_threshold"])
                c2.metric("Recall", f"{result['recall_at_optimal']:.1%}")
                c3.metric("FPR", f"{result['fpr_at_optimal']:.3%}")
                st.success(result["message"])
            else:
                st.error(err)


# ══════════════════════════════════════════════════════════════
# TAB 5 — SETTINGS
# ══════════════════════════════════════════════════════════════

with tab5:
    st.header("⚙️ System Settings")

    if st.button("🔄 Reload Health Check"):
        data, err = api_get("/health")
        if data:
            st.session_state["health"] = data
        else:
            st.error(err)

    health = st.session_state.get("health", health_data or {})
    if health:
        st.json(health)

    st.markdown("---")
    st.subheader("📤 Upload Reference Data")
    st.markdown(
        "Replace the drift baseline with a new reference CSV.  \n"
        "Required columns: V1–V28 + Amount + Class (0 = legitimate, 1 = fraud)."
    )

    ref_file = st.file_uploader("Reference CSV", type=["csv"], key="ref_upload")
    if ref_file and st.button("Upload Reference"):
        df_ref = pd.read_csv(ref_file)
        if "Class" not in df_ref.columns:
            st.error("CSV must have a `Class` column (0 = legitimate, 1 = fraud).")
        else:
            feat_cols = [c for c in df_ref.columns if c != "Class"]
            result, err = api_post("/reference", {
                "data":   df_ref[feat_cols].fillna(0).to_dict(orient="records"),
                "labels": df_ref["Class"].tolist(),
            })
            if result:
                st.success(f"✅ Reference baseline updated — {result['rows']:,} rows loaded.")
            else:
                st.error(err)

    st.markdown("---")
    st.subheader("✅ Resolve Alerts")
    alert_id = st.number_input("Alert ID to resolve", min_value=1, step=1)
    if st.button("Resolve Alert"):
        result, err = api_post(f"/alerts/{int(alert_id)}/resolve", {})
        if result:
            st.success(result["message"])
        else:
            st.error(err)
