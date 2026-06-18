"""
MACI Sentinel — Production AI Governance Demo
Maqasid AI · maqasidai.org
Hugging Face Spaces · API key protected

Demonstrates:
  1. PSI-based drift detection across production periods
  2. KS test distribution comparison
  3. Recall / AUC / FPR performance monitoring
  4. Cost-sensitive threshold optimization
  5. Governance documentation summary

Data: Synthetic only. No proprietary data processed or stored.
"""

import os
import io
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score, precision_score, roc_auc_score,
    f1_score, confusion_matrix
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ─── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MACI Sentinel · Production AI Governance",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Theme ───────────────────────────────────────────────────────────────
NAVY    = "#050F1E"
BLUE    = "#1A6FDB"
BLUE3   = "#5AA8FF"
TEAL    = "#0D9488"
TEAL2   = "#14B8A9"
GOLD    = "#C8961A"
CREAM   = "#EEF4FF"
MUTED   = "#8BA8CC"
RED     = "#EF4444"
AMBER   = "#F97316"
GREEN   = "#22C55E"

plt.rcParams.update({
    "figure.facecolor":  NAVY,
    "axes.facecolor":    "#081628",
    "axes.edgecolor":    "#1A3A5C",
    "axes.labelcolor":   MUTED,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        CREAM,
    "grid.color":        "#1A3A5C",
    "grid.linewidth":    0.5,
    "axes.grid":         True,
    "font.family":       "monospace",
})

# ─── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global */
  .stApp { background-color: #050F1E; color: #EEF4FF; }
  .stApp * { font-family: 'DM Mono', ui-monospace, monospace; }

  /* Header */
  .sentinel-header {
    background: linear-gradient(135deg, #081628 0%, #0A1A30 100%);
    border: 1px solid rgba(26,111,219,0.2);
    border-top: 3px solid #1A6FDB;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
  }
  .sentinel-header h1 {
    font-size: 2rem; font-weight: 700;
    color: #EEF4FF; margin: 0 0 0.25rem;
    letter-spacing: -0.02em;
  }
  .sentinel-header .tagline {
    font-size: 0.72rem; letter-spacing: 0.18em;
    text-transform: uppercase; color: #14B8A9;
  }
  .sentinel-header .sub {
    font-size: 0.84rem; color: #8BA8CC;
    margin-top: 0.5rem; line-height: 1.6;
  }

  /* Metric cards */
  .metric-card {
    background: #081628;
    border: 1px solid rgba(26,111,219,0.18);
    padding: 1rem 1.25rem;
    text-align: center;
  }
  .metric-val { font-size: 1.6rem; font-weight: 700; }
  .metric-lbl { font-size: 0.62rem; letter-spacing: 0.12em; text-transform: uppercase; color: #8BA8CC; margin-top: 0.15rem; }

  /* Alert boxes */
  .alert-critical {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-left: 3px solid #EF4444;
    padding: 0.85rem 1.1rem; margin: 0.75rem 0;
    font-size: 0.82rem;
  }
  .alert-warning {
    background: rgba(249,115,22,0.08);
    border: 1px solid rgba(249,115,22,0.3);
    border-left: 3px solid #F97316;
    padding: 0.85rem 1.1rem; margin: 0.75rem 0;
    font-size: 0.82rem;
  }
  .alert-ok {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.3);
    border-left: 3px solid #22C55E;
    padding: 0.85rem 1.1rem; margin: 0.75rem 0;
    font-size: 0.82rem;
  }
  .alert-info {
    background: rgba(26,111,219,0.08);
    border: 1px solid rgba(26,111,219,0.3);
    border-left: 3px solid #1A6FDB;
    padding: 0.85rem 1.1rem; margin: 0.75rem 0;
    font-size: 0.82rem;
  }

  /* Section headers */
  .section-label {
    font-size: 0.62rem; letter-spacing: 0.22em;
    text-transform: uppercase; color: #5AA8FF;
    margin-bottom: 0.4rem;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #081628;
    border-right: 1px solid rgba(26,111,219,0.18);
  }

  /* Buttons */
  .stButton > button {
    background: #1A6FDB; color: #fff;
    border: none; font-weight: 600;
    letter-spacing: 0.02em; transition: background 0.2s;
  }
  .stButton > button:hover { background: #2E8AF0; }

  /* Disclaimer */
  .disclaimer {
    font-size: 0.68rem; color: rgba(139,168,204,0.5);
    border-top: 1px solid rgba(26,111,219,0.12);
    padding-top: 0.75rem; margin-top: 1.5rem;
    letter-spacing: 0.04em;
  }

  /* Governance table */
  .gov-row {
    display: flex; justify-content: space-between;
    padding: 0.6rem 0; border-bottom: 1px solid rgba(26,111,219,0.12);
    font-size: 0.78rem;
  }
  .gov-row:last-child { border-bottom: none; }
</style>
""", unsafe_allow_html=True)

# ─── API key gate ─────────────────────────────────────────────────────────
VALID_KEYS = set(os.environ.get("SENTINEL_API_KEYS", "").split(","))
DEMO_KEY   = os.environ.get("SENTINEL_DEMO_KEY", "DEMO-SENTINEL-2026")
VALID_KEYS.add(DEMO_KEY)

def check_access():
    """Gate: require API key or demo key."""
    if "authenticated" in st.session_state and st.session_state.authenticated:
        return True

    st.markdown("""
    <div class="sentinel-header">
      <div class="tagline">Maqasid AI · MACI Sentinel</div>
      <h1>Production AI Governance</h1>
      <div class="sub">Detect · Explain · Govern — enter your access key to continue.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        key = st.text_input(
            "Access key",
            type="password",
            placeholder="Enter your Sentinel API key or demo key",
            help="Contact contact@maqasidai.org for access · Demo key available on maqasidai.org"
        )
        if st.button("Access Sentinel →"):
            if key.strip() in VALID_KEYS and key.strip():
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.markdown('<div class="alert-critical">⚠ Invalid key. Contact contact@maqasidai.org for access.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="alert-info">
        <strong>Demo access</strong><br/>
        Use key: <code>DEMO-SENTINEL-2026</code><br/><br/>
        For full API access:<br/>
        contact@maqasidai.org<br/>
        maqasidai.org/mlops-audit
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="disclaimer">MACI Sentinel uses synthetic data only in this demo. No proprietary data is processed or stored. API key protected.</div>', unsafe_allow_html=True)
    return False

# ─── Data generation ──────────────────────────────────────────────────────
@st.cache_data
def generate_data(n=50000, fraud_rate=0.0017, seed=42):
    np.random.seed(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    L = np.random.randn(n_legit, 28)
    L_amt  = np.random.exponential(50, (n_legit, 1))
    L_time = np.sort(np.random.uniform(0, 172800, n_legit)).reshape(-1, 1)
    Lx = np.hstack([L, L_time, L_amt])

    F = np.random.randn(n_fraud, 28)
    F[:, 0]  -= 2.5
    F[:, 3]  += 1.8
    F[:, 13] -= 3.2
    F_amt  = np.random.exponential(120, (n_fraud, 1))
    F_time = np.random.uniform(0, 172800, n_fraud).reshape(-1, 1)
    Fx = np.hstack([F, F_time, F_amt])

    X = np.vstack([Lx, Fx])
    y = np.hstack([np.zeros(n_legit), np.ones(n_fraud)])
    idx = np.random.permutation(len(y))
    cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    df = pd.DataFrame(X[idx], columns=cols)
    df["Class"] = y[idx].astype(int)
    return df

@st.cache_data
def inject_drift(n=10000, period=1, seed=42):
    np.random.seed(seed + period)
    df = generate_data(n=n, seed=seed + period)
    mag = period * 0.05
    for f in ["V1", "V4", "V14", "Amount"]:
        df[f] += np.random.normal(0, mag, len(df))
    if period >= 3:
        fi = df[df.Class == 1].index
        shift = (period - 2) * 0.5
        df.loc[fi, "V1"]  += shift
        df.loc[fi, "V14"] += shift
    return df

@st.cache_resource
def train_models(seed=42):
    df = generate_data(seed=seed)
    FEATURES = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    X = df[FEATURES].values
    y = df["Class"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=seed)
    lr.fit(X_tr_s, y_tr)

    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    xgb = XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=200,
                         scale_pos_weight=spw, eval_metric="auc",
                         random_state=seed, verbosity=0)
    xgb.fit(X_tr_s, y_tr)

    return scaler, lr, xgb, FEATURES, X_te_s, y_te

# ─── Monitoring functions ─────────────────────────────────────────────────
def calc_psi(ref, cur, bins=10):
    edges = np.linspace(min(ref.min(), cur.min()), max(ref.max(), cur.max()), bins + 1)
    r, _ = np.histogram(ref, bins=edges)
    c, _ = np.histogram(cur, bins=edges)
    rp = (r + 0.0001) / len(ref)
    cp = (c + 0.0001) / len(cur)
    return float(np.sum((cp - rp) * np.log(cp / rp)))

def psi_alert(psi):
    if psi > 0.25: return "CRITICAL", RED
    if psi > 0.10: return "WARNING",  AMBER
    return "OK", GREEN

def evaluate(model, scaler, df, features, threshold=0.5):
    X = scaler.transform(df[features].values)
    y = df["Class"].values
    if y.sum() == 0:
        return None
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    return {
        "recall":    recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "fpr":       ((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1),
        "auc":       roc_auc_score(y, prob) if len(np.unique(y)) > 1 else 0.0,
        "f1":        f1_score(y, pred, zero_division=0),
    }

def optimize_threshold(model, scaler, df, features, lam=0.2):
    X = scaler.transform(df[features].values)
    y = df["Class"].values
    prob = model.predict_proba(X)[:, 1]
    best_t, best_s = 0.5, -np.inf
    results = []
    for t in np.linspace(0.01, 0.99, 200):
        pred = (prob >= t).astype(int)
        rec = recall_score(y, pred, zero_division=0)
        fpr = ((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)
        s = rec - lam * fpr
        results.append({"threshold": t, "recall": rec, "fpr": fpr, "score": s})
        if s > best_s:
            best_s, best_t = s, t
    return best_t, pd.DataFrame(results)

# ─── Plot functions ───────────────────────────────────────────────────────
def plot_psi_heatmap(psi_data: dict, features: list):
    """PSI heatmap across periods."""
    mat = pd.DataFrame({f"P{p}": [psi_data[p].get(f, 0) for f in features] for p in sorted(psi_data)}, index=features)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="RdYlGn_r",
                vmin=0, vmax=0.5, linewidths=0.5, ax=ax,
                annot_kws={"size": 9})
    ax.set_title("PSI Heatmap — Feature Drift Across Production Periods", fontsize=12, color=CREAM, pad=12)
    ax.set_ylabel("")
    fig.tight_layout()
    return fig

def plot_performance(perf_log: list, model_name: str):
    """Recall + FPR across periods."""
    df = pd.DataFrame(perf_log)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Recall
    axes[0].plot(df.period, df.recall, "o-", color=BLUE3, lw=2, ms=7, label="Recall")
    axes[0].axhline(0.3, color=RED,   ls="--", lw=1.5, label="Alert threshold (0.30)")
    axes[0].fill_between(df.period, 0, 0.3, alpha=0.07, color=RED)
    axes[0].set_title(f"{model_name} — Fraud Recall", color=CREAM, fontsize=11)
    axes[0].set_xlabel("Period"); axes[0].set_ylabel("Recall")
    axes[0].set_ylim(-0.05, 1.05); axes[0].legend(fontsize=8)

    # FPR
    axes[1].plot(df.period, df.fpr, "s-", color=AMBER, lw=2, ms=7)
    axes[1].axhline(0.05, color=RED, ls="--", lw=1.5, label="FPR threshold (0.05)")
    axes[1].set_title(f"{model_name} — False Positive Rate", color=CREAM, fontsize=11)
    axes[1].set_xlabel("Period"); axes[1].set_ylabel("FPR")
    axes[1].set_ylim(-0.01, None); axes[1].legend(fontsize=8)

    fig.tight_layout()
    return fig

def plot_threshold_curve(opt_df: pd.DataFrame, best_t: float):
    """Threshold optimization curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(opt_df.threshold, opt_df.recall, color=BLUE3,  lw=2, label="Recall")
    axes[0].plot(opt_df.threshold, opt_df.fpr,    color=AMBER,  lw=2, label="FPR")
    axes[0].axvline(best_t, color=TEAL2, ls="--", lw=1.5, label=f"Optimal θ = {best_t:.3f}")
    axes[0].set_title("Recall vs FPR — Threshold Sweep", color=CREAM, fontsize=11)
    axes[0].set_xlabel("Threshold"); axes[0].legend(fontsize=8)

    axes[1].plot(opt_df.threshold, opt_df.score, color=TEAL2, lw=2)
    axes[1].axvline(best_t, color=TEAL2, ls="--", lw=1.5, label=f"Max score at θ = {best_t:.3f}")
    axes[1].set_title("Optimization Score [Recall − λ·FPR]", color=CREAM, fontsize=11)
    axes[1].set_xlabel("Threshold"); axes[1].legend(fontsize=8)

    fig.tight_layout()
    return fig

def plot_feature_distributions(df_ref: pd.DataFrame, df_cur: pd.DataFrame, feature: str, period: int):
    """Before/after feature distribution for a single feature."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(df_ref[feature], bins=60, alpha=0.55, color=BLUE,  density=True, label="Reference (P0)")
    ax.hist(df_cur[feature], bins=60, alpha=0.55, color=RED,   density=True, label=f"Production (P{period})")
    psi_val = calc_psi(df_ref[feature].values, df_cur[feature].values)
    level, _ = psi_alert(psi_val)
    ax.set_title(f"{feature} — PSI: {psi_val:.4f} [{level}]", color=CREAM, fontsize=11)
    ax.set_xlabel(feature); ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig

# ─── Sidebar ──────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown('<div class="section-label">MACI SENTINEL</div>', unsafe_allow_html=True)
        st.markdown("**Production AI Governance**")
        st.markdown(f"<span style='color:{TEAL2};font-size:0.72rem'>Detect · Explain · Govern</span>", unsafe_allow_html=True)
        st.divider()

        page = st.radio(
            "Navigation",
            ["Overview & Status", "Drift Detection", "Performance Monitor", "Threshold Optimization", "Governance Report"],
            label_visibility="collapsed"
        )
        st.divider()

        model_choice = st.selectbox("Model", ["XGBoost", "Logistic Regression"])
        period       = st.slider("Production period", 1, 5, 3)
        lam          = st.slider("Cost weight λ", 0.05, 0.50, 0.20, 0.05,
                                 help="λ = FPR cost relative to missed fraud. Lower = more recall-focused.")
        st.divider()

        st.markdown(f"""
        <div style='font-size:0.68rem;color:{MUTED};line-height:1.8'>
        <div style='color:{CREAM};margin-bottom:0.4rem'>Contact</div>
        contact@maqasidai.org<br/>
        maqasidai.org<br/><br/>
        <div style='color:{TEAL2}'>Synthetic data only</div>
        No data stored or transmitted
        </div>
        """, unsafe_allow_html=True)

    return page, model_choice, period, lam

# ─── Pages ────────────────────────────────────────────────────────────────

def page_overview(scaler, lr, xgb, features, X_te, y_te, df_base):
    st.markdown("""
    <div class="sentinel-header">
      <div class="tagline">Maqasid AI · MACI Sentinel · Production AI Governance Demo</div>
      <h1>🛡 MACI Sentinel</h1>
      <div class="sub">
        Detect silent model failure · Explain decisions · Produce governance documentation<br/>
        Synthetic data · Research-validated methodology · Global deployment
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Baseline metrics
    st.markdown('<div class="section-label">BASELINE PERFORMANCE — PERIOD 0</div>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    for model_label, model in [("XGBoost", xgb), ("Logistic Regression", lr)]:
        m = evaluate(model, scaler, df_base, features)
        if m:
            pass

    m_xgb = evaluate(xgb, scaler, df_base, features)
    m_lr  = evaluate(lr,  scaler, df_base, features)

    metrics = [
        ("ROC-AUC", f"{m_xgb['auc']:.4f}", BLUE3),
        ("Recall",  f"{m_xgb['recall']:.3f}", GREEN),
        ("FPR",     f"{m_xgb['fpr']:.3f}",   TEAL2),
        ("F1",      f"{m_xgb['f1']:.3f}",    GOLD),
        ("Fraud rate", "0.17%", AMBER),
    ]
    for col, (lbl, val, color) in zip([col1, col2, col3, col4, col5], metrics):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-val" style="color:{color}">{val}</div>
          <div class="metric-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # What this demo shows
    st.markdown('<div class="section-label">WHAT THIS DEMO SHOWS</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        st.markdown(f"""
        <div style='background:#081628;border:1px solid rgba(26,111,219,0.2);border-top:2px solid {BLUE};padding:1.1rem;'>
        <div style='font-size:0.6rem;letter-spacing:0.14em;text-transform:uppercase;color:{BLUE3};margin-bottom:0.5rem'>SENTINEL DETECT</div>
        <div style='font-weight:600;margin-bottom:0.4rem;color:{CREAM}'>Drift & Performance</div>
        <div style='font-size:0.78rem;color:{MUTED};line-height:1.6'>
        PSI-based drift detection across 5 production periods. 
        See how data distribution shifts before recall collapses.
        KS test + PSI heatmap + performance decay timeline.
        </div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div style='background:#081628;border:1px solid rgba(13,148,136,0.2);border-top:2px solid {TEAL};padding:1.1rem;'>
        <div style='font-size:0.6rem;letter-spacing:0.14em;text-transform:uppercase;color:{TEAL2};margin-bottom:0.5rem'>SENTINEL EXPLAIN</div>
        <div style='font-weight:600;margin-bottom:0.4rem;color:{CREAM}'>Threshold Optimization</div>
        <div style='font-size:0.78rem;color:{MUTED};line-height:1.6'>
        Cost-sensitive threshold optimization recovers recall 
        without retraining. Business cost weight λ controls
        the trade-off between recall and false positives.
        </div>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"""
        <div style='background:#081628;border:1px solid rgba(200,150,26,0.2);border-top:2px solid {GOLD};padding:1.1rem;'>
        <div style='font-size:0.6rem;letter-spacing:0.14em;text-transform:uppercase;color:{GOLD};margin-bottom:0.5rem'>SENTINEL GOVERN</div>
        <div style='font-weight:600;margin-bottom:0.4rem;color:{CREAM}'>Governance Documentation</div>
        <div style='font-size:0.78rem;color:{MUTED};line-height:1.6'>
        EU AI Act Articles 9–17 compliance mapping. 
        MACI Framework Shariah governance scoring.
        Audit-ready documentation produced automatically.
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f'<div class="disclaimer">Synthetic data only · No proprietary data processed or stored · maqasidai.org · contact@maqasidai.org</div>', unsafe_allow_html=True)


def page_drift(scaler, model, features, df_base, period, model_name):
    st.markdown(f'<div class="section-label">SENTINEL DETECT — DRIFT ANALYSIS · PERIOD {period}</div>', unsafe_allow_html=True)
    st.markdown(f"### Feature Distribution Drift — Production Period {period}")

    df_prod = inject_drift(period=period)
    monitor_features = ["V1", "V4", "V14", "Amount", "V2", "V3", "V10", "V12"]

    # PSI table
    rows = []
    for f in monitor_features:
        psi_val = calc_psi(df_base[f].values, df_prod[f].values)
        ks_stat, ks_p = stats.ks_2samp(df_base[f].values, df_prod[f].values)
        level, color = psi_alert(psi_val)
        rows.append({"Feature": f, "PSI": round(psi_val, 4), "KS stat": round(ks_stat, 4),
                     "KS p-value": round(ks_p, 6), "Alert": level})
    df_psi = pd.DataFrame(rows)
    mean_psi = df_psi["PSI"].mean()
    overall_level, overall_color = psi_alert(mean_psi)

    # Alert banner
    if overall_level == "CRITICAL":
        st.markdown(f'<div class="alert-critical">🚨 CRITICAL — Mean PSI: {mean_psi:.4f} · Immediate action required · {(df_psi["Alert"] == "CRITICAL").sum()} features in critical drift</div>', unsafe_allow_html=True)
    elif overall_level == "WARNING":
        st.markdown(f'<div class="alert-warning">⚠ WARNING — Mean PSI: {mean_psi:.4f} · Investigate features · Prepare retraining</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-ok">✓ STABLE — Mean PSI: {mean_psi:.4f} · No significant drift detected</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        # PSI table with color coding
        def style_row(row):
            if row["Alert"] == "CRITICAL": return ["background-color: rgba(239,68,68,0.1)"] * len(row)
            if row["Alert"] == "WARNING":  return ["background-color: rgba(249,115,22,0.08)"] * len(row)
            return [""] * len(row)
        st.dataframe(df_psi.style.apply(style_row, axis=1), use_container_width=True, height=300)

    with col2:
        # Summary metrics
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:8px">
          <div class="metric-val" style="color:{overall_color}">{mean_psi:.4f}</div>
          <div class="metric-lbl">Mean PSI — {overall_level}</div>
        </div>
        <div class="metric-card" style="margin-bottom:8px">
          <div class="metric-val" style="color:{RED}">{(df_psi['Alert']=='CRITICAL').sum()}</div>
          <div class="metric-lbl">Critical features</div>
        </div>
        <div class="metric-card">
          <div class="metric-val" style="color:{AMBER}">{(df_psi['Alert']=='WARNING').sum()}</div>
          <div class="metric-lbl">Warning features</div>
        </div>
        """, unsafe_allow_html=True)

    # Heatmap across all periods
    st.markdown("#### PSI Heatmap — All Periods")
    psi_all = {}
    for p in range(1, 6):
        df_p = inject_drift(period=p)
        psi_all[p] = {f: calc_psi(df_base[f].values, df_p[f].values) for f in monitor_features}
    st.pyplot(plot_psi_heatmap(psi_all, monitor_features))

    # Single feature distribution
    st.markdown("#### Feature Distribution — Reference vs Production")
    feat_sel = st.selectbox("Select feature", monitor_features, index=0)
    st.pyplot(plot_feature_distributions(df_base, df_prod, feat_sel, period))

    st.markdown('<div class="disclaimer">PSI thresholds: < 0.10 Stable · 0.10–0.25 Warning · > 0.25 Critical · Industry standard from credit risk analytics</div>', unsafe_allow_html=True)


def page_performance(scaler, model, features, df_base, model_name):
    st.markdown('<div class="section-label">SENTINEL DETECT — PERFORMANCE MONITORING</div>', unsafe_allow_html=True)
    st.markdown("### Model Performance Across Production Periods")

    perf_log = [{"period": 0, **evaluate(model, scaler, df_base, features)}]
    for p in range(1, 6):
        df_p = inject_drift(period=p)
        m = evaluate(model, scaler, df_p, features)
        if m:
            perf_log.append({"period": p, **m})
    df_perf = pd.DataFrame(perf_log)

    # Key metrics at selected vs baseline
    col1, col2, col3, col4 = st.columns(4)
    p0 = df_perf[df_perf.period == 0].iloc[0]
    p5 = df_perf[df_perf.period == 5].iloc[0]

    for col, lbl, val0, val5, color in [
        (col1, "Recall P0 → P5",    f"{p0.recall:.3f}", f"{p5.recall:.3f}", RED),
        (col2, "AUC P0 → P5",       f"{p0.auc:.4f}",   f"{p5.auc:.4f}",   BLUE3),
        (col3, "FPR P0 → P5",       f"{p0.fpr:.3f}",   f"{p5.fpr:.3f}",   AMBER),
        (col4, "F1 P0 → P5",        f"{p0.f1:.3f}",    f"{p5.f1:.3f}",    TEAL2),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-val" style="color:{color}">{val0} → {val5}</div>
          <div class="metric-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    if p5.recall == 0:
        st.markdown('<div class="alert-critical">🚨 SILENT FAILURE DETECTED — Recall has collapsed to 0%. The model is operational but detecting zero fraud. This is the pattern MACI Sentinel catches before it costs revenue.</div>', unsafe_allow_html=True)

    st.pyplot(plot_performance(perf_log, model_name))

    st.markdown("#### Period-by-Period Summary")
    display_cols = ["period", "recall", "precision", "fpr", "auc", "f1"]
    st.dataframe(
        df_perf[display_cols].round(4).rename(columns={"period": "Period", "recall": "Recall", "precision": "Precision", "fpr": "FPR", "auc": "AUC", "f1": "F1"}),
        use_container_width=True
    )


def page_threshold(scaler, model, features, period, lam, model_name):
    st.markdown('<div class="section-label">SENTINEL EXPLAIN — THRESHOLD OPTIMIZATION</div>', unsafe_allow_html=True)
    st.markdown(f"### Cost-Sensitive Threshold Optimization · Period {period}")

    df_prod = inject_drift(period=period)
    default  = evaluate(model, scaler, df_prod, features, threshold=0.5)
    best_t, opt_df = optimize_threshold(model, scaler, df_prod, features, lam=lam)
    optimized = evaluate(model, scaler, df_prod, features, threshold=best_t)

    st.markdown(f"""
    <div class="alert-info">
    <strong>Optimization formula:</strong>  θ* = argmax [ Recall(θ) − λ · FPR(θ) ]
    where λ = {lam:.2f} (business cost weight — lower values prioritize recall over FPR control)
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.markdown(f"""
    <div class="metric-card">
      <div class="metric-val" style="color:{MUTED}">0.50</div>
      <div class="metric-lbl">Default threshold</div>
    </div>
    """, unsafe_allow_html=True)
    col2.markdown(f"""
    <div class="metric-card">
      <div class="metric-val" style="color:{TEAL2}">{best_t:.3f}</div>
      <div class="metric-lbl">Optimized threshold</div>
    </div>
    """, unsafe_allow_html=True)
    col3.markdown(f"""
    <div class="metric-card">
      <div class="metric-val" style="color:{GREEN}">{optimized['recall']:.3f} vs {default['recall']:.3f}</div>
      <div class="metric-lbl">Recall: optimized vs default</div>
    </div>
    """, unsafe_allow_html=True)

    comp_df = pd.DataFrame({
        "Metric":    ["Recall", "Precision", "FPR", "F1"],
        "Default (θ=0.50)":    [f"{default[k]:.4f}"   for k in ["recall","precision","fpr","f1"]],
        f"Optimized (θ={best_t:.3f})": [f"{optimized[k]:.4f}" for k in ["recall","precision","fpr","f1"]],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    if optimized["recall"] > default["recall"]:
        st.markdown(f'<div class="alert-ok">✓ Threshold optimization recovered recall from {default["recall"]:.3f} to {optimized["recall"]:.3f} — without any model retraining or infrastructure changes.</div>', unsafe_allow_html=True)

    st.pyplot(plot_threshold_curve(opt_df, best_t))

    st.markdown(f"""
    <div class="alert-info" style="margin-top:1rem">
    <strong>Business interpretation:</strong><br/>
    λ = {lam:.2f} means a false positive costs {lam:.0%} as much as a missed fraud.
    For most fraud systems λ = 0.1–0.3 (missed fraud is significantly more costly than a false alert).
    Adjust λ in the sidebar to reflect your specific business cost assumptions.
    </div>
    """, unsafe_allow_html=True)


def page_governance(df_base, period, model_name):
    st.markdown('<div class="section-label">SENTINEL GOVERN — REGULATORY DOCUMENTATION</div>', unsafe_allow_html=True)
    st.markdown("### Governance & Compliance Summary")

    df_prod = inject_drift(period=period)
    monitor_features = ["V1", "V4", "V14", "Amount", "V2", "V3", "V10", "V12"]
    psi_scores = {f: calc_psi(df_base[f].values, df_prod[f].values) for f in monitor_features}
    mean_psi = np.mean(list(psi_scores.values()))
    critical_count = sum(1 for v in psi_scores.values() if v > 0.25)
    overall_level, _ = psi_alert(mean_psi)

    st.markdown(f"""
    <div class="alert-info">
    This governance summary maps MACI Sentinel findings to regulatory requirements.
    In a full Sentinel Govern engagement, this is produced as a structured PDF
    ready for your compliance team, legal counsel, or regulatory submission.
    </div>
    """, unsafe_allow_html=True)

    # EU AI Act mapping
    st.markdown("#### EU AI Act — Articles 9–17 Compliance Mapping")
    eu_items = [
        ("Article 9",  "Risk management system",           "✓ Documented" if overall_level != "CRITICAL" else "⚠ Gaps identified", "Drift monitoring and alert system documented"),
        ("Article 10", "Data and data governance",         "✓ Documented",       "Training data baseline and production distribution logged"),
        ("Article 11", "Technical documentation",          "✓ Produced",         "Full audit report with methodology and results"),
        ("Article 12", "Record-keeping",                   "✓ MLflow tracked",   "All monitoring runs logged with timestamps and metrics"),
        ("Article 13", "Transparency and information",     "✓ Produced",         "Plain-language executive summary included"),
        ("Article 14", "Human oversight",                  "⚠ Recommend",        "Alert thresholds defined; human review protocol recommended"),
        ("Article 17", "Quality management",               "✓ Ongoing",          "Monthly monitoring cycle with documented review process"),
    ]

    for art, title, status, note in eu_items:
        color = GREEN if "✓" in status else AMBER
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:flex-start;padding:0.65rem 0;border-bottom:1px solid rgba(26,111,219,0.1);font-size:0.8rem'>
          <div style='min-width:90px;color:{MUTED}'>{art}</div>
          <div style='flex:1;color:{CREAM}'>{title}</div>
          <div style='min-width:120px;text-align:right;color:{color}'>{status}</div>
          <div style='min-width:280px;text-align:right;color:{MUTED};font-size:0.72rem;padding-left:1rem'>{note}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # MACI Framework / Shariah scoring
    st.markdown("#### MACI Framework — Shariah Governance Scoring")
    st.markdown(f"""
    <div class="alert-info">
    The MACI (Maqasid AI Compliance Index) Framework maps AI governance
    objectives to Maqasid al-Shariah principles. For Islamic finance institutions,
    this provides explainability and auditability aligned with both regulatory
    requirements and Sharia board expectations.
    </div>
    """, unsafe_allow_html=True)

    maqasid_items = [
        ("Hifz al-Mal (Protection of Wealth)",    "Model reliability monitoring prevents fraud losses", "Active"),
        ("Hifz al-Nafs (Protection of Life)",      "Fair, non-discriminatory credit decisions",          "Active"),
        ("Hifz al-Aql (Protection of Intellect)", "Explainable AI — SHAP glass-box decisions",          "Active"),
        ("Hifz al-Nasl (Continuity)",              "Audit trail and governance documentation",           "Active"),
        ("Hifz al-Din (Values alignment)",         "Ethical AI deployment within Islamic values",        "Configured"),
    ]

    for principle, desc, status in maqasid_items:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;padding:0.65rem 0;border-bottom:1px solid rgba(13,148,136,0.1);font-size:0.8rem'>
          <div style='flex:2;color:{TEAL2}'>{principle}</div>
          <div style='flex:3;color:{MUTED};padding:0 1rem'>{desc}</div>
          <div style='color:{GREEN}'>{status}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Audit trail summary
    st.markdown("#### Audit Trail Summary")
    summary = {
        "Model":                model_name,
        "Production period":    f"Period {period}",
        "Mean PSI":             f"{mean_psi:.4f}",
        "Overall drift status": overall_level,
        "Critical features":    str(critical_count),
        "Monitoring method":    "PSI + KS test",
        "EU AI Act status":     "Aligned (Articles 9–17)",
        "MACI Framework":       "Active",
        "Report format":        "PDF + JSON export",
        "Generated":            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M UTC"),
    }
    for k, v in summary.items():
        val_color = RED if v == "CRITICAL" else (AMBER if v == "WARNING" else CREAM)
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid rgba(26,111,219,0.1);font-size:0.8rem'>
          <div style='color:{MUTED}'>{k}</div>
          <div style='color:{val_color};font-weight:500'>{v}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div style='background:#081628;border:1px solid rgba(13,148,136,0.2);border-top:2px solid {TEAL};padding:1.25rem;margin-top:1rem'>
    <div style='font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;color:{TEAL2};margin-bottom:0.5rem'>REQUEST FULL GOVERNANCE PACKAGE</div>
    <div style='font-size:0.84rem;color:{CREAM};margin-bottom:0.35rem'>In a full Sentinel Govern engagement:</div>
    <div style='font-size:0.76rem;color:{MUTED};line-height:1.8'>
    ✓ Complete EU AI Act Articles 9–17 documentation package (PDF)<br/>
    ✓ MACI Framework Shariah compliance scoring report<br/>
    ✓ DORA-aligned audit trail and logging structure<br/>
    ✓ Regulatory-ready technical records for submission<br/>
    ✓ Board-level governance summary
    </div>
    <div style='margin-top:1rem;font-size:0.76rem;color:{TEAL2}'>
    contact@maqasidai.org · maqasidai.org/mlops-audit
    </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    if not check_access():
        return

    scaler, lr, xgb, features, X_te, y_te = train_models()
    df_base = generate_data()

    page, model_choice, period, lam = sidebar()
    model      = xgb if model_choice == "XGBoost" else lr
    model_name = model_choice

    if page == "Overview & Status":
        page_overview(scaler, lr, xgb, features, X_te, y_te, df_base)
    elif page == "Drift Detection":
        page_drift(scaler, model, features, df_base, period, model_name)
    elif page == "Performance Monitor":
        page_performance(scaler, model, features, df_base, model_name)
    elif page == "Threshold Optimization":
        page_threshold(scaler, model, features, period, lam, model_name)
    elif page == "Governance Report":
        page_governance(df_base, period, model_name)

if __name__ == "__main__":
    main()
