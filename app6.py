"""
GenAI BNPL Risk Guard Dashboard
================================
Clean, deployable Streamlit app — no ngrok, no Colab, no threading.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime, timedelta
import random
import math

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BNPL Risk Guard · Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLES  (deep blue / glassmorphism)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #020b18 0%, #041429 40%, #061d3a 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #061830 0%, #04122a 100%);
    border-right: 1px solid rgba(56, 139, 253, 0.15);
}
[data-testid="stSidebar"] .stMarkdown, 
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider label {
    color: #94a3b8 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(15, 40, 80, 0.55);
    border: 1px solid rgba(56, 139, 253, 0.22);
    border-radius: 12px;
    padding: 16px 20px;
    backdrop-filter: blur(12px);
}
[data-testid="stMetricLabel"] { color: #7fa8d4 !important; font-size: 0.78rem; font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #e2f0ff !important; font-size: 1.9rem !important; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* Section headers */
h1 { color: #e2f0ff !important; font-weight: 700; letter-spacing: -0.02em; }
h2 { color: #c5daf5 !important; font-weight: 600; }
h3 { color: #7fa8d4 !important; font-weight: 500; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.06em; }

/* Glass panels */
.glass-card {
    background: rgba(12, 34, 65, 0.6);
    border: 1px solid rgba(56, 139, 253, 0.18);
    border-radius: 14px;
    padding: 22px 26px;
    backdrop-filter: blur(16px);
    margin-bottom: 18px;
}

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.05em;
    font-family: 'JetBrains Mono', monospace;
}
.risk-LOW    { background: rgba(16,185,129,0.15); border:1px solid #10b981; color:#6ee7b7; }
.risk-MEDIUM { background: rgba(245,158,11,0.15); border:1px solid #f59e0b; color:#fcd34d; }
.risk-HIGH   { background: rgba(239,68,68,0.15);  border:1px solid #ef4444; color:#fca5a5; }
.risk-CRITICAL { background: rgba(139,0,0,0.25); border:1px solid #dc2626; color:#ff8080; }

/* Score circle */
.score-ring {
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
}
.score-number { font-size: 3.2rem; font-weight: 700; line-height: 1; }
.score-label  { font-size: 0.72rem; color: #7fa8d4; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 4px; }

/* Dividers */
hr { border-color: rgba(56,139,253,0.15) !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1a5fb4 0%, #0d3d73 100%);
    color: #e2f0ff;
    border: 1px solid rgba(56,139,253,0.4);
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 10px 28px;
    letter-spacing: 0.03em;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2265c4 0%, #154d8d 100%);
    border-color: rgba(56,139,253,0.7);
    transform: translateY(-1px);
}

/* Tabs */
[data-baseweb="tab-list"] { background: rgba(6,24,48,0.7); border-radius: 10px; padding: 4px; gap: 4px; border: 1px solid rgba(56,139,253,0.15); }
[data-baseweb="tab"] { border-radius: 7px !important; color: #7fa8d4 !important; font-size: 0.82rem; font-weight: 500; }
[aria-selected="true"] { background: rgba(26,95,180,0.45) !important; color: #e2f0ff !important; }

/* Selectbox / slider accent */
[data-baseweb="select"] { background: rgba(8,28,56,0.8) !important; border-color: rgba(56,139,253,0.25) !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background: #1a5fb4 !important; }

/* Expander */
[data-testid="stExpander"] {
    background: rgba(8,24,50,0.5);
    border: 1px solid rgba(56,139,253,0.15);
    border-radius: 10px;
}

/* Info / warning boxes */
.stAlert { border-radius: 10px; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #020b18; }
::-webkit-scrollbar-thumb { background: #1a5fb4; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS / COLOUR PALETTE
# ─────────────────────────────────────────────
PALETTE = {
    "bg":        "#020b18",
    "panel":     "#061830",
    "accent1":   "#1a5fb4",
    "accent2":   "#388bfd",
    "low":       "#10b981",
    "medium":    "#f59e0b",
    "high":      "#ef4444",
    "critical":  "#8b0000",
    "text":      "#e2f0ff",
    "muted":     "#7fa8d4",
}

def mpl_style():
    """Apply consistent dark theme to every matplotlib figure."""
    plt.rcParams.update({
        "figure.facecolor":   PALETTE["bg"],
        "axes.facecolor":     PALETTE["panel"],
        "axes.edgecolor":     "#1a3a6e",
        "axes.labelcolor":    PALETTE["muted"],
        "axes.titlecolor":    PALETTE["text"],
        "xtick.color":        PALETTE["muted"],
        "ytick.color":        PALETTE["muted"],
        "grid.color":         "#0d2a50",
        "grid.linestyle":     "--",
        "grid.linewidth":     0.6,
        "text.color":         PALETTE["text"],
        "font.family":        "DejaVu Sans",
        "legend.facecolor":   PALETTE["panel"],
        "legend.edgecolor":   "#1a3a6e",
    })


# ─────────────────────────────────────────────
# AI / RISK ENGINE
# ─────────────────────────────────────────────

def compute_risk_score(
    credit_score: int,
    transaction_amount: float,
    monthly_income: float,
    existing_debt: float,
    payment_history: float,   # % on-time  0-100
    num_missed_payments: int,
    age: int,
    employment_status: str,
    loan_tenure: int,         # months
) -> dict:
    score = 0.0
    contrib = {}

    # 1. Credit score  (weight 30)
    credit_norm = max(0, (credit_score - 300) / (850 - 300))
    credit_contrib = (1 - credit_norm) * 30
    contrib["Credit Score"] = round(credit_contrib, 2)
    score += credit_contrib

    # 2. DTI – debt-to-income  (weight 20)
    dti = existing_debt / max(monthly_income, 1)
    dti_contrib = min(dti * 25, 20)
    contrib["Debt-to-Income"] = round(dti_contrib, 2)
    score += dti_contrib

    # 3. Transaction / income ratio  (weight 15)
    tir = transaction_amount / max(monthly_income, 1)
    tir_contrib = min(tir * 15, 15)
    contrib["Transaction Burden"] = round(tir_contrib, 2)
    score += tir_contrib

    # 4. Payment history  (weight 15)
    ph_contrib = (1 - payment_history / 100) * 15
    contrib["Payment History"] = round(ph_contrib, 2)
    score += ph_contrib

    # 5. Missed payments  (weight 10)
    miss_contrib = min(num_missed_payments * 2, 10)
    contrib["Missed Payments"] = round(miss_contrib, 2)
    score += miss_contrib

    # 6. Employment  (weight 5)
    emp_map = {"Full-time": 0, "Part-time": 2.5, "Self-employed": 3, "Unemployed": 5, "Student": 4}
    emp_contrib = emp_map.get(employment_status, 2)
    contrib["Employment"] = round(emp_contrib, 2)
    score += emp_contrib

    # 7. Age factor  (weight 3)
    age_contrib = 3 if age < 22 else (1.5 if age < 26 else 0)
    contrib["Age Risk"] = round(age_contrib, 2)
    score += age_contrib

    # 8. Loan tenure  (weight 2)
    tenure_contrib = min(loan_tenure / 24, 1) * 2
    contrib["Loan Tenure"] = round(tenure_contrib, 2)
    score += tenure_contrib

    score = round(min(max(score, 0), 100), 1)

    if score < 25:
        tier, colour = "LOW", PALETTE["low"]
    elif score < 50:
        tier, colour = "MEDIUM", PALETTE["medium"]
    elif score < 75:
        tier, colour = "HIGH", PALETTE["high"]
    else:
        tier, colour = "CRITICAL", PALETTE["critical"]

    pod = round(1 / (1 + math.exp(-0.1 * (score - 50))) * 100, 1)

    base_limit = monthly_income * 0.3
    limit_multiplier = max(0.1, 1 - score / 100)
    recommended_limit = round(base_limit * limit_multiplier, -2)

    return {
        "score": score,
        "tier": tier,
        "colour": colour,
        "pod": pod,
        "contributions": contrib,
        "recommended_limit": recommended_limit,
    }


def generate_synthetic_portfolio(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    credit_scores = rng.integers(300, 850, n)
    incomes       = rng.integers(15_000, 200_000, n)
    debts         = (rng.random(n) * 0.6 * incomes).astype(int)
    txn_amounts   = rng.integers(500, 80_000, n)
    pay_hist      = rng.uniform(40, 100, n)
    missed        = rng.integers(0, 10, n)
    ages          = rng.integers(18, 65, n)
    emp_choices   = ["Full-time", "Part-time", "Self-employed", "Unemployed", "Student"]
    employment    = rng.choice(emp_choices, n)
    tenures       = rng.integers(1, 24, n)

    rows = []
    for i in range(n):
        r = compute_risk_score(
            credit_scores[i], txn_amounts[i], incomes[i], debts[i],
            pay_hist[i], missed[i], ages[i], employment[i], tenures[i]
        )
        rows.append({
            "credit_score":    credit_scores[i],
            "income":          incomes[i],
            "debt":            debts[i],
            "txn_amount":      txn_amounts[i],
            "payment_history": round(pay_hist[i], 1),
            "missed_payments": missed[i],
            "age":              ages[i],
            "employment":      employment[i],
            "risk_score":      r["score"],
            "risk_tier":       r["tier"],
            "pod":              r["pod"],
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────

def chart_risk_donut(tier_counts: dict):
    mpl_style()
    labels = list(tier_counts.keys())
    sizes  = list(tier_counts.values())
    colors = {
        "LOW":      PALETTE["low"],
        "MEDIUM":   PALETTE["medium"],
        "HIGH":     PALETTE["high"],
        "CRITICAL": PALETTE["critical"],
    }
    clr = [colors.get(l, "#888") for l in labels]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor=PALETTE["bg"])
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=clr,
        autopct="%1.1f%%", startangle=90,
        pctdistance=0.78, wedgeprops=dict(width=0.52, edgecolor=PALETTE["bg"], linewidth=2),
    )
    for at in autotexts:
        at.set_color(PALETTE["text"])
        at.set_fontsize(9)
        at.set_fontweight("bold")

    ax.legend(
        wedges, [f"{l} ({v})" for l, v in zip(labels, sizes)],
        loc="lower center", bbox_to_anchor=(0.5, -0.08),
        ncol=2, fontsize=8, framealpha=0.0,
    )
    ax.set_title("Risk Tier Distribution", color=PALETTE["text"], fontsize=11, pad=12)
    fig.tight_layout()
    return fig


def chart_score_histogram(df: pd.DataFrame):
    mpl_style()
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])

    bins = np.linspace(0, 100, 26)
    n, edges, patches = ax.hist(df["risk_score"], bins=bins, edgecolor=PALETTE["bg"], linewidth=0.5)
    for patch, left in zip(patches, edges[:-1]):
        if left < 25:      patch.set_facecolor(PALETTE["low"])
        elif left < 50:    patch.set_facecolor(PALETTE["medium"])
        elif left < 75:    patch.set_facecolor(PALETTE["high"])
        else:              patch.set_facecolor(PALETTE["critical"])

    ax.axvline(df["risk_score"].mean(), color="#388bfd", lw=1.5, linestyle="--", label=f"Mean {df['risk_score'].mean():.1f}")
    ax.set_xlabel("Risk Score", fontsize=9)
    ax.set_ylabel("Applicants", fontsize=9)
    ax.set_title("Portfolio Risk Score Distribution", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y")
    fig.tight_layout()
    return fig


def chart_credit_vs_risk(df: pd.DataFrame):
    mpl_style()
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])
    tier_colours = {
        "LOW": PALETTE["low"], "MEDIUM": PALETTE["medium"],
        "HIGH": PALETTE["high"], "CRITICAL": PALETTE["critical"],
    }
    for tier, grp in df.groupby("risk_tier"):
        ax.scatter(grp["credit_score"], grp["risk_score"],
                   color=tier_colours[tier], alpha=0.55, s=18, label=tier)
    ax.set_xlabel("Credit Score", fontsize=9)
    ax.set_ylabel("Risk Score", fontsize=9)
    ax.set_title("Credit Score vs Risk Score", fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True)
    fig.tight_layout()
    return fig


def chart_feature_bar(contributions: dict):
    mpl_style()
    items  = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in items]
    values = [i[1] for i in items]

    cmap = plt.get_cmap("RdYlGn_r")
    max_v = max(values) if values else 1
    bar_colors = [cmap(v / max_v) for v in values]

    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])
    bars = ax.barh(labels, values, color=bar_colors, edgecolor=PALETTE["bg"], linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8, color=PALETTE["text"])
    ax.set_xlabel("Risk Contribution (pts)", fontsize=9)
    ax.set_title("Feature Risk Contributions", fontsize=11)
    ax.invert_yaxis()
    ax.grid(axis="x")
    fig.tight_layout()
    return fig


def chart_pod_gauge(pod: float):
    mpl_style()
    fig, ax = plt.subplots(figsize=(4, 2.4), facecolor=PALETTE["bg"], subplot_kw={"aspect": "equal"})
    ax.set_facecolor(PALETTE["bg"])
    ax.axis("off")

    theta = np.linspace(0, np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#1a3a6e", linewidth=14, solid_capstyle="round")

    frac   = pod / 100
    theta2 = np.linspace(0, np.pi * frac, 200)
    colour = PALETTE["low"] if pod < 33 else (PALETTE["medium"] if pod < 66 else PALETTE["high"])
    ax.plot(np.cos(theta2), np.sin(theta2), color=colour, linewidth=14, solid_capstyle="round")

    ax.text(0, 0.05, f"{pod}%", ha="center", va="center",
            fontsize=20, fontweight="bold", color=PALETTE["text"], family="DejaVu Sans")
    ax.text(0, -0.28, "Probability of Default", ha="center", va="center",
            fontsize=7.5, color=PALETTE["muted"])
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.5, 1.25)
    fig.tight_layout(pad=0.2)
    return fig


def chart_heatmap(df: pd.DataFrame):
    mpl_style()
    cols  = ["credit_score", "income", "debt", "txn_amount", "payment_history", "risk_score", "pod"]
    corr  = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap,
                ax=ax, linewidths=0.5, linecolor="#0d1f3c",
                annot_kws={"size": 8}, vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.75})
    ax.set_title("Feature Correlation Matrix", fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    return fig


def chart_income_risk_violin(df: pd.DataFrame):
    mpl_style()
    fig, ax = plt.subplots(figsize=(6, 3.8), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])
    tier_order  = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    tier_colors = [PALETTE["low"], PALETTE["medium"], PALETTE["high"], PALETTE["critical"]]
    data_by_tier = [df[df["risk_tier"] == t]["income"].values / 1000 for t in tier_order]
    parts = ax.violinplot(data_by_tier, showmedians=True, widths=0.7)
    for pc, col in zip(parts["bodies"], tier_colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color(PALETTE["text"])
    parts["cbars"].set_color(PALETTE["muted"])
    parts["cmins"].set_color(PALETTE["muted"])
    parts["cmaxes"].set_color(PALETTE["muted"])
    ax.set_xticks(range(1, len(tier_order) + 1))
    ax.set_xticklabels(tier_order, fontsize=9)
    ax.set_ylabel("Monthly Income (₹ 000s)", fontsize=9)
    ax.set_title("Income Distribution by Risk Tier", fontsize=11)
    ax.grid(axis="y")
    fig.tight_layout()
    return fig


def chart_timeline_risk(n_days: int = 30, seed: int = 7):
    mpl_style()
    rng   = np.random.default_rng(seed)
    dates = [datetime.today() - timedelta(days=n_days - i) for i in range(n_days)]
    volume     = rng.integers(40, 200, n_days)
    avg_score  = rng.uniform(30, 65, n_days)

    fig, ax1 = plt.subplots(figsize=(8, 3.5), facecolor=PALETTE["bg"])
    ax1.set_facecolor(PALETTE["panel"])
    ax2 = ax1.twinx()

    ax1.fill_between(dates, volume, alpha=0.25, color=PALETTE["accent2"])
    ax1.plot(dates, volume, color=PALETTE["accent2"], linewidth=1.5, label="Applications")
    ax1.set_ylabel("Applications", color=PALETTE["accent2"], fontsize=9)
    ax1.tick_params(axis="y", labelcolor=PALETTE["accent2"])

    ax2.plot(dates, avg_score, color=PALETTE["medium"], linewidth=1.5, linestyle="--", label="Avg Risk Score")
    ax2.set_ylabel("Avg Risk Score", color=PALETTE["medium"], fontsize=9)
    ax2.tick_params(axis="y", labelcolor=PALETTE["medium"])

    ax1.set_xlabel("Date", fontsize=9)
    ax1.set_title("Daily Application Volume & Avg Risk Score", fontsize=11)
    plt.xticks(rotation=20, ha="right", fontsize=7)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# CACHED DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_portfolio():
    return generate_synthetic_portfolio(350)


# ─────────────────────────────────────────────
# SIDEBAR  –  APPLICANT INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Risk Guard")
    st.markdown("**BNPL Risk Evaluation Engine**")
    st.markdown("---")

    st.markdown("### Applicant Details")

    credit_score = st.slider("Credit Score", 300, 850, 640, 10,
        help="CIBIL / Experian credit score of applicant")
    transaction_amount = st.slider("Transaction Amount (₹)", 500, 1_00_000, 15_000, 500,
        help="Value of purchase to be financed via BNPL")
    monthly_income = st.slider("Monthly Income (₹)", 5_000, 3_00_000, 45_000, 1_000,
        help="Applicant's net monthly income")
    existing_debt = st.slider("Existing Monthly Debt (₹)", 0, 1_50_000, 8_000, 500,
        help="Total existing EMI obligations per month")
    payment_history = st.slider("On-Time Payment History (%)", 0, 100, 78, 1,
        help="% of past payments made on time")
    num_missed = st.slider("Missed Payments (last 12 mo)", 0, 12, 1, 1)
    age = st.slider("Applicant Age", 18, 70, 29, 1)
    loan_tenure = st.slider("Loan Tenure (months)", 1, 24, 6, 1)

    employment_status = st.selectbox(
        "Employment Status",
        ["Full-time", "Part-time", "Self-employed", "Unemployed", "Student"],
        index=0,
    )

    st.markdown("---")
    evaluate_btn = st.button("⚡  Evaluate Risk", width="stretch")
    st.markdown("---")
    st.markdown(
        "<small style='color:#4a6fa5'>Powered by GenAI Risk Engine · v2.1<br>For demo/educational use only.</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# COMPUTE RESULT
# ─────────────────────────────────────────────
result = compute_risk_score(
    credit_score, transaction_amount, monthly_income, existing_debt,
    payment_history, num_missed, age, employment_status, loan_tenure
)
df = load_portfolio()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 10])
with col_title:
    st.markdown("# 🛡️ BNPL Risk Guard Dashboard")
    st.markdown(
        "<span style='color:#4a90d9;font-size:0.85rem'>"
        "GenAI-Powered Buy Now Pay Later · Credit Risk Intelligence Platform"
        "</span>",
        unsafe_allow_html=True,
    )

st.markdown("---")


# ─────────────────────────────────────────────
# TOP KPI ROW
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
tier_counts = df["risk_tier"].value_counts().to_dict()

k1.metric("Total Applications",   f"{len(df):,}",        delta="+12 today")
k2.metric("Avg Risk Score",        f"{df['risk_score'].mean():.1f}",  delta=f"{df['risk_score'].mean()-50:.1f} vs baseline")
k3.metric("High / Critical",       f"{tier_counts.get('HIGH',0)+tier_counts.get('CRITICAL',0):,}", delta_color="inverse", delta="-3 vs yesterday")
k4.metric("Avg Credit Score",      f"{int(df['credit_score'].mean()):,}")
k5.metric("Avg Prob. of Default",  f"{df['pod'].mean():.1f}%",        delta_color="inverse", delta="+1.2%")

st.markdown("---")


# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab_eval, tab_portfolio, tab_analysis = st.tabs([
    "⚡  Applicant Evaluation",
    "📊  Portfolio Overview",
    "🔬  Deep Analysis",
])


# ════════════════════════════════════════════
# TAB 1  –  APPLICANT EVALUATION
# ════════════════════════════════════════════
with tab_eval:
    st.markdown("### Applicant Risk Assessment")
    st.caption("Adjust sliders in the sidebar and click **Evaluate Risk** (or results update live below).")

    col_score, col_gauge, col_detail = st.columns([1.6, 1.6, 2.8])

    with col_score:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        tier   = result["tier"]
        score  = result["score"]
        colour = result["colour"]
        score_colour = PALETTE["low"] if score < 25 else (PALETTE["medium"] if score < 50 else (PALETTE["high"] if score < 75 else PALETTE["critical"]))
        st.markdown(
            f"""
            <div class="score-ring">
                <div class="score-number" style="color:{score_colour}">{score}</div>
                <div class="score-label">Risk Score / 100</div>
            </div>
            <br>
            <div style="text-align:center">
                <span class="risk-badge risk-{tier}">{tier} RISK</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Decision Summary**")
        if tier == "LOW":
            st.success("✅ **APPROVE** — Low default risk. Proceed with standard terms.")
        elif tier == "MEDIUM":
            st.warning("⚠️ **REVIEW** — Moderate risk. Consider reduced limit or co-signer.")
        elif tier == "HIGH":
            st.error("🚫 **CAUTION** — High default risk. Manual underwriter review required.")
        else:
            st.error("🔴 **DECLINE** — Critical risk. Auto-reject recommended.")

        st.markdown(f"**Recommended Credit Limit:** ₹ {result['recommended_limit']:,.0f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_gauge:
        st.pyplot(chart_pod_gauge(result["pod"]), width="stretch")

        st.markdown('<div class="glass-card" style="margin-top:14px">', unsafe_allow_html=True)
        st.markdown("**Input Summary**")
        data_summary = {
            "Credit Score":      credit_score,
            "Income (₹/mo)":     f"₹{monthly_income:,}",
            "Txn Amount":        f"₹{transaction_amount:,}",
            "Debt (₹/mo)":       f"₹{existing_debt:,}",
            "Payment History":   f"{payment_history}%",
            "Missed Payments":   num_missed,
            "Age":               age,
            "Employment":        employment_status,
            "Tenure":            f"{loan_tenure} mo",
        }
        for k, v in data_summary.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;font-size:0.8rem;"
                f"padding:3px 0;border-bottom:1px solid rgba(56,139,253,0.1)'>"
                f"<span style='color:#7fa8d4'>{k}</span>"
                f"<span style='color:#e2f0ff;font-weight:500'>{v}</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_detail:
        st.markdown("**Feature Risk Contributions**")
        st.pyplot(chart_feature_bar(result["contributions"]), width="stretch")

        with st.expander("📋  Raw Contribution Values"):
            contrib_df = pd.DataFrame(
                list(result["contributions"].items()),
                columns=["Feature", "Risk Points"]
            ).sort_values("Risk Points", ascending=False)
            st.dataframe(contrib_df, width="stretch", hide_index=True)

    # ── Risk Insight Cards ──────────────────
    st.markdown("### ⚡ AI Insights")
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        dti = existing_debt / max(monthly_income, 1)
        dti_flag = "🟢 Healthy" if dti < 0.3 else ("🟡 Elevated" if dti < 0.5 else "🔴 High")
        st.markdown(f"**Debt-to-Income Ratio**\n\n`{dti:.2f}` — {dti_flag}")
        st.markdown(f"<small>DTI below 0.3 is ideal. Yours is {'within safe range' if dti < 0.3 else 'above recommended threshold'}.</small>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with i2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        tir  = transaction_amount / max(monthly_income, 1)
        flag = "🟢 Low burden" if tir < 0.3 else ("🟡 Moderate" if tir < 0.7 else "🔴 High burden")
        st.markdown(f"**Transaction Burden**\n\n`{tir:.2f}x` monthly income — {flag}")
        st.markdown(f"<small>BNPL amount is {tir*100:.0f}% of monthly income.</small>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with i3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        miss_flag = "🟢 Excellent" if num_missed == 0 else ("🟡 Minor" if num_missed <= 2 else "🔴 Concerning")
        st.markdown(f"**Payment Behaviour**\n\n`{num_missed}` missed · {payment_history}% on-time — {miss_flag}")
        st.markdown(f"<small>{'No missed payments detected — positive signal.' if num_missed == 0 else f'{num_missed} missed payment(s) in past 12 months increases default probability.'}</small>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 2  –  PORTFOLIO OVERVIEW
# ════════════════════════════════════════════
with tab_portfolio:
    st.markdown("### Portfolio Risk Overview")
    st.caption(f"Synthetic portfolio of {len(df):,} BNPL applicants — auto-generated for demonstration.")

    col_donut, col_hist = st.columns([1, 1.6])
    with col_donut:
        ordered = {t: tier_counts.get(t, 0) for t in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]}
        st.pyplot(chart_risk_donut(ordered), width="stretch")
    with col_hist:
        st.pyplot(chart_score_histogram(df), width="stretch")

    st.markdown("---")
    col_scatter, col_violin = st.columns(2)
    with col_scatter:
        st.pyplot(chart_credit_vs_risk(df), width="stretch")
    with col_violin:
        st.pyplot(chart_income_risk_violin(df), width="stretch")

    st.markdown("---")
    st.markdown("### 📈 30-Day Application Timeline")
    st.pyplot(chart_timeline_risk(), width="stretch")

    st.markdown("---")
    st.markdown("### 📋 Sample Applicant Records")

    tier_filter = st.multiselect(
        "Filter by Risk Tier",
        ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        default=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    )
    display_df = df[df["risk_tier"].isin(tier_filter)].head(50).copy()
    display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]

    st.dataframe(
        display_df.style.background_gradient(subset=["Risk Score", "Pod"], cmap="RdYlGn_r"),
        width="stretch",
        height=320,
    )


# ════════════════════════════════════════════
# TAB 3  –  DEEP ANALYSIS
# ════════════════════════════════════════════
with tab_analysis:
    st.markdown("### 🔬 Deep Analysis")

    col_heat, col_stats = st.columns([2.2, 1])
    with col_heat:
        st.pyplot(chart_heatmap(df), width="stretch")

    with col_stats:
        st.markdown("**Portfolio Statistics**")
        stats_dict = {
            "Applicants":       f"{len(df):,}",
            "Avg Risk Score":   f"{df['risk_score'].mean():.1f}",
            "Std Dev":          f"{df['risk_score'].std():.1f}",
            "Min Score":        f"{df['risk_score'].min():.1f}",
            "Max Score":        f"{df['risk_score'].max():.1f}",
            "Median Score":     f"{df['risk_score'].median():.1f}",
            "Avg POD":          f"{df['pod'].mean():.1f}%",
            "Avg Credit Score": f"{int(df['credit_score'].mean()):,}",
            "Avg Income":       f"₹{int(df['income'].mean()):,}",
            "Avg Debt":         f"₹{int(df['debt'].mean()):,}",
        }
        for k, v in stats_dict.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;font-size:0.82rem;"
                f"padding:5px 0;border-bottom:1px solid rgba(56,139,253,0.1)'>"
                f"<span style='color:#7fa8d4'>{k}</span>"
                f"<span style='color:#e2f0ff;font-weight:600'>{v}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### 🧮 Segment Comparison")

    seg_col = st.selectbox("Segment By", ["employment", "risk_tier"])
    seg_metric = st.selectbox("Metric", ["risk_score", "pod", "credit_score", "income"])

    mpl_style()
    fig_seg, ax_seg = plt.subplots(figsize=(8, 3.8), facecolor=PALETTE["bg"])
    ax_seg.set_facecolor(PALETTE["panel"])
    seg_data = df.groupby(seg_col)[seg_metric].mean().sort_values(ascending=False)
    bar_c = [PALETTE["accent1"] if i % 2 == 0 else PALETTE["accent2"] for i in range(len(seg_data))]
    bars_s = ax_seg.bar(seg_data.index, seg_data.values, color=bar_c, edgecolor=PALETTE["bg"], linewidth=0.5, width=0.55)
    
    for b, v in zip(bars_s, seg_data.values):
        ax_seg.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9, color=PALETTE["text"])
                    
    ax_seg.set_xlabel(seg_col.replace("_", " ").title(), fontsize=9)
    ax_seg.set_ylabel(seg_metric.replace("_", " ").title(), fontsize=9)
    title_text = f"Average {seg_metric.replace('_',' ').title()} by {seg_col.replace('_',' ').title()}"
    ax_seg.set_title(title_text, fontsize=11)
    
    st.pyplot(fig_seg, width="stretch")
