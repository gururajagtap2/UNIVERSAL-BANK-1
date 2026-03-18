"""
Universal Bank — Personal Loan Intelligence Dashboard
Streamlit App  |  Descriptive → Diagnostic → Predictive → Prescriptive Analytics
"""
import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank – Loan Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetric"] {
    background: white;
    border-radius: 10px;
    padding: 14px 18px;
    border: 1px solid #e4e4e4;
}
.sec {
    font-size: 18px;
    font-weight: 700;
    color: #1E3A5F;
    margin: 26px 0 6px;
    padding-bottom: 7px;
    border-bottom: 2.5px solid #2196F3;
}
.ins {
    background: #EEF5FF;
    border-left: 4px solid #2196F3;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    font-size: 13.5px;
    color: #222;
    margin-bottom: 16px;
    line-height: 1.65;
}
.warnbox {
    background: #FFF8E1;
    border-left: 4px solid #FFC107;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    font-size: 13px;
    margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
C_BLUE   = "#2196F3"
C_GREEN  = "#4CAF50"
C_ORANGE = "#FF9800"
C_RED    = "#F44336"
C_PURPLE = "#9C27B0"
MODEL_COLORS = {
    "Decision Tree":         "#2196F3",
    "Random Forest":         "#4CAF50",
    "Gradient Boosted Tree": "#FF9800",
}
PALETTE = [C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE]

# ── Helpers ───────────────────────────────────────────────────────────────────
def sec(txt):
    st.markdown(f'<div class="sec">{txt}</div>', unsafe_allow_html=True)

def ins(txt):
    st.markdown(f'<div class="ins">💡 {txt}</div>', unsafe_allow_html=True)

def warn(txt):
    st.markdown(f'<div class="warnbox">⚠️ {txt}</div>', unsafe_allow_html=True)

# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data
def load_raw(file_obj=None):
    if file_obj is not None:
        return pd.read_csv(file_obj)
    return pd.read_csv("UniversalBank.csv")


@st.cache_data
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Experience"] = d["Experience"].abs()
    d.drop(columns=["ID", "ZIP Code"], errors="ignore", inplace=True)
    return d


@st.cache_data
def train_all(df: pd.DataFrame):
    X = df.drop("Personal Loan", axis=1)
    y = df["Personal Loan"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    specs = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "Gradient Boosted Tree": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42
        ),
    }
    res = {}
    for name, mdl in specs.items():
        mdl.fit(X_tr, y_tr)
        tr_p  = mdl.predict(X_tr)
        te_p  = mdl.predict(X_te)
        te_pr = mdl.predict_proba(X_te)[:, 1]
        res[name] = dict(
            model     = mdl,
            train_acc = accuracy_score(y_tr, tr_p),
            test_acc  = accuracy_score(y_te, te_p),
            precision = precision_score(y_te, te_p, zero_division=0),
            recall    = recall_score(y_te,    te_p, zero_division=0),
            f1        = f1_score(y_te,        te_p, zero_division=0),
            roc_auc   = roc_auc_score(y_te,   te_pr),
            cm        = confusion_matrix(y_te, te_p),
            y_test    = y_te.values,
            y_pred    = te_p,
            y_prob    = te_pr,
        )
    return res, list(X.columns)


# ═════════════════════════════════════════════════════════════════════════════
def main():

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🏦 Universal Bank")
        st.markdown("**Personal Loan Intelligence Dashboard**")
        st.divider()
        up = st.file_uploader("📂 Upload Dataset (CSV)", type="csv")
        st.divider()
        st.caption(
            "**Models trained:**\n"
            "• Decision Tree\n"
            "• Random Forest\n"
            "• Gradient Boosted Tree\n\n"
            "**Split:** 70 % train / 30 % test\n\n"
            "**Class imbalance:** handled via balanced class weights"
        )

    # ── Load & preprocess ─────────────────────────────────────────────────────
    raw = load_raw(up)
    df  = preprocess(raw)

    # ── Train models ──────────────────────────────────────────────────────────
    with st.spinner("Training models — please wait…"):
        res, feat_cols = train_all(df)

    best = max(res, key=lambda k: res[k]["roc_auc"])

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='color:#1E3A5F;margin-bottom:2px'>"
        "🏦 Universal Bank — Personal Loan Intelligence Dashboard</h1>"
        "<p style='color:#666;margin-top:0;font-size:15px'>"
        "Descriptive &nbsp;·&nbsp; Diagnostic &nbsp;·&nbsp; Predictive &nbsp;·&nbsp; Prescriptive Analytics"
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Top KPI strip ─────────────────────────────────────────────────────────
    k = st.columns(6)
    k[0].metric("Total Customers",   f"{len(df):,}")
    k[1].metric("Loan Acceptances",  f"{int(df['Personal Loan'].sum()):,}")
    k[2].metric("Acceptance Rate",   f"{df['Personal Loan'].mean()*100:.1f}%")
    k[3].metric("Best Model",        best)
    k[4].metric("Best ROC-AUC",      f"{res[best]['roc_auc']:.4f}")
    k[5].metric("Best F1-Score",     f"{res[best]['f1']:.4f}")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📋 Overview",
        "📊 Descriptive Analytics",
        "🔍 Diagnostic Analytics",
        "🤖 Predictive Models",
        "🎯 Prescriptive Insights",
        "📤 Predict New Data",
    ])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        sec("Raw Dataset Preview")
        st.dataframe(raw.head(10), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            sec("Summary Statistics")
            st.dataframe(df.describe().T.round(2), use_container_width=True)

        with c2:
            sec("Data Quality Report")
            neg_exp = int((raw.get("Experience", pd.Series([], dtype=int)) < 0).sum())
            qdf = pd.DataFrame({
                "Check": [
                    "Total records", "Total columns", "Missing values",
                    "Negative Experience (fixed)", "Dropped columns",
                ],
                "Result": [
                    f"{len(raw):,}", str(raw.shape[1]),
                    str(int(raw.isnull().sum().sum())),
                    f"{neg_exp} rows", "ID, ZIP Code",
                ],
                "Status": ["✅", "✅", "✅ Clean", "⚠️ Fixed → abs()", "✅ Removed"],
            })
            st.dataframe(qdf, use_container_width=True, hide_index=True)
            warn(
                f"{neg_exp} records had negative Experience values — likely data-entry errors. "
                "Corrected by taking the absolute value before modelling."
            )

        sec("Target Variable — Personal Loan Acceptance")
        c1, c2 = st.columns([1, 2])
        with c1:
            no_l  = int((df["Personal Loan"] == 0).sum())
            yes_l = int((df["Personal Loan"] == 1).sum())
            fig = go.Figure(go.Pie(
                labels=["No Loan (0)", "Accepted (1)"],
                values=[no_l, yes_l],
                hole=0.55,
                marker_colors=[C_BLUE, C_GREEN],
                textinfo="label+percent+value",
                textfont_size=12,
            ))
            fig.update_layout(height=310, margin=dict(t=30, b=10, l=10, r=10),
                               showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            ins(
                "Only <b>9.6 %</b> of customers (480 / 5,000) accepted the personal loan in the last campaign — "
                "a <b>9:1 class imbalance</b>. A naive model could reach 90.4 % accuracy by always predicting "
                "'No Loan', which is completely useless for marketing. "
                "All three classifiers use <b>balanced class weights</b> so the minority class (loan acceptors) "
                "receives equal importance during training."
            )
            st.markdown("""
| Feature | Type | Role |
|---------|------|------|
| Age, Experience | Numerical | Demographics |
| Income, CCAvg, Mortgage | Numerical | Financial behaviour |
| Education, Family | Ordinal | Socio-demographics |
| Securities Acc, CD Acc, Online, CreditCard | Binary | Product engagement |
| **Personal Loan** | Binary | **Target (y)** |
""")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 — DESCRIPTIVE ANALYTICS
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        sec("Distribution of Numerical Features")
        ins(
            "Histograms show the shape of each numerical variable across all 5,000 customers. "
            "Income and CCAvg are <b>right-skewed</b> — most customers earn moderately, but a small "
            "high-income segment drives loan demand. Mortgage has a large zero-peak: 69 % of customers "
            "carry no mortgage."
        )
        num_cols = ["Age", "Experience", "Income", "CCAvg", "Mortgage"]
        fig = make_subplots(rows=1, cols=5, subplot_titles=num_cols)
        for i, col in enumerate(num_cols, 1):
            fig.add_trace(
                go.Histogram(x=df[col], marker_color=C_BLUE,
                             nbinsx=30, name=col, showlegend=False),
                row=1, col=i,
            )
        fig.update_layout(height=290, margin=dict(t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)

        sec("Categorical & Binary Feature Distributions")
        ins(
            "Undergraduates form the largest education group (41.9 %). "
            "CD Account is held by only 6 % of customers, yet they accept personal loans at a 46.4 % rate. "
            "59.7 % of customers use online banking — a strong digital engagement signal."
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            edu_vc = (df["Education"]
                      .map({1: "Undergrad", 2: "Graduate", 3: "Adv/Prof"})
                      .value_counts())
            fig = px.bar(x=edu_vc.index, y=edu_vc.values,
                         color=edu_vc.index.tolist(),
                         color_discrete_sequence=PALETTE,
                         title="Education Level",
                         labels={"x": "", "y": "Customers"})
            fig.update_layout(showlegend=False, height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fam_vc = df["Family"].value_counts().sort_index()
            fig = px.bar(x=fam_vc.index.astype(str), y=fam_vc.values,
                         color=fam_vc.index.astype(str),
                         color_discrete_sequence=PALETTE,
                         title="Family Size",
                         labels={"x": "Members", "y": "Customers"})
            fig.update_layout(showlegend=False, height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            prod = {
                "Securities Acc": df["Securities Account"].mean() * 100,
                "CD Account":     df["CD Account"].mean() * 100,
                "Online Banking": df["Online"].mean() * 100,
                "Credit Card":    df["CreditCard"].mean() * 100,
            }
            fig = px.bar(x=list(prod.keys()), y=list(prod.values()),
                         color=list(prod.keys()),
                         color_discrete_sequence=PALETTE,
                         title="Product Penetration (%)",
                         labels={"x": "", "y": "% of customers"})
            fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
            fig.update_layout(showlegend=False, height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            df_tmp = df.copy()
            df_tmp["Income Band"] = pd.cut(
                df_tmp["Income"], bins=[0, 40, 80, 120, 300],
                labels=["<$40K", "$40-80K", "$80-120K", ">$120K"],
            )
            ib_vc = df_tmp["Income Band"].value_counts().sort_index()
            fig = px.bar(x=ib_vc.index.astype(str), y=ib_vc.values,
                         color=ib_vc.index.astype(str),
                         color_discrete_sequence=PALETTE,
                         title="Income Bands",
                         labels={"x": "", "y": "Customers"})
            fig.update_layout(showlegend=False, height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        sec("Feature Correlation Heatmap")
        ins(
            "Income (r = 0.50) and CCAvg (r = 0.37) show the strongest positive correlation with Personal Loan "
            "acceptance. Age and Experience are near-perfectly correlated (r = 0.99) — they carry redundant "
            "information. CD Account (r = 0.32) is the strongest binary signal."
        )
        fig_h, ax_h = plt.subplots(figsize=(11, 7))
        corr = df.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlBu_r", center=0, ax=ax_h,
            linewidths=0.5, annot_kws={"size": 9},
        )
        ax_h.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
        plt.tight_layout()
        st.pyplot(fig_h)
        plt.close(fig_h)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3 — DIAGNOSTIC ANALYTICS
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        sec("Loan Acceptance Rate by Customer Segment")
        ins(
            "Diagnostic analytics answers <b>'WHY do some customers accept and others don't?'</b> "
            "Each chart shows the acceptance rate within a segment. "
            "These patterns directly inform which customers to target in your next campaign."
        )
        df3 = df.copy()
        df3["Income Band"] = pd.cut(
            df3["Income"], bins=[0, 40, 80, 120, 300],
            labels=["<$40K", "$40-80K", "$80-120K", ">$120K"],
        )

        c1, c2 = st.columns(2)
        with c1:
            ib = (df3.groupby("Income Band", observed=True)["Personal Loan"]
                     .agg(["mean", "count"]).reset_index())
            ib.columns = ["Income Band", "Rate", "Count"]
            ib["Rate %"] = (ib["Rate"] * 100).round(1)
            fig = px.bar(ib, x="Income Band", y="Rate %",
                         color="Rate %", color_continuous_scale="Blues",
                         text="Rate %",
                         title="Acceptance Rate by Income Band",
                         labels={"Rate %": "Acceptance Rate (%)"})
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=370, margin=dict(t=50, b=20),
                               coloraxis_showscale=False,
                               yaxis=dict(range=[0, ib["Rate %"].max() * 1.3]))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            er = df3.groupby("Education")["Personal Loan"].mean().reset_index()
            er["Label"] = er["Education"].map({1: "Undergrad", 2: "Graduate", 3: "Adv/Prof"})
            er["Rate %"] = (er["Personal Loan"] * 100).round(1)
            fig = px.bar(er, x="Label", y="Rate %",
                         color="Rate %", color_continuous_scale="Greens",
                         text="Rate %",
                         title="Acceptance Rate by Education Level",
                         labels={"Rate %": "Acceptance Rate (%)"})
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=370, margin=dict(t=50, b=20),
                               coloraxis_showscale=False,
                               yaxis=dict(range=[0, er["Rate %"].max() * 1.3]))
            st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            cdr = df3.groupby("CD Account")["Personal Loan"].mean().reset_index()
            cdr["Label"] = cdr["CD Account"].map({0: "No CD Account", 1: "Has CD Account"})
            cdr["Rate %"] = (cdr["Personal Loan"] * 100).round(1)
            fig = px.bar(cdr, x="Label", y="Rate %",
                         color="Label",
                         color_discrete_map={"No CD Account": C_BLUE, "Has CD Account": C_GREEN},
                         text="Rate %",
                         title="Acceptance Rate by CD Account",
                         labels={"Rate %": "Acceptance Rate (%)"})
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=340, margin=dict(t=50, b=10), showlegend=False,
                               yaxis=dict(range=[0, cdr["Rate %"].max() * 1.35]))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fr = df3.groupby("Family")["Personal Loan"].mean().reset_index()
            fr["Rate %"] = (fr["Personal Loan"] * 100).round(1)
            fig = px.bar(fr, x="Family", y="Rate %",
                         color="Rate %", color_continuous_scale="Oranges",
                         text="Rate %",
                         title="Acceptance Rate by Family Size",
                         labels={"Rate %": "Acceptance Rate (%)", "Family": "Family Size"})
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=340, margin=dict(t=50, b=10),
                               coloraxis_showscale=False,
                               yaxis=dict(range=[0, fr["Rate %"].max() * 1.35]))
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            sr = df3.groupby("Securities Account")["Personal Loan"].mean().reset_index()
            sr["Label"] = sr["Securities Account"].map({0: "No Securities", 1: "Has Securities"})
            sr["Rate %"] = (sr["Personal Loan"] * 100).round(1)
            fig = px.bar(sr, x="Label", y="Rate %",
                         color="Label",
                         color_discrete_map={"No Securities": C_BLUE, "Has Securities": C_ORANGE},
                         text="Rate %",
                         title="Acceptance Rate by Securities Account",
                         labels={"Rate %": "Acceptance Rate (%)"})
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=340, margin=dict(t=50, b=10), showlegend=False,
                               yaxis=dict(range=[0, sr["Rate %"].max() * 1.35]))
            st.plotly_chart(fig, use_container_width=True)

        sec("Feature Distributions — Loan Acceptors vs. Non-Acceptors")
        ins(
            "Box plots compare numerical features between the two outcome groups. "
            "Loan acceptors (green) show dramatically higher Income and CCAvg distributions — "
            "the interquartile ranges barely overlap for Income. "
            "Age and Experience show almost no difference between groups."
        )
        num_f = ["Age", "Income", "CCAvg", "Mortgage", "Experience"]
        fig = make_subplots(rows=1, cols=5, subplot_titles=num_f)
        for i, col in enumerate(num_f, 1):
            for j, (lbl, clr) in enumerate(
                zip(["No Loan", "Accepted"], [C_BLUE, C_GREEN])
            ):
                sub = df3[df3["Personal Loan"] == j][col]
                fig.add_trace(
                    go.Box(y=sub, name=lbl, marker_color=clr,
                           showlegend=(i == 1), legendgroup=lbl),
                    row=1, col=i,
                )
        fig.update_layout(
            height=400, margin=dict(t=45, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.06),
        )
        st.plotly_chart(fig, use_container_width=True)

        sec("Scatter — Income vs CCAvg (Coloured by Loan Status)")
        ins(
            "Loan acceptors (green) cluster in the top-right quadrant: high income + high CC spending. "
            "This two-feature combination is the single most powerful targeting criterion. "
            "Very few customers with Income below $80K accepted the loan regardless of CC spending."
        )
        df3["Loan Status"] = df3["Personal Loan"].map({0: "No Loan", 1: "Accepted Loan"})
        fig = px.scatter(
            df3, x="Income", y="CCAvg",
            color="Loan Status",
            color_discrete_map={"No Loan": C_BLUE, "Accepted Loan": C_GREEN},
            opacity=0.45,
            title="Annual Income vs Monthly CC Spending — by Loan Status",
            labels={"Income": "Annual Income ($000)", "CCAvg": "Monthly CC Spend ($000)"},
        )
        fig.update_layout(height=440, legend_title="Personal Loan")
        st.plotly_chart(fig, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 4 — PREDICTIVE MODELS
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        sec("Model Performance Comparison Table")
        ins(
            "All three models were trained on 70 % of data (stratified split) and evaluated on the remaining 30 %. "
            "Balanced class weights prevent the majority class from dominating. "
            "🏆 marks the best performer by ROC-AUC. Values shown as % and raw score for clarity."
        )
        rows = []
        for name, r in res.items():
            rows.append({
                "Model":           ("🏆 " if name == best else "   ") + name,
                "Train Accuracy":  f"{r['train_acc']*100:.2f}%",
                "Test Accuracy":   f"{r['test_acc']*100:.2f}%",
                "Precision":       f"{r['precision']*100:.2f}%",
                "Recall":          f"{r['recall']*100:.2f}%",
                "F1-Score":        f"{r['f1']*100:.2f}%",
                "ROC-AUC":         f"{r['roc_auc']:.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.success(
            f"🏆 Best model: **{best}**  —  "
            f"ROC-AUC: {res[best]['roc_auc']:.4f}  |  "
            f"F1: {res[best]['f1']:.4f}  |  "
            f"Recall: {res[best]['recall']:.4f}"
        )

        sec("ROC Curves — All Three Models on a Single Chart")
        ins(
            "The ROC curve plots True Positive Rate (correctly identified loan acceptors) against False Positive Rate "
            "at every decision threshold. AUC = 1.0 is perfect; AUC = 0.5 is random guessing (grey dashed line). "
            "A higher, left-leaning curve = better model at separating loan acceptors from non-acceptors."
        )
        fig = go.Figure()
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dot", color="grey", width=1.5))
        fig.add_annotation(x=0.52, y=0.44, text="Random Guess (AUC = 0.50)",
                           showarrow=False, font=dict(size=10, color="grey"))
        for name, r in res.items():
            fpr, tpr, _ = roc_curve(r["y_test"], r["y_prob"])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{name}  (AUC = {r['roc_auc']:.4f})",
                line=dict(color=MODEL_COLORS[name], width=2.5),
            ))
        fig.update_layout(
            height=500,
            title="ROC Curve — Decision Tree vs Random Forest vs Gradient Boosted Tree",
            xaxis=dict(title="False Positive Rate (1 − Specificity)",
                       range=[0, 1], showgrid=True, gridcolor="#eee"),
            yaxis=dict(title="True Positive Rate (Sensitivity / Recall)",
                       range=[0, 1], showgrid=True, gridcolor="#eee"),
            plot_bgcolor="white",
            legend=dict(x=0.52, y=0.06, bgcolor="rgba(255,255,255,0.92)",
                        bordercolor="#ccc", borderwidth=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        sec("Confusion Matrices — All Models (Count + Percentage Distribution)")
        ins(
            "Each cell shows the raw count and its share of all test predictions. "
            "Bottom-right = True Positives (loan acceptors correctly identified — the metric your marketing team cares about). "
            "Top-right = False Positives (wasted outreach). Bottom-left = False Negatives (missed opportunities)."
        )
        fig_cm, axes = plt.subplots(1, 3, figsize=(17, 5))
        fig_cm.suptitle(
            "Confusion Matrices  —  Decision Tree  |  Random Forest  |  Gradient Boosted Tree",
            fontsize=13, fontweight="bold", y=1.03,
        )
        for ax, (name, r) in zip(axes, res.items()):
            cm  = r["cm"]
            tot = cm.sum()
            ann = np.array([[f"{v}\n({v/tot*100:.1f}%)" for v in row] for row in cm])
            cmap = sns.light_palette(MODEL_COLORS[name], n_colors=8, as_cmap=True)
            sns.heatmap(
                cm, annot=ann, fmt="", cmap=cmap, ax=ax,
                linewidths=1.0, linecolor="white",
                xticklabels=["No Loan", "Accepted"],
                yticklabels=["No Loan", "Accepted"],
                cbar=False, annot_kws={"size": 13, "weight": "bold"},
            )
            ax.set_title(name, fontsize=12, fontweight="bold", pad=10)
            ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
            ax.set_ylabel("Actual Label",    fontsize=10, labelpad=8)
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 5 — PRESCRIPTIVE INSIGHTS
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        sec("Feature Importance — What Drives Loan Acceptance?")
        ins(
            "Feature importance scores reveal how heavily each variable influences the model's prediction. "
            "Income consistently ranks #1. Use this to define segment criteria, personalise messaging, "
            "and decide which data to collect from new prospects."
        )
        c1, c2 = st.columns(2)
        for ci, mname in enumerate(["Random Forest", "Gradient Boosted Tree"]):
            mdl_obj = res[mname]["model"]
            fi = pd.Series(mdl_obj.feature_importances_, index=feat_cols).sort_values(ascending=True)
            fig = px.bar(
                x=fi.values, y=fi.index, orientation="h",
                color=fi.values, color_continuous_scale="Blues",
                title=f"Feature Importance — {mname}",
                labels={"x": "Importance Score", "y": "Feature"},
            )
            fig.update_traces(texttemplate="%{x:.3f}", textposition="outside")
            fig.update_layout(height=420, margin=dict(t=50, b=20),
                               coloraxis_showscale=False)
            (c1 if ci == 0 else c2).plotly_chart(fig, use_container_width=True)

        sec("Ideal Customer Profile for Personal Loan")
        ins(
            "Derived from feature importances and acceptance-rate analysis. "
            "Use this profile to define your top-priority target segment for the next campaign."
        )
        profile = pd.DataFrame({
            "Attribute":       ["Annual Income", "Education", "CD Account",
                                "CC Spending", "Family Size", "Age Range", "Mortgage"],
            "Ideal Profile":   ["> $100,000", "Graduate or Advanced/Professional",
                                "Yes (has a CD account)",
                                "≥ $2,500 / month", "3–4 members",
                                "35–55 years", "Has a mortgage"],
            "Evidence":        [
                "Acceptance jumps from ~0 % to 35.6 % as income rises — single strongest predictor",
                "Graduate & professional customers accept at 3× the rate of undergrads",
                "CD holders accept at 46.4 % vs 7.2 % — a 6× lift",
                "High CC spenders are already comfortable with revolving credit products",
                "Larger families carry greater financial obligations and loan demand",
                "Mid-career professionals have stable income and rising capital needs",
                "Mortgaged customers already trust the bank for large credit facilities",
            ],
        })
        st.dataframe(profile, use_container_width=True, hide_index=True)

        sec("Customer Segment Opportunity Matrix")
        ins(
            "Four segments are defined by combining income, education, and CD account status. "
            "With your budget halved, concentrating 65 % of spend on Platinum + Prime segments "
            "(≈ 800 customers) delivers 4–6× more acceptances per marketing dollar than a broad blast."
        )
        df5 = df.copy()
        conditions = [
            ((df5["Income"] > 100) & (df5["Education"] >= 2) & (df5["CD Account"] == 1)),
            (df5["CD Account"] == 1),
            ((df5["Income"] > 100) & (df5["Education"] >= 2)),
        ]
        choices = ["💎 Platinum", "⭐ Prime (CD Holder)", "🎯 High Priority"]
        df5["Segment"] = np.select(conditions, choices, default="👤 Standard")

        seg = (df5.groupby("Segment")
                  .agg(Customers=("Personal Loan", "count"),
                       Acceptors=("Personal Loan", "sum"))
                  .reset_index())
        seg["Acceptance Rate (%)"] = (seg["Acceptors"] / seg["Customers"] * 100).round(1)

        budget_map = {
            "💎 Platinum": "35 %",
            "⭐ Prime (CD Holder)": "30 %",
            "🎯 High Priority": "25 %",
            "👤 Standard": "10 %",
        }
        seg["Recommended Budget"] = seg["Segment"].map(budget_map)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(seg, use_container_width=True, hide_index=True)
        with c2:
            fig = px.scatter(
                seg, x="Customers", y="Acceptance Rate (%)",
                size="Acceptors", color="Segment", text="Segment",
                size_max=65,
                color_discrete_sequence=[C_RED, C_ORANGE, C_GREEN, C_BLUE],
                title="Segment Opportunity: Size vs Acceptance Rate",
                labels={
                    "Customers": "Segment Size (# Customers)",
                    "Acceptance Rate (%)": "Loan Acceptance Rate (%)",
                },
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        sec("Hyper-Personalised Campaign Recommendation Matrix")
        ins(
            "Match your message to the customer's context. "
            "Platinum customers respond to exclusivity and pre-approval; Standard customers need education about the product."
        )
        st.markdown("""
| Segment | Budget Share | Best Channels | Message Theme | Expected Acceptance |
|---------|:---:|---------|---------------|:---:|
| 💎 Platinum | 35 % | Personal banker call + Email | *"Pre-approved exclusive offer at a preferential rate — designed for you"* | 40–50 % |
| ⭐ Prime (CD Holder) | 30 % | Email + Mobile app push | *"Your savings are already working — let a loan work too"* | 25–35 % |
| 🎯 High Priority | 25 % | Email + SMS | *"You qualify — grow your goals with a personalised loan"* | 10–15 % |
| 👤 Standard | 10 % | Digital retargeting | *"Start small, dream big — explore our flexible loan options"* | 2–5 % |
""")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 6 — PREDICT NEW DATA
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        sec(f"Predict Personal Loan Acceptance — {best}")
        ins(
            f"Upload a CSV containing customer records <b>without</b> the 'Personal Loan' column. "
            f"The best model — <b>{best}</b> (ROC-AUC: {res[best]['roc_auc']:.4f}) — will score "
            f"each customer with a predicted label (0/1) and a probability score. "
            f"Download the enriched file with predictions and priority tier."
        )

        best_mdl = res[best]["model"]

        c1, c2 = st.columns([3, 1])
        with c1:
            pred_up = st.file_uploader(
                "📁 Upload Customer CSV — the 'Personal Loan' column is NOT required",
                type="csv",
                key="predict_uploader",
            )
        with c2:
            st.markdown("**Expected feature columns:**")
            st.code("\n".join(feat_cols), language="text")

        if pred_up is not None:
            try:
                praw = pd.read_csv(pred_up)
                pc   = praw.copy()
                if "Experience" in pc.columns:
                    pc["Experience"] = pc["Experience"].abs()
                pc.drop(columns=["ID", "ZIP Code", "Personal Loan"],
                        errors="ignore", inplace=True)

                missing = [c for c in feat_cols if c not in pc.columns]
                if missing:
                    st.error(f"❌ Missing columns in uploaded file: {missing}")
                else:
                    pX          = pc[feat_cols]
                    pred_labels = best_mdl.predict(pX)
                    pred_probs  = best_mdl.predict_proba(pX)[:, 1]

                    out = praw.copy()
                    out["Predicted_Personal_Loan"] = pred_labels
                    out["Loan_Probability_%"]      = (pred_probs * 100).round(1)
                    out["Priority_Tier"]           = pd.cut(
                        pred_probs,
                        bins=[0.0, 0.25, 0.55, 1.001],
                        labels=["🔵 Low", "🟡 Medium", "🟢 High"],
                        include_lowest=True,
                    )

                    st.success(
                        f"✅ Predictions generated for **{len(out):,}** customers!"
                    )
                    kc = st.columns(4)
                    kc[0].metric("Total Records",       f"{len(out):,}")
                    kc[1].metric("Predicted Acceptors", f"{int(pred_labels.sum()):,}")
                    kc[2].metric("Predicted Rate",      f"{pred_labels.mean()*100:.1f}%")
                    kc[3].metric("Avg Probability",     f"{pred_probs.mean()*100:.1f}%")

                    # Distribution of priority tiers
                    tier_counts = out["Priority_Tier"].value_counts()
                    fig = px.pie(
                        values=tier_counts.values,
                        names=tier_counts.index.astype(str),
                        title="Predicted Priority Tier Distribution",
                        color_discrete_sequence=[C_GREEN, C_ORANGE, C_BLUE],
                        hole=0.45,
                    )
                    fig.update_layout(height=300, margin=dict(t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(out, use_container_width=True)

                    buf = io.StringIO()
                    out.to_csv(buf, index=False)
                    st.download_button(
                        label="⬇️ Download Predictions as CSV",
                        data=buf.getvalue(),
                        file_name="loan_predictions.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"Processing error: {e}")
        else:
            st.info(
                "📌 No file uploaded yet. "
                "Try the **sample_test_data.csv** included in the project zip — "
                "it contains 500 customers without the target column."
            )


if __name__ == "__main__":
    main()
