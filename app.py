"""
ChurnLens - Telecom Customer Churn Classifier Dashboard
ML Assignment 2 | M.Tech AIML, BITS Pilani WILP
Author : Kunal Vaishnav (2026aj05283)

The app loads the five pre-trained models from ./model, evaluates them on an
uploaded test CSV (features + 'churn' target) and shows metrics, confusion
matrix, classification report and ROC curves. Models are NEVER retrained here.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_TEST = ROOT / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
PALETTE = ["#0F766E", "#D97706", "#7C3AED", "#DC2626", "#2563EB"]

st.set_page_config(page_title="ChurnLens | Kunal Vaishnav", page_icon="📡", layout="wide")

st.markdown(
    """
    <style>
      .hero {background: linear-gradient(120deg,#0F766E 0%,#115E59 55%,#134E4A 100%);
             padding: 1.4rem 1.8rem; border-radius: 14px; color: #F0FDFA; margin-bottom: 1rem;}
      .hero h1 {margin: 0; font-size: 2rem; color: #F0FDFA;}
      .hero p  {margin: .3rem 0 0 0; opacity: .9;}
      div[data-testid="stMetric"] {background:#F0FDFA; border:1px solid #99F6E4;
             border-radius:12px; padding:.6rem .9rem;}
      .footer {text-align:center; color:#64748B; font-size:.85rem; margin-top:2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- loaders
@st.cache_resource(show_spinner="Loading trained models...")
def load_models() -> dict:
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}


@st.cache_data
def load_meta() -> tuple[dict, dict]:
    meta = json.loads((MODEL_DIR / "feature_meta.json").read_text())
    train_metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    return meta, train_metrics


def compute_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def validate(df: pd.DataFrame, features: list[str], target: str) -> list[str]:
    missing = [c for c in features + [target] if c not in df.columns]
    errors = []
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}")
    else:
        non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            errors.append(f"Non-numeric feature column(s): {', '.join(non_numeric)}")
        if df[features + [target]].isna().any().any():
            errors.append("The file contains missing values - please clean it first.")
        if not set(df[target].unique()).issubset({0, 1}):
            errors.append(f"Target '{target}' must contain only 0 / 1 values.")
    return errors


# ----------------------------------------------------------------- header
models = load_models()
meta, train_metrics = load_meta()
FEATURES, TARGET = meta["features"], meta["target"]
CLASS_NAMES = [meta["class_names"]["0"], meta["class_names"]["1"]]

st.markdown(
    """
    <div class="hero">
      <h1>📡 ChurnLens — Telecom Churn Classifier</h1>
      <p>Compare five pre-trained ML models on your own test data &nbsp;·&nbsp;
         ML Assignment 2 &nbsp;·&nbsp; Kunal Vaishnav (2026aj05283)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("1 · Upload test CSV", type=["csv"],
                                help=f"Must contain the {len(FEATURES)} feature columns and '{TARGET}'.")
    use_sample = st.toggle("Use bundled test_data.csv when no file is uploaded", value=True,
                           help="Handy for a quick demo if you don't have a file.")
    model_name = st.selectbox("2 · Choose a model", list(MODEL_FILES.keys()), index=4)
    threshold = st.slider("3 · Decision threshold (P(churn))", 0.05, 0.95, 0.50, 0.05,
                          help="Rows with predicted churn probability ≥ threshold are labelled 'Churn'.")
    st.divider()
    if SAMPLE_TEST.exists():
        st.download_button("⬇️ Download sample test_data.csv", SAMPLE_TEST.read_bytes(),
                           file_name="test_data.csv", mime="text/csv", width="stretch")
    with st.expander("Required columns"):
        st.code("\n".join(FEATURES + [TARGET]), language="text")

# ----------------------------------------------------------------- data
if uploaded is not None:
    df = pd.read_csv(uploaded)
    source = f"Uploaded file: **{uploaded.name}**"
elif use_sample and SAMPLE_TEST.exists():
    df = pd.read_csv(SAMPLE_TEST)
    source = "Bundled **test_data.csv** (20% stratified hold-out, never seen in training)"
else:
    st.info("👈 Upload a test CSV from the sidebar (or switch on the bundled sample) to begin.")
    st.stop()

df.columns = [c.strip() for c in df.columns]
errs = validate(df, FEATURES, TARGET)
if errs:
    for e in errs:
        st.error(e)
    st.stop()

X, y = df[FEATURES], df[TARGET].astype(int)
st.caption(f"{source} — {len(df):,} rows · churn rate {y.mean():.1%}")


def to_labels(mdl, X_: pd.DataFrame, p: np.ndarray, thr: float) -> np.ndarray:
    """At the default 0.50 threshold use the model's own predict() (identical to
    training-time evaluation; matters for SVM whose Platt-scaled probabilities can
    disagree slightly with its decision function). Otherwise threshold P(churn)."""
    if abs(thr - 0.5) < 1e-9:
        return mdl.predict(X_).astype(int)
    return (p >= thr).astype(int)


@st.cache_data(show_spinner="Scoring all models...")
def score_all(X_: pd.DataFrame, y_: pd.Series, thr: float) -> tuple[pd.DataFrame, dict]:
    rows, probs = [], {}
    for name, mdl in load_models().items():
        p = mdl.predict_proba(X_)[:, 1]
        probs[name] = p
        rows.append({"Model": name, **compute_metrics(y_, to_labels(mdl, X_, p, thr), p)})
    return pd.DataFrame(rows).set_index("Model")[METRIC_ORDER], probs


summary, all_probs = score_all(X, y, threshold)
prob = all_probs[model_name]
pred = to_labels(models[model_name], X, prob, threshold)

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Selected model", "📊 Compare all models", "🔍 Predictions", "ℹ️ About"])

# ----------------------------------------------------------------- tab 1
with tab1:
    st.subheader(f"{model_name} — evaluation on test data")
    cols = st.columns(6)
    for c, m in zip(cols, METRIC_ORDER):
        c.metric(m, f"{summary.loc[model_name, m]:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="crest", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig, width="stretch"); plt.close(fig)
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"TP={tp} · FP={fp} · FN={fn} · TN={tn} — {fn} churners missed, {fp} false alarms.")
    with right:
        st.markdown("**ROC curve**")
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, color="#0F766E", lw=2.2, label=f"AUC = {summary.loc[model_name, 'AUC']:.3f}")
        ax.plot([0, 1], [0, 1], ls="--", color="#94A3B8", lw=1)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right"); ax.grid(alpha=.25)
        st.pyplot(fig, width="stretch"); plt.close(fig)

    st.markdown("**Classification report**")
    rep = classification_report(y, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.style.format("{:.3f}"), width="stretch")

# ----------------------------------------------------------------- tab 2
with tab2:
    st.subheader("All five models on the same test data")
    best = summary["MCC"].idxmax()
    st.success(f"🏆 Best model by MCC at threshold {threshold:.2f}: **{best}** "
               f"(MCC {summary.loc[best, 'MCC']:.4f}, F1 {summary.loc[best, 'F1']:.4f})")
    st.dataframe(summary.style.format("{:.4f}").highlight_max(axis=0, color="#99F6E4"),
                 width="stretch")

    metric_pick = st.radio("Metric to plot", METRIC_ORDER, horizontal=True, index=5)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    vals = summary[metric_pick].sort_values()
    ax.barh(vals.index, vals.values, color=[PALETTE[list(MODEL_FILES).index(n)] for n in vals.index])
    for i, v in enumerate(vals.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.08); ax.set_xlabel(metric_pick); ax.grid(axis="x", alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, width="stretch"); plt.close(fig)

    st.markdown("**ROC curves — all models**")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for (name, p), colr in zip(all_probs.items(), PALETTE):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2, color=colr, label=f"{name} ({summary.loc[name, 'AUC']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#94A3B8", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.25)
    st.pyplot(fig, width="stretch"); plt.close(fig)

    st.download_button("⬇️ Download comparison (CSV)", summary.round(4).to_csv().encode(),
                       file_name="model_comparison.csv", mime="text/csv")

# ----------------------------------------------------------------- tab 3
with tab3:
    st.subheader(f"Row-level predictions — {model_name}")
    out = df.copy()
    out["churn_probability"] = prob.round(4)
    out["predicted_churn"] = pred
    out["correct"] = np.where(out["predicted_churn"] == y, "✅", "❌")
    only_wrong = st.checkbox("Show only misclassified rows")
    view = out[out["correct"] == "❌"] if only_wrong else out
    st.dataframe(view, width="stretch", height=420)
    st.download_button("⬇️ Download predictions", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_FILES[model_name].split('.')[0]}.csv",
                       mime="text/csv")

# ----------------------------------------------------------------- tab 4
with tab4:
    st.subheader("About this app")
    st.markdown(
        f"""
- **Dataset:** Telecom Customer Churn (Kaggle / BigML, via PMLB) — 5,000 customers,
  {len(FEATURES)} input features, binary target `{TARGET}` (1 = customer left).
- **Training:** 80/20 stratified split (seed {meta['random_state']}), 5-fold GridSearchCV
  optimising F1. Every model is a saved scikit-learn `Pipeline` (scaling + classifier).
- **This app never retrains** — it only loads `model/*.joblib` and scores the uploaded rows.
- **Threshold slider:** metrics above use the chosen probability threshold
  (default 0.50 reproduces the README numbers exactly when the bundled test set is used).
        """
    )
    st.markdown("**Tuned hyper-parameters (from training)**")
    st.dataframe(pd.DataFrame({v["name"]: {"best params": str(v["best_params"]), "CV F1": v["cv_f1"]}
                               for v in train_metrics.values()}).T, width="stretch")

st.markdown('<div class="footer">Built by Kunal Vaishnav · BITS ID 2026aj05283 · '
            'M.Tech AIML — Machine Learning Assignment 2</div>', unsafe_allow_html=True)
