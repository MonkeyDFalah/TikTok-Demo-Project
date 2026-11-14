"""Streamlit dashboard for moderation quality operations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import plotly.express as px
import streamlit as st

from liveguard.analysis import annotate_quality, pairwise_agreement, recommendations, segment_summary, team_summary, threshold_analysis, train_and_evaluate
from liveguard.data import build_dataset

st.set_page_config(page_title="LiveGuard | Quality Ops", page_icon="🛡️", layout="wide")
st.title("🛡️ LiveGuard Quality Ops")
st.caption("Synthetic portfolio demo · Human QA · Model evaluation · Operational actions")
st.info("All records and metrics are synthetic. This demo is not affiliated with TikTok and does not represent production policy or performance.")

with st.sidebar:
    st.header("Scenario controls")
    seed = st.number_input("Random seed", min_value=1, max_value=99999, value=42)
    n_cases = st.slider("Synthetic cases", min_value=240, max_value=1800, value=720, step=120)
    st.caption("Changing these controls regenerates the seeded demo queue and retrains the baseline.")

@st.cache_data(show_spinner="Generating synthetic queue and training baseline…")
def load_data(n: int, random_seed: int):
    cases, annotations = build_dataset(n, random_seed)
    scored = annotate_quality(cases, annotations)
    model, metrics, predictions = train_and_evaluate(cases, random_seed)
    teams = team_summary(scored)
    return cases, annotations, scored, model, metrics, predictions, teams

cases, annotations, scored, model, metrics, predictions, teams = load_data(int(n_cases), int(seed))
thresholds = threshold_analysis(predictions)
actions = recommendations(scored, teams, predictions)

tab_overview, tab_human, tab_model, tab_actions, tab_method = st.tabs(["Overview", "Human quality", "Model lab", "Action plan", "Method & limits"])

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cases", f"{len(cases):,}")
    c2.metric("Human annotations", f"{len(annotations):,}")
    c3.metric("Mean BPO accuracy", f"{teams[teams.team.str.startswith('BPO')].accuracy.mean():.1%}")
    c4.metric("Baseline macro F1", f"{metrics['macro_f1']:.3f}")
    left, right = st.columns(2)
    with left:
        st.subheader("Human accuracy by team")
        fig = px.bar(teams, x="team", y="accuracy", color="accuracy", range_y=[0, 1], text_auto=".1%", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Queue composition")
        mix = cases.groupby(["policy_area", "gold_action"], as_index=False).size()
        fig = px.sunburst(mix, path=["gold_action", "policy_area"], values="size", color="gold_action")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Recommended operational actions")
    for item in actions:
        st.markdown(f"**{item['priority']} · {item['signal']}**  \n{item['action']}  \n*Owner: {item['owner']}*")

with tab_human:
    filter1, filter2, filter3, filter4 = st.columns(4)
    policy_filter = filter1.multiselect("Policy area", sorted(scored.policy_area.unique()), default=sorted(scored.policy_area.unique()))
    difficulty_filter = filter2.multiselect("Case difficulty", sorted(scored.difficulty.unique()), default=sorted(scored.difficulty.unique()))
    market_filter = filter3.multiselect("Market", sorted(scored.market.unique()), default=sorted(scored.market.unique()))
    queue_filter = filter4.multiselect("Queue", sorted(scored.queue.unique()), default=sorted(scored.queue.unique()))
    view = scored[scored.policy_area.isin(policy_filter) & scored.difficulty.isin(difficulty_filter) & scored.market.isin(market_filter) & scored.queue.isin(queue_filter)]
    st.subheader("Team × policy area scorecard")
    segment = segment_summary(view)
    pivot = segment.pivot(index="team", columns="policy_area", values="accuracy")
    st.dataframe(pivot.style.format("{:.1%}").background_gradient(cmap="RdYlGn", vmin=0, vmax=1), use_container_width=True)
    st.subheader("Annotator agreement")
    agreement = pairwise_agreement(annotations)
    st.dataframe(agreement.sort_values("agreement"), use_container_width=True, hide_index=True)
    st.subheader("Error rates by team")
    error_rates = view.groupby("team", as_index=False).agg(over_enforcement=("is_over_enforcement", "mean"), under_enforcement=("is_under_enforcement", "mean"))
    st.plotly_chart(px.bar(error_rates, x="team", y=["over_enforcement", "under_enforcement"], barmode="group", labels={"value": "Rate", "variable": "Error type"}), use_container_width=True)
    st.caption("Over-enforcement = an allowed case labeled restrict/remove. Under-enforcement = a policy-violating case labeled allow. Ground truth is synthetic adjudication.")
    st.subheader("Weekly quality trend")
    weekly = view.groupby(["week", "team"], as_index=False).agg(accuracy=("is_correct", "mean"), cases=("case_id", "nunique"))
    st.plotly_chart(px.line(weekly, x="week", y="accuracy", color="team", markers=True, hover_data=["cases"], range_y=[0, 1], labels={"accuracy": "Accuracy"}), use_container_width=True)
    st.caption("Small weekly samples can swing sharply. Use the case count in each hover tooltip before escalating a trend.")

with tab_model:
    st.subheader("Holdout evaluation")
    m1, m2, m3 = st.columns(3)
    m1.metric("Holdout accuracy", f"{metrics['accuracy']:.1%}")
    m2.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
    m3.metric("Holdout cases", metrics["holdout_cases"])
    labels = metrics["labels"]
    import pandas as pd
    cm = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)
    st.plotly_chart(px.imshow(cm, text_auto=True, labels={"x": "Predicted", "y": "Gold", "color": "Cases"}, color_continuous_scale="Blues"), use_container_width=True)
    st.subheader("Automation threshold trade-off")
    st.plotly_chart(px.line(thresholds, x="threshold", y=["automation_coverage", "auto_accuracy"], markers=True, labels={"value": "Rate", "variable": "Metric"}), use_container_width=True)
    st.dataframe(thresholds.style.format({"threshold": "{:.2f}", "automation_coverage": "{:.1%}", "auto_accuracy": "{:.1%}", "mean_cost_all_cases": "{:.3f}"}), use_container_width=True, hide_index=True)
    st.caption("The asymmetric cost values are illustrative assumptions. Threshold selection needs policy-owner approval, risk appetite, and human review requirements.")
    with st.expander("Inspect model misses"):
        misses = predictions[predictions.gold_action != predictions.model_action].copy()
        st.dataframe(misses[["case_id", "text", "policy_area", "difficulty", "gold_action", "model_action", "confidence"]].sort_values("confidence"), use_container_width=True, hide_index=True)

with tab_actions:
    st.subheader("Evidence-linked action register")
    for item in actions:
        with st.container(border=True):
            st.markdown(f"### {item['priority']} · {item['owner']}")
            st.write("**Signal:**", item["signal"])
            st.write("**Action:**", item["action"])
    st.subheader("Download reproducible review tables")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Cases CSV", cases.to_csv(index=False), "synthetic_cases.csv", "text/csv")
    d2.download_button("Annotations CSV", annotations.to_csv(index=False), "synthetic_annotations.csv", "text/csv")
    d3.download_button("Model predictions CSV", predictions.to_csv(index=False), "holdout_predictions.csv", "text/csv")

with tab_method:
    st.subheader("What this project demonstrates")
    st.markdown("""
- Benchmarking and calibration across simulated partner teams.
- Adjudication-oriented edge-case and policy-area analysis.
- Error taxonomy, over/under-enforcement tracking, and action ownership.
- A reproducible TF-IDF + logistic regression baseline with a true holdout split.
- Confidence-based human-review routing and an explicit error-cost sensitivity view.
- Synthetic data generation so the demo can be published without real user content.
""")
    st.subheader("Limitations and responsible use")
    st.markdown("""
- Every case, market, label, annotator result, and metric is synthetic; this is not production evidence.
- The categories are a portfolio taxonomy, not TikTok policy. A real system must use current, approved policy and region-specific requirements.
- The classifier is a lightweight text baseline, not an LLM and not suitable for content decisions. It lacks video, audio, conversational context, appeals, and policy updates.
- The simple cost matrix is illustrative. Real thresholds need policy, legal, trust and safety, and operations review.
- Do not use this demo to make decisions about real people or content.
""")
    st.write("Reproducibility: seed", int(seed), "· train/test split 75/25, stratified by action")

