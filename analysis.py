"""Quality, model, and operational analysis helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

LABELS = ["allow", "restrict", "remove"]


def annotate_quality(cases: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    joined = annotations.merge(cases[["case_id", "gold_action", "policy_area", "difficulty", "market", "queue", "week"]], on="case_id")
    joined["is_correct"] = joined["label"] == joined["gold_action"]
    joined["is_over_enforcement"] = (joined["label"] != "allow") & (joined["gold_action"] == "allow")
    joined["is_under_enforcement"] = (joined["label"] == "allow") & (joined["gold_action"] != "allow")
    return joined


def team_summary(scored: pd.DataFrame) -> pd.DataFrame:
    return (scored.groupby("team", as_index=False)
            .agg(accuracy=("is_correct", "mean"), cases=("case_id", "nunique"),
                 avg_review_seconds=("review_seconds", "mean"),
                 over_enforcement_rate=("is_over_enforcement", "mean"),
                 under_enforcement_rate=("is_under_enforcement", "mean"),
                 avg_confidence=("confidence", "mean"))
            .sort_values("accuracy", ascending=False))


def segment_summary(scored: pd.DataFrame, by: str = "policy_area") -> pd.DataFrame:
    return (scored.groupby(["team", by], as_index=False)
            .agg(accuracy=("is_correct", "mean"), cases=("case_id", "nunique"),
                 under_enforcement_rate=("is_under_enforcement", "mean"),
                 over_enforcement_rate=("is_over_enforcement", "mean")))


def pairwise_agreement(annotations: pd.DataFrame) -> pd.DataFrame:
    wide = annotations.pivot(index="case_id", columns="team", values="label")
    teams = list(wide.columns)
    rows = []
    for i, left in enumerate(teams):
        for right in teams[i + 1:]:
            rows.append({"team_a": left, "team_b": right, "agreement": float((wide[left] == wide[right]).mean())})
    return pd.DataFrame(rows)


def train_and_evaluate(cases: pd.DataFrame, seed: int = 42) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    """Train an interpretable text baseline; hold out cases before fitting."""
    train, test = train_test_split(cases, test_size=0.25, random_state=seed, stratify=cases["gold_action"])
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
    ])
    model.fit(train["text"], train["gold_action"])
    probs = model.predict_proba(test["text"])
    classes = list(model.classes_)
    pred = model.predict(test["text"])
    output = test[["case_id", "text", "policy_area", "gold_action", "difficulty", "market", "queue", "week"]].copy()
    output["model_action"] = pred
    output["confidence"] = probs.max(axis=1)
    output["expected_cost"] = [cost_for(y, p) for y, p in zip(test["gold_action"], pred)]
    for ix, label in enumerate(classes):
        output[f"p_{label}"] = probs[:, ix]
    metrics = {
        "accuracy": float(accuracy_score(test["gold_action"], pred)),
        "macro_f1": float(f1_score(test["gold_action"], pred, average="macro")),
        "classification_report": classification_report(test["gold_action"], pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(test["gold_action"], pred, labels=LABELS).tolist(),
        "labels": LABELS,
        "holdout_cases": int(len(test)),
        "training_cases": int(len(train)),
    }
    return model, metrics, output


def cost_for(gold: str, predicted: str) -> float:
    """Illustrative asymmetric error cost; policy owners must set real costs."""
    if gold == predicted:
        return 0.0
    if gold == "allow" and predicted in ("restrict", "remove"):
        return 1.5
    if gold in ("restrict", "remove") and predicted == "allow":
        return 3.0
    return 0.6  # adjacent intervention levels


def threshold_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    """Abstain to human review below confidence threshold; report coverage and cost."""
    rows = []
    for threshold in np.arange(0.35, 0.96, 0.05):
        auto = predictions[predictions["confidence"] >= threshold]
        rows.append({
            "threshold": round(float(threshold), 2),
            "automation_coverage": len(auto) / len(predictions),
            "auto_accuracy": float((auto["gold_action"] == auto["model_action"]).mean()) if len(auto) else np.nan,
            "mean_cost_all_cases": float(auto["expected_cost"].sum() / len(predictions)),
            "cases_routed_to_human": int(len(predictions) - len(auto)),
        })
    return pd.DataFrame(rows)


def recommendations(scored: pd.DataFrame, team_metrics: pd.DataFrame, predictions: pd.DataFrame) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    weak = team_metrics.sort_values("accuracy").iloc[0]
    actions.append({
        "priority": "P1" if weak.accuracy < 0.85 else "P2",
        "signal": f"{weak.team} accuracy is {weak.accuracy:.1%} across {int(weak.cases)} cases",
        "action": "Run a focused calibration on the team's two lowest-performing policy areas; use blind gold examples and record rationale before a short re-check.",
        "owner": "Quality lead",
    })
    hard = scored[scored["difficulty"] != "standard"].groupby("policy_area")["is_correct"].mean().sort_values()
    if len(hard):
        area = hard.index[0]
        actions.append({
            "priority": "P1",
            "signal": f"Lowest edge-case agreement is in {area} ({hard.iloc[0]:.1%})",
            "action": "Publish a decision tree with paired boundary examples, then sample the next 100 cases for adjudication and update the policy FAQ.",
            "owner": "Policy + Quality",
        })
    if not predictions.empty:
        uncertain = predictions[predictions["confidence"] < 0.60]
        actions.append({
            "priority": "P2",
            "signal": f"{len(uncertain)} of {len(predictions)} holdout cases fall below 0.60 model confidence",
            "action": "Keep low-confidence cases in human review; send representative false positives and false negatives to the model team with context and policy rationale.",
            "owner": "ML + Operations",
        })
    return actions

