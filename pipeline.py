"""Generate synthetic data and reproducible quality reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .analysis import annotate_quality, recommendations, team_summary, threshold_analysis, train_and_evaluate
from .data import build_dataset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def run(n_cases: int = 720, seed: int = 42) -> dict:
    DATA.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    cases, annotations = build_dataset(n_cases=n_cases, seed=seed)
    scored = annotate_quality(cases, annotations)
    model, model_metrics, predictions = train_and_evaluate(cases, seed)
    teams = team_summary(scored)
    thresholds = threshold_analysis(predictions)
    actions = recommendations(scored, teams, predictions)
    cases.to_csv(DATA / "cases.csv", index=False)
    annotations.to_csv(DATA / "annotations.csv", index=False)
    scored.to_csv(REPORTS / "annotation_quality.csv", index=False)
    teams.to_csv(REPORTS / "team_scorecard.csv", index=False)
    predictions.to_csv(REPORTS / "model_holdout_predictions.csv", index=False)
    thresholds.to_csv(REPORTS / "threshold_analysis.csv", index=False)
    (REPORTS / "model_metrics.json").write_text(json.dumps(model_metrics, indent=2), encoding="utf-8")
    (REPORTS / "action_plan.json").write_text(json.dumps(actions, indent=2), encoding="utf-8")
    markdown = [
        "# LiveGuard Quality Ops — generated report", "",
        "> Synthetic data only. Results describe this seeded demo dataset and are not TikTok or production performance.", "",
        f"- Cases: {len(cases):,}; annotations: {len(annotations):,}",
        f"- Baseline model holdout accuracy: {model_metrics['accuracy']:.1%}; macro F1: {model_metrics['macro_f1']:.3f}",
        "", "## Team scorecard", "", teams.to_markdown(index=False), "", "## Recommended actions", "",
    ]
    markdown += [f"- **{a['priority']} · {a['signal']}** — {a['action']} (Owner: {a['owner']})" for a in actions]
    (REPORTS / "quality_brief.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return {"cases": cases, "annotations": annotations, "scored": scored, "teams": teams,
            "model": model, "model_metrics": model_metrics, "predictions": predictions,
            "thresholds": thresholds, "actions": actions}

