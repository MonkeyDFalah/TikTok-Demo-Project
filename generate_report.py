"""Generate CSV and Markdown artifacts from the seeded synthetic queue."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from liveguard.pipeline import run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cases", type=int, default=720)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
result = run(args.cases, args.seed)
print(f"Generated {len(result['cases'])} synthetic cases and reports in {ROOT / 'reports'}")
print(f"Baseline holdout accuracy: {result['model_metrics']['accuracy']:.1%}")
print(f"Baseline macro F1: {result['model_metrics']['macro_f1']:.3f}")

