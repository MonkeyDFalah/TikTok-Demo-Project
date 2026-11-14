# LiveGuard Quality Ops

**A synthetic AI data operations portfolio project for live content quality.** It demonstrates how a quality specialist can connect human annotation, BPO calibration, model evaluation, and clear operational actions in one reproducible workflow.

> This is an independent portfolio demo. All content and metrics are generated from synthetic templates. It is not affiliated with TikTok, does not use TikTok data, and does not represent TikTok policy or production performance.

## What it does

- Generates a deterministic case queue across six illustrative policy areas, three review actions, five fictional markets, and three fictional live queues.
- Simulates four annotator teams, confidence, review time, adjudicated labels, difficult examples, over-enforcement, and under-enforcement.
- Benchmarks team accuracy and error patterns by policy area and difficulty; calculates pairwise agreement and produces calibration actions.
- Trains a TF-IDF + logistic regression text baseline on a stratified 75/25 split and reports holdout accuracy, macro F1, a confusion matrix, and individual misses.
- Evaluates a confidence threshold as a human-review routing decision, showing automation coverage, accuracy, and an explicitly illustrative asymmetric error cost.
- Provides a Streamlit dashboard with filters, charts, evidence-linked actions, and downloadable synthetic review tables.
- Exports reproducible scorecards and a concise quality brief.

## Screens in the dashboard

1. **Overview:** key metrics, team scorecard, queue mix, and the action register.
2. **Human quality:** team-by-policy heatmap, annotator agreement, and over/under-enforcement rates.
3. **Model lab:** holdout evaluation, confusion matrix, threshold trade-offs, and inspectable model misses.
4. **Action plan:** owner-tagged, evidence-linked interventions and CSV downloads.
5. **Method & limits:** reproducibility notes, intended use, and limitations.

## Run locally

Requires Python 3.10 or newer.

```bash
git clone https://github.com/YOUR-USERNAME/liveguard-quality-ops.git
cd liveguard-quality-ops
python -m venv .venv
```

Activate the environment, then install and launch:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux: use `source .venv/bin/activate`
pip install -r requirements.txt
streamlit run src/liveguard/dashboard.py
```

The dashboard opens at `http://localhost:8501`. It builds a seeded dataset and baseline when loaded; no API key or external data is needed.

## Generate downloadable report artifacts

```bash
python scripts/generate_report.py --cases 720 --seed 42
```

This writes the synthetic case and annotation CSVs to `data/` and scorecards, holdout predictions, threshold analysis, JSON metrics/actions, and a Markdown quality brief to `reports/`. Change the seed to produce a new deterministic scenario. Generated artifacts are git-ignored by default; remove relevant ignore entries if you want to commit a particular run.

## Deploy from GitHub with Streamlit Community Cloud

GitHub stores the repository; it does not run a Python dashboard by itself. Streamlit Community Cloud can deploy the app directly from the public GitHub repository.

1. Create a new **public** GitHub repository named `liveguard-quality-ops` (or another name you prefer).
2. From this project folder, commit and push the files:

   ```bash
   git init
   git add .
   git commit -m "Build LiveGuard quality operations portfolio demo"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/liveguard-quality-ops.git
   git push -u origin main
   ```

3. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) using GitHub and choose **Create app**.
4. Select your repository and `main` branch. Set the app file path to `src/liveguard/dashboard.py`.
5. Deploy. Streamlit installs `requirements.txt`; the app needs no secrets.
6. Add the deployed app URL to the top of this README and to your GitHub profile or CV project section.

For a private repo or a service you control, the included Dockerfile can run the same app:

```bash
docker build -t liveguard-quality-ops .
docker run --rm -p 8501:8501 liveguard-quality-ops
```

## Repository layout

```text
src/liveguard/data.py        synthetic queue and annotation generation
src/liveguard/analysis.py    quality metrics, model evaluation, and actions
src/liveguard/pipeline.py    report generation pipeline
src/liveguard/dashboard.py   Streamlit application
scripts/generate_report.py   CLI for exporting report artifacts
data/                        generated synthetic inputs
reports/                     generated quality and model outputs
```

## Metric definitions

- **Team accuracy:** fraction of simulated team labels matching the synthetic adjudicated action.
- **Pairwise agreement:** fraction of cases where two teams select the same action. This is a descriptive agreement rate, not a chance-corrected statistic.
- **Over-enforcement:** a synthetic `allow` case labeled `restrict` or `remove`.
- **Under-enforcement:** a synthetic `restrict`/`remove` case labeled `allow`.
- **Macro F1:** unweighted mean class F1 on the holdout set, so minority actions matter as much as the majority action.
- **Automation coverage:** share of holdout examples above a confidence threshold. This is a routing simulation, not permission to automate decisions.
- **Illustrative error cost:** false allowance costs 3.0, over-enforcement costs 1.5, and a one-step action mismatch costs 0.6. These example weights are for demonstrating trade-off analysis only.

## Scope and responsible use

This is a portfolio work sample, not a moderation product. The taxonomy is illustrative and must not be treated as current platform policy. The text classifier is a lightweight benchmark, not an LLM; it cannot interpret video/audio, conversation context, intent, appeals, or changing rules. Synthetic results cannot support claims about a real market, vendor, model, or operation. A production initiative would require approved policy, privacy and data governance, representative evaluation sets, calibrated annotators, risk review, and human oversight.

## Interview walkthrough

Use the dashboard to tell a five-minute story: identify a weak team/policy slice, separate edge-case knowledge gaps from broad performance issues, show model confusion and confidence trade-offs, then propose a calibration and targeted re-sample with an owner and measurable follow-up. Be clear that the evidence is synthetic and focus on the operating method.

