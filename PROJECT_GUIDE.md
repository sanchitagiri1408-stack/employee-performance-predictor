# Project Guide: Employee Performance Predictor using Data Analytics

This guide turns the project idea into a small, explainable, evidence-first portfolio project. It is written for a beginner who wants to understand the reasoning, for a technical reviewer who wants to inspect the workflow, and for an interview candidate who needs to explain the work honestly.

> **Non-negotiable scope note**  
> The data in this project is synthetic. It is not data about real employees or a real company, and it is not validated for workplace use. The project must not be used to make employment decisions, rank real people, or infer a person’s value, potential, character, or future behavior.

## 1. How to use this guide

Follow the sections in order:

1. define the question and the target;
2. choose a stack and a folder contract;
3. create or document the synthetic dataset;
4. build the ten phases in sequence;
5. run the project and save outputs with stable names;
6. capture the evidence checklist;
7. publish the repository and rehearse the interview explanation.

This repository includes a runnable implementation. The guide explains the reasoning behind each module, the exact commands to run, and the evidence you should capture. The default entry point is `python main.py`.

## 2. What “employee performance prediction” means

### 2.1 Beginner explanation

Prediction means making an informed estimate from examples. If a dataset contains past synthetic records and a clearly defined outcome, a model can learn patterns that help estimate the outcome for another synthetic record.

For example, a fictional dataset might contain:

- completed training modules;
- number of projects or tasks in a period;
- attendance or availability summaries;
- a synthetic feedback score;
- role family and department; and
- a synthetic performance band created by a documented simulation rule.

The model sees columns and examples, not a person’s intentions or complete context. It can be wrong because the data can be incomplete, noisy, unrepresentative, or based on a target that is itself a rough proxy. A prediction is therefore a decision-support-style estimate in the simulation, not a judgment.

### 2.2 Technical explanation

This project is a tabular supervised-learning exercise. The implementation must choose one target formulation before model training:

| Target formulation | Example target | Typical task | Typical metrics |
| --- | --- | --- | --- |
| Binary classification | `meets_simulated_threshold` | Predict one of two labels | Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC where appropriate |
| Multiclass classification | `performance_band` with three or more bands | Predict a category | Macro-F1, balanced accuracy, per-class recall, confusion matrix |
| Regression | `simulated_score` on a defined numeric scale | Predict a number | MAE, RMSE, R² with context |
| Ordinal classification | Ordered bands such as low/medium/high | Predict an ordered category | Per-class metrics plus an ordinal-aware error discussion |

The target must be defined in plain language and in data-dictionary form. The data split, preprocessing, random seed, model version, and metric choice must be recorded. Any preprocessing that learns from data—such as imputation, scaling, or category encoding—must be fitted on training data only and then applied to validation/test data.

### 2.3 Business value in the simulation

The fictional business case is useful for practicing how an analyst might:

- find inconsistent or missing data before a report is trusted;
- summarize training, workload, and feedback patterns;
- compare a transparent baseline with a more flexible candidate model;
- show where errors occur rather than quoting one average score; and
- explain what additional evidence a stakeholder would need.

The business value is educational and process-oriented. It is not proof that an employment prediction system would be valid, fair, lawful, or useful in a real organization.

### 2.4 Responsible-use and ethical limitations

Keep these limitations visible in the README, presentation, and interview answer:

- **Synthetic data limitation:** the records are invented. They cannot establish real-world performance or fairness.
- **Label limitation:** a simulated target reflects the assumptions of the generator, not an objective measurement of human ability.
- **Measurement limitation:** feedback, attendance, output counts, and manager ratings can be noisy or affected by role, access, health, workload, and team context.
- **Proxy limitation:** excluding a sensitive field does not guarantee that correlated fields are harmless.
- **Fairness limitation:** average metrics can conceal different error rates between groups or roles.
- **Privacy limitation:** a future real dataset would require authorization, minimization, access control, retention rules, and legal/HR review.
- **Causality limitation:** feature importance describes a model association; it does not show that changing a feature will change performance.
- **Automation limitation:** a prediction should never be the sole basis for an employment action.
- **Drift limitation:** patterns can change when roles, systems, incentives, or policies change.
- **Human-context limitation:** a table cannot represent every relevant circumstance. Human review, appeal, and context would be required for any high-impact use.

### 2.5 Responsible-use gate before sharing results

Do not share a result until the answer to each question is recorded:

- Is the dataset clearly labeled synthetic?
- Is the target definition visible and understandable?
- Are sensitive attributes excluded from the modeling example or explicitly discussed as a limitation?
- Are the data-generation assumptions documented?
- Are test-set results kept separate from tuning decisions?
- Are subgroup or segment checks included where the synthetic data supports them?
- Are false positives and false negatives explained in plain language?
- Does the presentation state that this is not for employment decisions?
- Are no real names, employee identifiers, or confidential files present?
- Are metric values and screenshots traceable to an actual run?

If any answer is “no,” label the project as incomplete rather than filling the gap with an assumption.

## 3. Technology stack options

### 3.1 Option A — selected student option: Python analysis workflow

**Suggested tools:** Python, pandas, NumPy, scikit-learn, matplotlib/seaborn, and Jupyter.

**Why select it:**

- low cost and easy to run locally;
- teaches the foundations instead of hiding the workflow behind a dashboard;
- supports tabular data, model comparison, charts, and reproducible outputs;
- is familiar to many entry-level data and analytics interviewers; and
- can later feed a dashboard without rewriting the whole analysis.

**Trade-offs:** the student must manage the environment, document the run command, and avoid allowing a notebook to become an untracked sequence of manual steps.

### 3.2 Option B — portfolio demo: Python plus Streamlit

**Suggested tools:** Option A plus Streamlit for a small local interface.

**Good for:** showing a reviewer how a user could select a synthetic record, view a prediction, inspect an explanation, and read the limitation notice.

**Trade-offs:** a polished interface can distract from data quality and evaluation. Build it only after the underlying analysis runs from a clean command, and keep the synthetic-data notice on the visible page.

### 3.3 Option C — business-analytics route: SQL/Power BI with Python support

**Suggested tools:** a small SQL store or CSV/Excel staging layer, Power BI for reporting, and Python for optional model work.

**Good for:** practicing data modeling, stakeholder dashboards, data dictionaries, and refresh-oriented thinking.

**Trade-offs:** licensing, desktop setup, and dashboard choices add complexity. A dashboard can also make an unvalidated prediction look more authoritative than it is.

### 3.4 Selection decision

Use **Option A** for the first complete version. This repository implements Option A with a script-based workflow so it can run reproducibly from a clean checkout. Add Option B or C only when the data contract, evaluation, and limitations are already reproducible.

## 4. Architecture and data flow

```mermaid
flowchart TD
    Q[Business question in a fictional setting]
    Q --> T[Target definition and data dictionary]
    T --> S[Synthetic source records]
    S --> V[Validation: schema, missingness, ranges, duplicates]
    V --> E[Exploration: distributions, segments, relationships]
    E --> P[Train/test split and preprocessing]
    P --> B[Simple baseline]
    P --> M[Candidate model(s)]
    B --> R[Held-out metrics and error analysis]
    M --> R
    R --> X[Explanation, subgroup checks, limitations]
    X --> O[Outputs: tables, charts, predictions, run metadata]
    O --> D[README, evidence screenshots, optional notebook/dashboard]
```

### 4.1 Component responsibilities

| Component | Responsibility | Evidence to save |
| --- | --- | --- |
| Question/target | Defines what is estimated and what is out of scope | One-paragraph problem statement and target definition |
| Synthetic source | Produces or stores invented records | Generator assumptions, seed, row-count note |
| Validation | Checks schema, values, missingness, duplicates, and leakage risks | `outputs/data_quality_report.csv`, dataset preview, and test assertions |
| Exploration | Describes data without claiming causality | `outputs/eda_summary.csv`, `images/class_distribution.png`, and `images/department_performance.png` |
| Split/preprocessing | Protects test-set integrity and handles feature types | Configuration or run notes |
| Models | Provides an interpretable logistic-regression pipeline | `models/employee_performance_model.joblib` |
| Evaluation | Measures held-out performance and error patterns | `outputs/metrics.json`, `outputs/predictions.csv`, and `images/confusion_matrix.png` |
| Explanation | Shows what the model uses and what it cannot tell us | Feature explanation chart and limitations |
| Packaging | Makes results easy to inspect and reproduce | CLI flags, fixed seed, and stable artifact names |

### 4.2 Suggested data contract

The implementation uses the following synthetic data contract. The generated CSV is written to `data/synthetic_hr_data.csv`:

| Column | Type | Example meaning | Modeling note |
| --- | --- | --- | --- |
| `employee_id` | String | Synthetic row identifier | Keep for traceability; do not use as a predictive feature |
| `department` | Category | Fictional organizational area | Check category coverage |
| `job_level` | Category | Entry, Mid, Senior, or Lead | Watch for small groups |
| `training_hours_last_year` | Numeric | Simulated completed training hours | Define range and missing-value rule |
| `projects_completed` | Numeric | Simulated completed project count | Do not interpret as a universal productivity measure |
| `absenteeism_days` | Numeric | Simulated absence days | Do not interpret as a universal productivity measure |
| `job_satisfaction` | Numeric | Synthetic 1–5 satisfaction response | Document scale and possible noise |
| `years_at_company` | Numeric | Simulated time at the fictional company | Avoid treating tenure as merit |
| `performance_label` | Category | `High` or `Needs Improvement` | Define construction rule and keep out of features |

If a column is used to create the target, check for target leakage. A feature generated after the outcome or copied from the target is not a legitimate predictor for an earlier decision point.

## 5. Ten implementation phases

The phases below are the smallest complete path from question to evidence. Each phase has a purpose, expected outputs, and common mistakes.

### Phase 1 — Define the question and target

**Purpose:** Turn “predict employee performance” into one measurable, bounded question.

**Expected outputs:**

- one-sentence problem statement;
- target type and target definition;
- prediction time point;
- list of allowed features and explicitly excluded fields;
- success criteria that include responsible-use checks, not just a metric.

**Common mistakes:** using a vague target such as “good employee,” mixing future information into inputs, changing the target after seeing the results, or presenting the project as an HR decision system.

### Phase 2 — Create or document the synthetic data source

**Purpose:** Make the fictional population and assumptions explicit before analysis.

**Expected outputs:**

- synthetic-data description;
- generation assumptions and random seed;
- source file or documented generation step;
- initial row/column count;
- data dictionary draft.

**Common mistakes:** copying real employee data, using realistic names or identifiers, hiding the target-generation rule, or implying that synthetic distributions represent a real company.

### Phase 3 — Validate the data contract

**Purpose:** Confirm that the input is structurally usable and that the analyst understands its limits.

**Expected outputs:**

- schema/type check;
- missing-value summary;
- duplicate check;
- range and category validation;
- target-balance summary;
- `data/synthetic_hr_data.csv`;
- `outputs/data_quality_report.csv`.

**Common mistakes:** silently dropping rows, treating every missing value as zero, ignoring impossible values, using an identifier as a feature, or letting validation decisions depend on the test set.

### Phase 4 — Explore the data

**Purpose:** Understand distributions, relationships, unusual records, and possible leakage before modeling.

**Expected outputs:**

- descriptive summary;
- at least one distribution chart;
- at least one relationship or segment chart;
- plain-language captions that distinguish association from cause;
- `outputs/eda_summary.csv`;
- `images/department_performance.png` and selected `.png` files.

**Common mistakes:** creating dozens of uncaptured charts, interpreting a synthetic correlation as a causal fact, ignoring small categories, or using the test set to make repeated modeling decisions.

### Phase 5 — Build the preprocessing pipeline

**Purpose:** Apply consistent transformations while protecting evaluation integrity.

**Expected outputs:**

- numeric and categorical feature lists;
- documented missing-value strategy;
- encoding/scaling decision;
- train/test split rule and random seed;
- a reproducible preprocessing configuration or pipeline.

**Common mistakes:** fitting an imputer or encoder on all rows before the split, manually changing categories between runs, scaling the target accidentally, or failing to record the seed.

### Phase 6 — Train a baseline and candidate models

**Purpose:** Establish a simple reference point before adding complexity.

**Expected outputs:**

- baseline definition, such as a majority-class, mean, or simple linear/logistic model;
- one or more candidate model names;
- training configuration and package versions;
- saved model comparison input.

**Common mistakes:** selecting a complex model without a baseline, tuning against the test set, reporting training metrics as if they were future performance, or using a model whose assumptions are not explained.

### Phase 7 — Evaluate and stress-test

**Purpose:** Measure held-out behavior and understand errors rather than hiding behind one score.

**Expected outputs:**

- `outputs/metrics.json`;
- `outputs/predictions.csv`;
- confusion matrix for classification or actual-vs-predicted chart for regression;
- error examples or residual summary;
- segment/subgroup summaries where meaningful and safe within the synthetic setting.

**Common mistakes:** claiming accuracy without the class distribution, skipping per-class metrics, reporting a single split as certainty, comparing models on different test sets, or treating a metric as a business outcome.

### Phase 8 — Explain results and complete the responsible-use review

**Purpose:** Make the result understandable and put limitations next to the conclusion.

**Expected outputs:**

- feature-importance or coefficient chart with method labeled;
- explanation of false positives and false negatives;
- limitation and non-use statement;
- notes on proxy risk, drift, missing context, and human review;
- a decision about whether the evidence is sufficient to share.

**Common mistakes:** calling feature importance “impact,” presenting a ranking of people, omitting uncertainty, or putting the ethics notice only in a hidden appendix.

### Phase 9 — Package reproducible outputs or a demo

**Purpose:** Make it easy for a reviewer to repeat the run and find the evidence.

**Expected outputs:**

- one documented entry point or notebook order;
- stable output filenames;
- `outputs/metrics.json`;
- optional local dashboard that displays the synthetic-data notice;
- a clean separation between source data, processed data, and generated artifacts.

**Common mistakes:** relying on notebook state, saving outputs with timestamps only, committing large generated files without a policy, including local paths or secrets, or building a dashboard before validating the analysis.

### Phase 10 — Document, capture evidence, and publish

**Purpose:** Turn the work into an honest GitHub-ready portfolio artifact.

**Expected outputs:**

- completed README and project guide;
- exact screenshots named according to the checklist;
- reproducible run instructions;
- commit history that shows meaningful stages;
- final limitation statement and submission checklist.

**Common mistakes:** screenshots without filenames or context, claims that cannot be reproduced, a README that omits setup, committing a virtual environment, or presenting synthetic results as company evidence.

## 6. Folder structure and file responsibilities

The implemented repository uses this structure:

```text
employee-performance-predictor/
├── README.md
├── PROJECT_GUIDE.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── main.py                         # Runnable CLI entry point
├── data/
│   └── synthetic_hr_data.csv       # Generated synthetic source data
├── src/
│   ├── config.py                   # Paths, constants, and validation
│   ├── analysis.py                 # Data-quality and EDA summaries
│   ├── data_generation.py          # Reproducible synthetic data generator
│   ├── preprocessing.py            # Leakage-safe feature selection/encoding
│   ├── model_training.py           # Train/test split and logistic regression
│   ├── evaluation.py               # Metrics and JSON persistence
│   ├── predictions.py              # Prediction table creation
│   ├── visualizations.py            # Saved matplotlib charts
│   ├── pipeline.py                 # End-to-end orchestration
│   └── cli.py                      # Command-line arguments
├── models/
│   └── employee_performance_model.joblib
├── outputs/
│   ├── data_quality_report.csv
│   ├── eda_summary.csv
│   ├── metrics.json
│   └── predictions.csv
├── images/
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── feature_coefficients.png
│   └── department_performance.png
└── tests/
    └── test_smoke.py              # Reproducibility and end-to-end checks
```

Keep the tree honest: remove paths that are not used, and do not add empty claims such as “dashboard available” until there is a tested dashboard.

## 7. Setup instructions

### 7.1 Prerequisites

Recommended starting tools:

- Python 3.10 or newer, subject to dependency compatibility;
- Git;
- a code editor or Jupyter-capable environment; and
- enough disk space for the virtual environment and generated artifacts.

Check the installed tools:

**Windows PowerShell**

```powershell
py --version
git --version
```

**macOS/Linux**

```bash
python3 --version
git --version
```

If the project later pins another Python version in `requirements.txt` or a project configuration file, follow that documented version instead.

### 7.2 Windows PowerShell

From the repository root:

```powershell
py -3 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use a permitted shell or follow the local machine’s execution-policy guidance. Do not commit the `.venv` directory.

Deactivate when finished:

```powershell
deactivate
```

### 7.3 macOS/Linux

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Deactivate when finished:

```bash
deactivate
```

### 7.4 Optional notebook setup

If the chosen implementation is notebook-first, install and launch Jupyter only after the dependency list is finalized:

```bash
python -m pip install jupyter
jupyter notebook
```

On Windows PowerShell, the same `python -m` commands work inside the activated environment.

## 8. How to run the project

The implementation has one stable run contract. Run it from the repository root after activating the virtual environment:

### 8.1 Runnable script contract

The beginner-friendly entry point is:

```bash
python main.py
```

Useful repeatable variants are:

```bash
python main.py --samples 1000 --seed 42
python main.py --samples 250 --seed 42
python -m src --samples 80 --seed 11 --project-root /tmp/employee-performance-smoke
```

The run creates `data/synthetic_hr_data.csv`, `outputs/data_quality_report.csv`, `outputs/eda_summary.csv`, `models/employee_performance_model.joblib`, `outputs/metrics.json`, `outputs/predictions.csv`, and four PNG charts. Use `--project-root` to write a clean test run to another directory without changing the repository artifacts.

### 8.2 Recommended notebook order

If you later add notebooks, run them in this order:

1. `notebooks/01_data_quality_and_eda.ipynb`;
2. save the data-quality report and exploratory outputs;
3. `notebooks/02_model_evaluation.ipynb`;
4. save metrics, predictions, charts, and run metadata; and
5. restart the kernel and run all cells from top to bottom to check reproducibility.

### 8.3 Run validation after execution

Confirm all of the following:

- the command exits successfully;
- output files exist with the documented names;
- the row count in `outputs/metrics.json` matches the generated CSV;
- `outputs/predictions.csv` contains only the held-out test rows;
- the model feature list excludes `employee_id` and `performance_label`;
- metrics are labeled with the correct split;
- charts have readable titles and units;
- the test set was not used to fit preprocessing or tune repeatedly; and
- the README contains no metric values that are absent from the saved outputs.

## 9. Virtual company simulation

### 9.1 Scenario

Use **Northstar Services** as a fictional company with several role families, such as customer support, operations, engineering support, and project coordination. The company name and roles are invented for the exercise. They do not represent an actual employer.

### 9.2 Simulation design questions

Before generating records, write down:

- What is one observation: a person-period, a team-period, or a project-period?
- What time window does each row represent?
- Which fields are available at the prediction time?
- How is the synthetic target created?
- Which random seed makes the run repeatable?
- Which columns are excluded because they are identifiers, post-outcome data, or sensitive/proxy fields?
- What kinds of noise, missingness, and imbalance are intentionally included?

### 9.3 Example synthetic fields

The following fields are examples only. Adjust them to the target and document the final schema:

| Area | Illustrative fields | Questions to answer |
| --- | --- | --- |
| Organization | `department`, `role_family` | Are categories large enough to summarize? |
| Work context | `projects_completed`, `tenure_months` | What does the period mean? Are counts comparable across roles? |
| Development | `training_hours`, `training_modules_completed` | Does completion mean attendance or learning? |
| Feedback | `feedback_score` | Who created the score and what is its scale in the simulation? |
| Availability | `attendance_rate` | What does the synthetic rate represent, and what context is missing? |
| Outcome | `performance_band` or `simulated_score` | Is it a label, a ranking, or a continuous measure? |

### 9.4 Questions a fictional stakeholder might ask

Practice answering these without overstating the model:

- Which fields are most associated with the simulated target?
- Where does the model make mistakes?
- Are the classes balanced?
- Does performance differ by role family in the synthetic data?
- What would need to be validated before any real-world use?
- Which data should never be collected or used for this purpose?

The correct answer to the last two questions must include: real-world validation, legal/HR review, privacy safeguards, fairness analysis, human oversight, appeal mechanisms, and a clear non-use boundary for high-impact employment decisions.

## 10. Exact screenshots and evidence checklist

Capture evidence only after a real run. Use the exact filenames below so a reviewer can follow the story. Do not include personal data, API keys, local usernames, or unredacted file paths.

| # | Exact filename | What must be visible | Acceptance check |
| ---: | --- | --- | --- |
| 1 | `docs/evidence/01-repository-overview.png` | GitHub/local tree with README, guide, and top-level folders | The project purpose is recognizable in one screen |
| 2 | `docs/evidence/02-environment-ready.png` | Terminal showing Python and Git versions, with the environment activated if applicable | Versions are readable; no secrets appear |
| 3 | `docs/evidence/03-data-preview.png` | First rows or a documented preview of the synthetic dataset | Synthetic label and target column are visible |
| 4 | `docs/evidence/04-data-quality-report.png` | Missingness, duplicates, types/ranges, and target balance | The report is traceable to `outputs/data_quality_report.csv` |
| 5 | `docs/evidence/05-eda-results.png` | One distribution and one relationship/segment view | Titles, axes, and plain-language interpretation are readable |
| 6 | `docs/evidence/06-model-training.png` | The actual training command or notebook section | The command matches the documented run contract |
| 7 | `docs/evidence/07-model-metrics.png` | Metrics table with model names and held-out split label | Values match `outputs/metrics.json`; no invented values |
| 8 | `docs/evidence/08-error-analysis.png` | Confusion matrix, residual view, or error table | The error type and target formulation are clear |
| 9 | `docs/evidence/09-explanation.png` | Feature-importance/coefficient view and its method label | Caption says association, not causation |
| 10 | `docs/evidence/10-reproducible-run.png` | Final command output plus output directory listing | Expected files exist and the run completed without hidden manual steps |
| 11 | `docs/evidence/11-github-history.png` | GitHub commit history or local `git log --oneline` | Commits are descriptive and secrets are absent |
| 12 | `docs/evidence/12-responsible-use-notice.png` | README or demo page showing the synthetic-data warning | The non-employment-use statement is visible, not buried |

### Evidence notes to keep with the screenshots

Create a small evidence index in the final project or pull request with:

- capture filename;
- date and environment;
- command or notebook used;
- output file(s) it proves;
- one-sentence interpretation; and
- any limitation or unresolved gap.

Suggested output references are `data/synthetic_hr_data.csv`, `outputs/data_quality_report.csv`, `outputs/eda_summary.csv`, `outputs/metrics.json`, `outputs/predictions.csv`, `images/class_distribution.png`, `images/confusion_matrix.png`, `images/feature_coefficients.png`, and `images/department_performance.png`. Use only the files that the implementation actually generates.

## 11. GitHub upload commands

Run these from the project root after reviewing the files. Replace the placeholder remote URL with the repository URL you own or are authorized to use.

### 11.1 First upload

```bash
git init
git add README.md PROJECT_GUIDE.md .gitignore LICENSE
git commit -m "docs: add project guide and repository documentation"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repository>.git
git push -u origin main
```

If implementation files are added later, review `git status` and `git diff --cached` before staging them. Never commit `.env`, `.venv`, tokens, passwords, personal employee information, or unreviewed generated data.

### 11.2 Later updates

```bash
git status
git add README.md PROJECT_GUIDE.md docs/evidence
git commit -m "docs: record reproducible model evidence"
git push
```

Only stage paths that have been checked. Generated files under `outputs/` are ignored by default; if a small, reviewed artifact should be versioned, add it explicitly after confirming that it contains no sensitive information, for example `git add -f outputs/metrics.json`.

### 11.3 Optional branch workflow

```bash
git switch -c docs/project-guide
git add README.md PROJECT_GUIDE.md .gitignore LICENSE
git commit -m "docs: establish project scope and runbook"
git push -u origin docs/project-guide
```

Open a pull request with the problem statement, files changed, checks run, screenshots, actual outputs, and known limitations.

## 12. Suggested commit plan

Use small commits that tell the project story:

1. `docs: define synthetic project scope and responsible use`
2. `docs: add data contract and architecture guide`
3. `feat: add synthetic data generation and validation` — only when implementation exists
4. `feat: add exploratory analysis outputs`
5. `feat: add baseline and candidate model evaluation`
6. `docs: add evidence checklist and interview notes`
7. `chore: finalize ignore rules and license`

Do not create an implementation-sounding commit until the corresponding code and evidence exist. The implementation is already present, so keep implementation and documentation commits separate when possible. If you publish generated artifacts, record the exact command and seed that created them.

## 13. Five-day proof-building strategy

The goal is not to produce a flashy score. The goal is to build a small, reproducible proof that you understand the full analytics workflow.

| Day | Focus | Work to complete | Proof to save |
| ---: | --- | --- | --- |
| 1 | Scope and setup | Choose target formulation, write the data dictionary, choose Option A, create the environment, and confirm the run contract | Problem statement, data contract, environment screenshot |
| 2 | Data and exploration | Generate or document synthetic records, validate schema/missingness/ranges, and create a small set of useful charts | Data-quality report, data preview, EDA screenshots |
| 3 | Baseline and candidates | Implement the split/preprocessing flow, train a simple baseline, and compare one or more candidates | Training command, model configuration, initial comparison table |
| 4 | Evaluation and explanation | Evaluate on held-out data, inspect errors, label the explanation method, and complete responsible-use review | Metrics table, error analysis, explanation chart, limitation notes |
| 5 | Proof and presentation | Re-run from a clean environment, update README, capture exact evidence, clean Git history, and rehearse pitches | Reproducible-run screenshot, GitHub-history screenshot, final checklist |

At the end of Day 5, any missing evidence should be called out as a gap. Do not replace it with a guessed metric or an illustrative screenshot.

## 14. Interview preparation

### 14.1 Ten common questions and strong answers

**1. What problem did you solve?**  
I built an educational workflow that estimates a clearly defined synthetic performance outcome from synthetic tabular records. The main goal was to demonstrate data validation, modeling, evaluation, and responsible communication—not to judge real employees.

**2. Why did you use synthetic data?**  
It lets me practice the workflow without exposing personal or confidential information. It also makes the assumptions visible. I would not treat synthetic results as evidence that a real workplace model is valid.

**3. How did you define the target?**  
I selected one target type before training, documented its meaning and time point, and kept any identifier or post-outcome field out of the feature set. The exact target definition is in the data dictionary.

**4. Why did you choose the Python stack?**  
Python with pandas and scikit-learn gives a transparent, low-cost workflow for tabular data. It lets me show each step—from validation through evaluation—before adding a dashboard.

**5. How did you avoid leakage?**  
I split the data before fitting learned preprocessing, kept the target out of the features, and checked whether any field was created after the prediction point. I also kept the held-out test set for final evaluation.

**6. Why use a baseline?**  
A baseline tells me whether a more complex model improves over a simple reference. Without it, a candidate score has no useful context.

**7. Which metric did you choose?**  
I chose metrics based on the target type and class balance, and I reported more than one view where appropriate. For classification, I would inspect per-class results and a confusion matrix; for regression, I would inspect MAE/RMSE and residual behavior. The actual values must come from the saved output.

**8. What do feature-importance results mean?**  
They describe how the fitted model used the available variables in this synthetic dataset. They do not prove causation, fairness, or that changing a feature would change a person’s performance.

**9. What are the biggest limitations?**  
The data and labels are synthetic, the measurement assumptions may be unrealistic, subgroup behavior may be unstable, and a table omits human context. A real deployment would require authorization, privacy, legal/HR review, fairness testing, monitoring, human oversight, and an appeal process.

**10. Would you deploy this for promotion decisions?**  
No. This project is explicitly not for employment decisions. If the organization wanted to explore analytics responsibly, I would start with aggregate, non-identifying learning insights and a formal governance review, not individual ranking or automated decisions.

### 14.2 HR-friendly project pitch

> “I created a synthetic employee-performance analytics project to demonstrate how I approach an ambiguous business question. I defined a measurable target, checked data quality, explored patterns, compared a simple baseline with a candidate model, and documented the errors and limitations. I was careful to label the data as synthetic and state that the result is not for employment decisions. The project shows how I communicate technical work responsibly, not just how I produce a score.”

### 14.3 Technical project pitch

> “This is a reproducible tabular supervised-learning workflow. I formalized the target and data contract, validated schema and missingness, split the data before fitting learned preprocessing, trained a baseline and candidate models, and evaluated them on held-out data with target-appropriate metrics. I saved metrics, predictions, charts, and run metadata, then added error and responsible-use analysis. Because the dataset is synthetic, I treat the metrics as a workflow demonstration rather than a performance claim about a real organization.”

### 14.4 Follow-up prompts to rehearse

- Show me the data dictionary.
- What would make the target invalid?
- Which result changed your thinking?
- What would you test before using real data?
- How would you monitor drift?
- How would you explain a false positive to a non-technical stakeholder?
- What would you remove from the project if time were limited?

## 15. Future improvements

Prioritize improvements in this order:

1. **Data contract:** schema validation, explicit versions, and a documented target-generation rule.
2. **Reproducibility:** pinned dependencies, configuration files, fixed seeds, and a clean-run command.
3. **Evaluation quality:** cross-validation, confidence intervals or uncertainty discussion, calibration, and per-segment error checks.
4. **Explainability:** model cards, example-level explanations, and clear separation of association from causation.
5. **Responsible governance:** privacy review, access control, retention policy, human oversight, appeal process, and an explicit non-use policy for employment decisions.
6. **User experience:** optional Streamlit demo that keeps the synthetic-data warning visible and avoids ranking people.
7. **Operations:** data-drift checks, dependency updates, artifact versioning, and CI checks for documentation and data contracts.
8. **Alternative framing:** aggregate team-learning or training-planning insights that do not produce individual employment scores.

## 16. Troubleshooting

| Symptom | Likely cause | What to check |
| --- | --- | --- |
| `python` is not recognized on Windows | Python is not on PATH or the launcher name differs | Try `py --version`, install Python for the user, and reopen the terminal |
| `python3` is missing on macOS/Linux | Python is not installed or has another command name | Check the platform package manager and use the documented supported version |
| Activation is blocked in PowerShell | Local execution policy prevents the activation script | Use an approved shell or the environment’s documented policy; do not commit the environment |
| `pip install` fails | Wrong interpreter, network issue, or incompatible dependency | Run `python -m pip --version`, confirm the active environment, and read the first dependency error |
| Notebook works only after running cells out of order | Hidden state in the kernel | Restart the kernel and run all cells from top to bottom |
| Metrics are unexpectedly perfect | Target leakage, duplicate rows, or a target-generation shortcut | Inspect feature columns, split order, duplicates, and any post-outcome fields |
| A category is unseen at evaluation time | Train/test category mismatch | Use a documented unknown-category strategy and report small groups |
| One class has no useful recall | Imbalance or too few synthetic examples | Inspect class counts, use appropriate metrics, and document the limitation |
| Results change between runs | No fixed seed or nondeterministic step | Record the seed, package versions, and configuration in `run_metadata.json` |
| Output files cannot be found | Working directory or output path differs | Run from the repository root and use the documented `--output-dir` |
| Charts show local usernames or sensitive paths | Screenshot captured too much terminal context | Crop/redact before publishing; never commit secrets or personal data |
| A dashboard makes the result look authoritative | Interface was built before limitations were designed | Keep the synthetic-data warning visible and show uncertainty/error context |

When troubleshooting, preserve the first error message, the command, the working directory, the Python version, and whether the virtual environment was active. Do not “fix” a reproducibility issue by manually editing generated outputs.

## 17. Final submission checklist

### Scope and ethics

- [ ] README states that the data is synthetic.
- [ ] README and guide state that the project is not for employment decisions.
- [ ] Target and prediction time point are defined.
- [ ] No real employee records, names, identifiers, or confidential files are included.
- [ ] Claims are limited to actual generated evidence.

### Technical workflow

- [ ] Data dictionary and validation report exist.
- [ ] Train/test split and preprocessing order are documented.
- [ ] Baseline and candidate models are named.
- [ ] Metrics are labeled with the correct split and target type.
- [ ] Error analysis and explanation are present.
- [ ] Random seed and environment details are recorded.

### GitHub proof

- [ ] Setup and run commands are tested in the documented environment.
- [ ] Recommended output filenames match the actual files.
- [ ] Exact screenshots are captured and redacted.
- [ ] `.gitignore` excludes environments, secrets, logs, caches, and unintended generated data.
- [ ] Commits are descriptive and do not contain secrets.
- [ ] License placeholder is updated before publication.

### Communication

- [ ] HR-friendly and technical pitches are rehearsed.
- [ ] At least one limitation is stated for every major conclusion.
- [ ] No metric is presented without its source artifact.
- [ ] The final demo makes the synthetic-data warning visible.
