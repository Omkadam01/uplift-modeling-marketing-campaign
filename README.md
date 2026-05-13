# Uplift Modeling Marketing Campaign Optimization

An advanced causal machine learning project to identify which customers should receive a marketing intervention and which should not using uplift modeling on a randomized controlled trial dataset.

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)](https://shap.readthedocs.io/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/bofulee/kevin-hillstrom-minethatdata-e-mailanalytics)

---

## Why This Project Is Different

Most machine learning projects answer: *"Who will convert?"*

This project answers a fundamentally harder question: *"Who will convert **because** of our intervention and who would have converted anyway?"*

This distinction is the difference between predictive modeling and causal reasoning. A standard classification model cannot identify customers who are harmed by an intervention (Sleeping Dogs), cannot distinguish sure-thing converters from persuadable ones, and therefore cannot optimize campaign ROI. Uplift modeling solves all three problems.

---

## Business Problem

A retail company sends email campaigns to its customer base. Sending emails to the wrong customers wastes budget in two ways:

- Contacting customers who would have converted anyway — wasted spend on Sure Things
- Contacting customers who are actually deterred by the email — active harm from Sleeping Dogs

The goal is to identify the Persuadables — the narrow segment of customers whose conversion probability genuinely increases as a direct result of receiving the email — and target only them.

---

## The Four Customer Types in Uplift Modeling

| Type | Behavior | Action |
|---|---|---|
| Persuadables | Convert only if contacted | Target - high ROI |
| Sure Things | Convert regardless | Skip - wasted spend |
| Lost Causes | Don't convert regardless | Skip - no value |
| Sleeping Dogs | Less likely to convert if contacted | Avoid - active harm |

A standard churn or conversion model cannot distinguish between these four groups.

---

## Dataset

| Property | Details |
|---|---|
| Source | [Hillstrom E-Mail Analytics — Kaggle](https://www.kaggle.com/datasets/bofulee/kevin-hillstrom-minethatdata-e-mailanalytics) |
| Size | 42,693 customers (after filtering to binary treatment) |
| Experiment | Randomized Controlled Trial — Women's Email vs No Email |
| Treatment | 1 = received email, 0 = no email |
| Outcome | Binary conversion (purchase after campaign) |
| Baseline conversion rate | 0.57% (control) vs 0.88% (treatment) |

---

## Project Structure

```
uplift-modeling-marketing-campaign/
│
├── uplift-modeling-marketing-campaign.ipynb
├── outputs/
│   └── model_comparison.csv
└── README.md
```

---

## Methodology

### 1. Experiment Design & EDA

- Confirmed randomized treatment/control split with balanced group sizes (21,387 treatment vs 21,306 control)
- Calculated Average Treatment Effect (ATE) email caused measurable conversion lift on average
- Discovered treatment effect is highly heterogeneous across segments:
  - Recency segment 3 showed -18.5% uplift (Sleeping Dogs)
  - History segment $500+ showed +100% uplift (strong Persuadables)
  - History segment $200–$500 showed -6.7% to -21.6% uplift (mid-tier Sleeping Dogs)
- Demonstrated why ATE alone is insufficient — averaging hides the heterogeneity that drives ROI

### 2. Feature Engineering

| Feature | Logic |
|---|---|
| `RecencyScore` | 1 / (recency + 1) - recent customers more responsive |
| `IsHighValue` | history > 500 - segment with strongest positive uplift |
| `IsMidValue` | history $200–500 - segment with negative uplift |
| `EngagementScore` | history / (recency + 1) - combined engagement signal |

### 3. Uplift Models

Two meta-learner approaches were implemented and compared:

**T-Learner (Two-Model Approach)**
- Train separate RandomForest models on treatment and control groups independently
- Uplift score = P(convert | treated) − P(convert | control)
- Better captures treatment heterogeneity by allowing each model to specialize

**S-Learner (Single Model Approach)**
- Train one model with treatment as an input feature
- Uplift score = P(convert | X, T=1) − P(convert | X, T=0)
- Tends to underweight treatment signal on small datasets

### 4. Evaluation — Qini Curve & Coefficient

| Model | Qini Coefficient |
|---|---|
| **T-Learner** | **162.11** |
| S-Learner | 130.17 |
| Random Baseline | 0.00 |

The Qini curve measures cumulative incremental conversions as customers are contacted in descending order of uplift score. The Qini coefficient is the area between the model curve and random targeting baseline higher is better.

T-Learner outperformed S-Learner by 24.5% on Qini coefficient, confirming that separate treatment/control modeling captures causal heterogeneity more effectively.

### 5. Customer Segmentation Results

| Segment | Customers | Avg Uplift | Action |
|---|---|---|---|
| High Persuadables | 848 | +7.67% | Target |
| Low Persuadables | 688 | +1.01% | Borderline |
| Lost Causes | 6,099 | ~0.00% | Skip |
| Sleeping Dogs | 904 | -4.45% | Never contact |

### 6. Business ROI Analysis

| Strategy | Contacts | Net ROI | vs Baseline |
|---|---|---|---|
| Target Everyone | 8,539 | -$17,107.50 | — |
| Target Persuadables Only | 619 | -$1,227.50 | +$15,880.00 saved |

- **92.8% reduction** in contacts needed to capture incremental conversions
- **$15,880 in campaign budget saved** vs blanket targeting
- **$1,367.50 saved** specifically by avoiding Sleeping Dogs

### 7. SHAP Explainability

SHAP was applied to both treatment and control models separately. Uplift SHAP — the difference between treatment and control SHAP values — provides causal feature importance: which customer attributes drive the treatment effect itself, not just predictive accuracy.

---

## Key Findings

**The treatment effect is concentrated in a small segment.** Only 9.9% of customers (848 out of 8,539) qualify as High Persuadables. A blanket campaign wastes 90%+ of its budget on customers who either convert anyway or are actively deterred.

**Sleeping Dogs are a real and costly phenomenon.** 904 customers showed negative uplift — contacting them reduces their conversion probability by an average of 4.45%. A standard model would have flagged many of these as targets based on their demographic profile alone.

**Mid-tier spenders ($200–$500) are Sleeping Dogs, not targets.** This is counterintuitive — these are seemingly valuable customers. But the data shows the email intervention harms their conversion. This insight is only discoverable through uplift modeling, not standard classification.

**T-Learner outperforms S-Learner for this problem.** When treatment heterogeneity is high, allowing separate models to specialize on each group captures causal structure that a single model with treatment as a feature cannot.

---

## Business Recommendation

Target only customers with uplift score > 0.02 (High Persuadables). Explicitly exclude customers with uplift score < -0.01 (Sleeping Dogs) from all campaign lists. For borderline customers (0.0–0.02), apply a cost-benefit threshold based on email cost vs expected incremental revenue per segment.

This targeting strategy reduces campaign spend by 92.8% while preserving incremental conversion gains — the definition of efficient marketing spend.

---

## How This Differs From Standard Classification

| Aspect | Standard Model | Uplift Model |
|---|---|---|
| Question answered | Who will convert? | Who will convert because of intervention? |
| Sleeping Dogs | Cannot identify | Explicitly identified and excluded |
| Evaluation metric | AUC-ROC, F1 | Qini coefficient, incremental gain |
| Business output | Ranked conversion list | ROI-optimized targeting list |
| Causal reasoning | No | Yes — counterfactual framework |

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| ML Models | Scikit-learn (RandomForest) |
| Explainability | SHAP |
| Uplift Evaluation | Custom Qini implementation |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Environment | Kaggle Notebooks |

---

## Author

**Om Kiran Kadam**
B.Tech - Artificial Intelligence & Data Science, Sanjivani University

[LinkedIn](https://linkedin.com/in/omkadam05) · [GitHub](https://github.com/Omkadam01) · [Kaggle](https://www.kaggle.com/omkadam05)

---

## License

This project is licensed under the MIT License.
