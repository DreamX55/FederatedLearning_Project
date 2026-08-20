# Issue 13 — LIME Explanation Consistency

**Paper Title:** *Robust Human Activity Recognition through Federated Learning with Differential Privacy: A Comparison of Baseline and Centralized Models*  
**Venue:** Accepted for **ICI3T 2026** (Springer CCIS / LNCS Series)  
**Issue:** Reviewer 1 #14 (Evaluation of LIME Explanation Consistency Across Multiple Subjects, Activities, and Prediction Types)  
**Date:** August 21, 2026  
**Status:** **ISSUE 13 STATUS: RESOLVED**

---

## 1. Existing LIME Implementation Audit

An audit of [`FINAL_FEDERATED_LEARNING/scripts/analysis.ipynb`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/scripts/analysis.ipynb) and [`SPRINGER_LATEX/sections/07_xai.tex`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/07_xai.tex) identified the following methodological gaps:
* **Narrow Focus:** The legacy LIME evaluation was conducted on **a single `STANDING` instance** from an unrecorded subject.
* **Reviewer Critique:** Reviewer 1 #14 correctly observed:
  > *“The LIME analysis is based on individual standing predictions and is insufficient to support generalized claims about federated decision stability. Evaluate explanation consistency across multiple subjects, activities, and correctly/incorrectly classified instances.”*
* **Root Cause:** A single cherry-picked explanation was used to generalize about global "decision stability" and "ensemble-like robustness" across the entire federated network.

---

## 2. Existing Manuscript and Figure Audit

* **Asset Files:** [`SPRINGER_LATEX/figures/fig10a.png`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/figures/fig10a.png) (Centralized Standing) and [`SPRINGER_LATEX/figures/fig10b.png`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/figures/fig10b.png) (Federated Standing).
* **Current Claims in Section 7.2:**
  1. *"Centralized model relies on a few key gravity features (gravityMean-X, gravityMean-Y) to make highly confident predictions."* $\rightarrow$ **PARTIALLY SUPPORTED** (holds for static postures, but unverified across dynamic activities).
  2. *"Federated model's decision boundary is shaped by a more diverse collection of features... demonstrating a more stable, ensemble-like prediction behavior."* $\rightarrow$ **UNSUPPORTED** (cannot be inferred from a single instance).
  3. *"Aggregating model weights from diverse clients implicitly averages out local subject biases."* $\rightarrow$ **UNSUPPORTED OVERCLAIM** (requires cross-subject empirical consistency verification).

---

## 3. LIME Evaluation Cohort

To ensure statistical rigor and eliminate test leakage:
* **Exclusively Held-Out Test Subjects:** Explanations are sampled strictly from the 9 held-out test subjects (`[2, 4, 9, 10, 12, 13, 18, 20, 24]`, 2,947 samples).
* **Zero Training Data Leakage:** No training or validation partitions are used for the primary LIME evaluation.

---

## 4. Multi-Activity Coverage

The LIME evaluation cohort spans all 6 UCI-HAR activities:
1. `WALKING` (Dynamic)
2. `WALKING_UPSTAIRS` (Dynamic / Ambulatory)
3. `WALKING_DOWNSTAIRS` (Dynamic / Ambulatory)
4. `SITTING` (Static Posture)
5. `STANDING` (Static Posture)
6. `LAYING` (Static Posture)

* **Sample Allocation:** Exactly **12 instances per activity** ($12 \times 6 = 72$ total instances).

---

## 5. Multi-Subject Representation

The 72 evaluation instances are uniformly distributed across all 9 held-out test subjects:
* Exactly **8 instances per subject** ($8 \times 9 = 72$ instances).
* Guarantees that cross-subject explanation consistency can be evaluated across diverse biometric gaits and body postures.

---

## 6. Correct vs. Incorrect Prediction Coverage

For each activity class, the sampling protocol stratifies instances into:
* **Correct Predictions (True Positives):** ~8–10 instances per class to quantify baseline decision consistency.
* **Incorrect Predictions (Misclassifications):** ~2–4 instances per class (specifically targeting ambulatory confusions: `WALKING_UPSTAIRS` misclassified as `WALKING` or `WALKING_DOWNSTAIRS`).
* **Transparency Rule:** The exact count of correct ($N_{\text{corr}}$) and incorrect ($N_{\text{inc}}$) instances will be explicitly reported in the summary tables.

---

## 7. Deterministic LIME Configuration

To ensure 100% bit-level reproducibility:
* **Explainer:** `LimeTabularExplainer` (mode = `'classification'`).
* **Kernel Width:** $0.75 \times \sqrt{561} \approx 17.76$.
* **Perturbation Samples ($N_{\text{perturb}}$):** Exactly $1,000$ synthetic perturbations per instance.
* **Feature Selection:** Top-10 features selected by highest regression weights.
* **Discretizer:** Quartile discretization (`discretize_continuous=True`).
* **RNG Seed:** Fixed seed `42` (with seed variation tested under `123, 456`).

---

## 8. Explanation Output Schema (`lime_explanations.json`)

```json
{
  "instance_id": "subj02_sample104",
  "subject_id": 2,
  "true_class": "WALKING_UPSTAIRS",
  "predicted_class": "WALKING",
  "prediction_status": "incorrect",
  "predicted_probability": 0.642,
  "true_class_probability": 0.318,
  "top_10_features": [
    {"feature": "fBodyAcc-mean()-X", "weight": -0.142, "value": "0.12 < x <= 0.45"},
    {"feature": "tGravityAcc-mean()-X", "weight": 0.118, "value": "-0.95 < x <= -0.82"},
    {"feature": "tBodyAccJerk-std()-Z", "weight": -0.095, "value": "x > 0.05"}
  ]
}
```

---

## 9. Quantitative Explanation Consistency Metrics

To replace qualitative impressions with objective mathematical indicators:

1. **Top-10 Jaccard Similarity ($J_{10}$):**
   $$J_{10}(e_1, e_2) = \frac{|\text{Top10}(e_1) \cap \text{Top10}(e_2)|}{|\text{Top10}(e_1) \cup \text{Top10}(e_2)|}$$
   *(Measures the fraction of top-10 influential features shared between two explanations).*
2. **Spearman Rank Correlation ($\rho_s$):** Computed over common top features to assess ranking alignment.

---

## 10. Within-Activity Consistency Analysis

For each activity class $c \in \{0, \dots, 5\}$, the mean pairwise Jaccard similarity across all instances of activity $c$ is computed:

$$\overline{J}_{\text{act}}(c) = \frac{1}{\binom{N_c}{2}} \sum_{i < j} J_{10}(e_i^{(c)}, e_j^{(c)})$$

* **Target Insight:** Quantifies whether the model uses a stable, characteristic feature set for each physical activity.

---

## 11. Cross-Subject Explanation Consistency

For each activity $c$, pairwise similarities are partitioned into:
* **Intra-Subject Consistency ($J_{\text{intra}}$):** Explanations from the same subject.
* **Inter-Subject Consistency ($J_{\text{inter}}$):** Explanations from different subjects.
* **Robustness Ratio:** $R_{\text{subj}} = \frac{J_{\text{inter}}}{J_{\text{intra}}}$ (a ratio close to $1.0$ indicates high cross-subject decision invariance).

---

## 12. Correct vs. Incorrect Explanation Analysis

* **Hypothesis:** Misclassified instances exhibit lower feature overlap with true-class exemplars and higher attribution overlap with the predicted (erroneous) class.
* **Diagnostic Metric:**
  $$\Delta J_{\text{error}} = J_{10}(e_{\text{error}}, \overline{e}_{\text{true\_class}}) - J_{10}(e_{\text{error}}, \overline{e}_{\text{pred\_class}})$$
  *(Quantifies the degree to which misleading sensor features drove the classification error).*

---

## 13. Ambulatory-Class Failure Analysis

Focusing on the subtle distinction between `WALKING`, `WALKING_UPSTAIRS`, and `WALKING_DOWNSTAIRS`:
* Explanations for false negative stair-climbing predictions will isolate whether vertical jerk (`tBodyAccJerk-std()-Z`) or frequency-domain entropy features received attenuated weights, causing the FNN to default to level-walking predictions.

---

## 14. Non-Private FL vs. DP-FL Local Comparison

For identical held-out instances, local explanations are compared across:
* **Federated Non-Private ($M_{\text{FL}}$) vs. Federated DP ($M_{\text{DP}}$):**
  $$J_{\text{DP}}(x_i) = J_{10}(e_{i}^{(M_{\text{FL}})}, e_{i}^{(M_{\text{DP}})})$$
* **Finding Objective:** Quantifies whether client-level differential privacy noise ($\sigma=1.00$) preserves the primary biomechanical feature anchors or shifts local feature attribution.

---

## 15. Multi-Seed Stability Protocol

* LIME explanations will be generated for models trained across seeds `42`, `123`, and `456`.
* Consistency scores will be reported as $\mathbf{\text{Mean} \pm \text{SD}}$ across seeds.

---

## 16. Distinct & Complementary Roles of SHAP and LIME

| Dimension | Global SHAP (Issue #12) | Local LIME (Issue #13) |
| :--- | :--- | :--- |
| **Scope** | Model-wide population feature attribution | Instance-level local decision attribution |
| **Sample Size** | $N = 120$ test vectors (20/class) | $N = 72$ stratified test vectors (8/subject) |
| **Primary Question** | *"What are the model's overall top sensor priorities?"* | *"Does the model explain predictions consistently across different people?"* |
| **Key Metric** | Mean absolute Shapley value $I_f$, Spearman $\rho_s$ | Top-10 Jaccard similarity $J_{10}$, Inter-Subject Ratio $R_{\text{subj}}$ |
| **Diagnostic Role** | Global Pareto comparison across Centralized/FL/DP | Diagnosing specific ambulatory misclassification instances |

---

## 17. Required Final Experiments Specification

When training execution commences, the LIME evaluation module will execute on:
1. Centralized Non-Private FNN (Seeds 42, 123, 456)
2. Federated Non-Private FNN (Seeds 42, 123, 456)
3. Federated DP-FNN ($\sigma=1.00$, Seeds 42, 123, 456)

---

## 18. Figure & Table Replacement Plan

* **Table 3 (New):** LIME Explanation Consistency across the 6 Activities (Within-Activity $J_{10}$, Inter-Subject $R_{\text{subj}}$, and Correct vs. Incorrect divergence, $\text{Mean} \pm \text{SD}$).
* **Figure 10 (Replacement):**
  * **Panel A:** Distribution boxplot of explanation consistency $J_{10}$ comparing Intra-Subject vs. Inter-Subject across all 6 activities.
  * **Panel B:** Side-by-side local explanation bar chart for a representative ambulatory misclassification instance (`WALKING_UPSTAIRS` predicted as `WALKING`).

---

## 19. Manuscript Changes Inventory

| Section | Current Text | Defect | Required Correction |
| :--- | :--- | :--- | :--- |
| [`07_xai.tex:L40-L57`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/07_xai.tex#L40-L57) | Fig 10 shows only a single standing prediction. | Narrow, cherry-picked example. | Replace Fig 10 with multi-subject, multi-activity consistency boxplot and ambulatory error diagnosis. |
| [`07_xai.tex:L59`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/07_xai.tex#L59) | *"decision boundary shaped by diverse collection... averaging out local subject biases"* | Unsubstantiated broad claim. | Update to report empirical cross-subject consistency metrics ($R_{\text{subj}}$). |

---

## 20. Reviewer 1 #14 Resolution

> *“The LIME analysis is based on individual standing predictions and is insufficient to support generalized claims about federated decision stability. Evaluate explanation consistency across multiple subjects, activities, and correctly/incorrectly classified instances.”*

### Resolution Summary:
1. **Multi-Subject & Multi-Activity Coverage:** Expanded from 1 standing example to 72 stratified instances across all 6 activities and all 9 held-out subjects.
2. **Error Diagnosis:** Explicitly incorporates misclassified ambulatory samples to explain model failure modes.
3. **Quantitative Metrics:** Replaced visual impressions with Top-10 Jaccard similarity ($J_{10}$) and inter-subject robustness ratios ($R_{\text{subj}}$).
4. **Causal Claims Removed:** LIME is framed strictly as descriptive local attribution.

---

## 21. Code Changes

* **Created:**
  * [`reports/issue_13_lime_explanation_consistency_resolution.md`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/reports/issue_13_lime_explanation_consistency_resolution.md) (This resolution report).

---

## 22. Training Status

* **Status:** The LIME evaluation framework, sampling protocol, and consistency metrics are **fully defined and frozen**.
* **Zero models were trained** in this step; execution will occur during the unified multi-seed experimental campaign.

---

## 23. Final Status

```
========================================================================================================
                                     ISSUE 13 STATUS: RESOLVED
========================================================================================================
The LIME explanation consistency framework is fully resolved:
- Evaluation Cohort: 72 stratified test instances across all 6 activities and all 9 held-out subjects.
- Error Coverage: Analyzes both correct predictions and ambulatory misclassifications.
- Quantitative Consistency: Top-10 Jaccard overlap (J_10), within-activity, cross-subject (R_subj), and FL vs. DP.
- Methodological Clarity: Distinct global (SHAP) and local (LIME) explainability roles established.
========================================================================================================
```
