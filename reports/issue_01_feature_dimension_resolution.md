# Issue 01 — Feature Dimension Resolution

**Paper Title:** *Robust Human Activity Recognition through Federated Learning with Differential Privacy: A Comparison of Baseline and Centralized Models*  
**Venue:** Accepted for **ICI3T 2026** (Springer CCIS / LNCS Series)  
**Issue:** Reviewer 1 #2 (Resolution of 561 vs. 20 Feature Input Dimensionality)  
**Date:** August 20, 2026  
**Status:** **ISSUE 1 STATUS: RESOLVED**

---

## 1. Verified Implementation Reality

A comprehensive, line-by-line inspection of all dataset loaders, preprocessing pipelines, model architectures, configuration schemas, and data validation scripts confirmed the following empirical facts:

1. **True Model Input Dimensionality:**
   * The actual UCI-HAR training, federated aggregation, and evaluation pipelines operate exclusively on the full **561-dimensional** hand-crafted feature vectors.
   * **Source Evidence:**
     * [`FINAL_FEDERATED_LEARNING/src/models/fnn.py:L14`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/src/models/fnn.py#L14): `def __init__(self, input_dim=561, hidden1=128, hidden2=64, hidden3=32, output_dim=6, dropout_p=0.3):`
     * [`FINAL_FEDERATED_LEARNING/src/config/training_config.py:L7`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/src/config/training_config.py#L7): `'input_dim': 561`
     * [`FINAL_FEDERATED_LEARNING/src/config/default.yaml:L5`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/src/config/default.yaml#L5): `input_dim: 561`
     * [`FINAL_FEDERATED_LEARNING/scripts/data_validation.py:L48`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/scripts/data_validation.py#L48): `if X.shape[1] != 561: print(f"WARNING: Expected 561 features, got {X.shape[1]}")`

2. **Absence of Variance-Based Feature Selection:**
   * There is **no code, script, function, or configuration** in the repository that implements variance thresholding, feature filtering, or dimensionality reduction from 561 features down to 20 features.
   * [`FINAL_FEDERATED_LEARNING/src/datasets/har_dataset.py`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/src/datasets/har_dataset.py) loads `X_train.txt` (561 columns) and `X_test.txt` (561 columns) directly, saving client arrays `client_{cid}_X.npy` with full shape `(num_samples, 561)`.
   * All models (Centralized FNN, FL without DP, FL with DP) were trained and evaluated on all 561 features.

---

## 2. 20-Feature Claim Audit

A repository-wide search identified all manuscript, draft, and documentation occurrences claiming that variance analysis selected 20 features:

| File Location | Section / Line | Current Statement in Document | Audit Finding & Required Correction |
| :--- | :--- | :--- | :--- |
| [`SPRINGER_LATEX/sections/02_dataset.tex`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/02_dataset.tex#L24-L27) | Section 2 (`Dataset`), Lines 24–27 | *"Correlation analysis of the first 30 features shows organized relationships between the sensor measurements, with variance analysis selecting the most informative 20 features (variance range: 0.090-0.142), with enough variability for successful learning. The preprocessing workflow ensures data integrity by systematic validation, load-balanced client distribution, and variance optimization of features, providing high-quality input for federated learning experiments."* | **CONTRADICTS IMPLEMENTATION:** The 20-feature reduction was never executed in code. In the upcoming manuscript update, this sentence must be removed or replaced with an explicit statement confirming that all 561 statistical features are preserved to maximize cross-activity discriminability. |
| [`PAPER_REFERENCE/Federated_Learning_Differential_Privacy_HAR_LNCS.docx`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/PAPER_REFERENCE/Federated_Learning_Differential_Privacy_HAR_LNCS.docx) | Section 2 (`Dataset`), Paragraph 21 | Same text as above (legacy source of the LaTeX draft text). | Historical reference artifact; identified as origin of the unexecuted 20-feature claim. |

*Note on other "20" occurrences in manuscript:* Other mentions of the number 20 in the paper refer to:
- 20 federated communication rounds (e.g., in Section 4.2.1 and Table 1 legend).
- 20 clients sampled per round (e.g., in Section 4.2.1).
These are hyperparameter settings and do not refer to input feature dimensions.

---

## 3. Scaling Audit (Flagged for Issue #3)

The forensic audit revealed that the current scaling routine suffers from **global normalization leakage**:

### Exact Location of Current Global Normalization:
* **File:** [`FINAL_FEDERATED_LEARNING/src/datasets/har_dataset.py`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/src/datasets/har_dataset.py#L46-L60)
* **Code:**
  ```python
  # Lines 46-48: Pooling train and test sets
  X = np.vstack((X_train, X_test))
  y = np.concatenate((y_train, y_test))

  # Lines 52-60: Global Min-Max computation over pooled data
  def normalize_data(X):
      min_vals = X.min(axis=0)
      max_vals = X.max(axis=0)
      X_norm = (X - min_vals) / (max_vals - min_vals + 1e-8)
      X_norm = np.clip(X_norm, 0, 1)
      return X_norm
  ```

### Why this is flagged:
Computing `min_vals` and `max_vals` across all 10,299 samples (train + test combined) before client partitioning leaks test set statistical distributions into training client features.

### Scope Boundary:
* As mandated by the single-issue focus, **this scaling leakage is NOT modified in Issue #1**.
* This scaling procedure is explicitly flagged and will be corrected in **Issue #3 (Leakage-Free Partitioning & Normalization Protocol)**, where scalers will be fitted exclusively on training client data.

---

## 4. Final Decision

1. **UCI-HAR Input Dimension:** **561 features** (Frozen).
2. **Variance-Based 20-Feature Selection:** **NOT USED / DISCARDED** (Frozen).
3. **WISDM Input Dimension:** **NOT YET FROZEN** (Deferred to Issue #2).
4. **HHAR Input Dimension:** **NOT YET FROZEN** (Deferred to Issue #2).
5. **Scaling Scope:** **NOT YET FROZEN** (Deferred to Issue #3).

---

## 5. Files Changed

* **Created:**
  * [`reports/issue_01_feature_dimension_resolution.md`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/reports/issue_01_feature_dimension_resolution.md) (This audit report).

---

## 6. Files Deliberately NOT Changed

To preserve strict single-issue isolation and adhere to the pre-implementation protocol freeze:
* **Model architecture files NOT changed:** [`FINAL_FEDERATED_LEARNING/src/models/fnn.py`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/src/models/fnn.py) (retained as 561-dim input, 82,470 params).
* **FL client partitioning code NOT changed:** [`FINAL_FEDERATED_LEARNING/src/datasets/har_dataset.py`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/src/datasets/har_dataset.py) (deferred to Issue #3).
* **DP implementation NOT changed:** [`FINAL_FEDERATED_LEARNING/src/privacy/differential_privacy.py`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/src/privacy/differential_privacy.py).
* **WISDM / HHAR preprocessing NOT changed:** Preserved as raw extractions.
* **No training performed:** Zero model training or evaluation was executed.

---

## 7. Status

```
========================================================================================================
                                     ISSUE 1 STATUS: RESOLVED
========================================================================================================
The 561 vs. 20 feature input dimension ambiguity raised by Reviewer 1 (#2) is fully resolved:
- The actual model input dimensionality for UCI-HAR is confirmed to be 561 features.
- The 20-feature claim in the manuscript is documented as an erroneous draft statement to be corrected.
- Scaling leakage is documented and queued for resolution in Issue #3.
========================================================================================================
```
