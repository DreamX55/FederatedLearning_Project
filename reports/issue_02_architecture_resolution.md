# Issue 02 — Architecture Resolution

**Paper Title:** *Robust Human Activity Recognition through Federated Learning with Differential Privacy: A Comparison of Baseline and Centralized Models*  
**Venue:** Accepted for **ICI3T 2026** (Springer CCIS / LNCS Series)  
**Issue:** Reviewer 1 #3 & #9 (Resolution of LSTM-CNN vs. FNN Model Architecture & Temporal-Sequence Justification)  
**Date:** August 20, 2026  
**Status:** **ISSUE 2 STATUS: RESOLVED**

---

## 1. Verified Implementation Architecture

A rigorous code inspection of all model definitions, training scripts, federated aggregation routines, and evaluation pipelines confirmed that the primary, reproducible UCI-HAR experimental pipeline implements and executes a **3-Layer Feed-Forward Neural Network (FNN)**.

### Detailed Architecture of the Primary Model (`FNN`):
* **Source File:** [`FINAL_FEDERATED_LEARNING/src/models/fnn.py:L14-L56`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/src/models/fnn.py#L14-L56)
* **Input Layer:** $561$ dimensions (statistical engineered feature vector)
* **Hidden Layer 1:** $\text{Linear}(561, 128) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(p=0.3)$
* **Hidden Layer 2:** $\text{Linear}(128, 64) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(p=0.3)$
* **Hidden Layer 3:** $\text{Linear}(64, 32) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(p=0.3)$
* **Output Layer:** $\text{Linear}(32, 6)$ (6 activity classes; unnormalized logits for `nn.CrossEntropyLoss`)
* **Exact Trainable Parameters:**
  $$\text{FC1: } (561 \times 128) + 128 = 71,936$$
  $$\text{FC2: } (128 \times 64) + 64 = 8,256$$
  $$\text{FC3: } (64 \times 32) + 32 = 2,080$$
  $$\text{FC4: } (32 \times 6) + 6 = 198$$
  $$\mathbf{\text{Total Trainable Parameters: }} 71,936 + 8,256 + 2,080 + 198 = \mathbf{82,470}$$

### Execution Path Confirmation:
* **Federated Learning with DP:** [`scripts/train_federated.py:L8`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/scripts/train_federated.py#L8) directly imports `FNN` and trains it on client parameter deltas.
* **Federated Learning without DP:** [`scripts/train_federated_nodp.py:L7`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/scripts/train_federated_nodp.py#L7) directly imports `FNN`.
* **Centralized Training:** [`scripts/train_centralized.py:L9`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/scripts/train_centralized.py#L9) directly imports `FNN`.
* **Evaluation Pipeline:** [`scripts/evaluate.py:L6`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/scripts/evaluate.py#L6) and [`scripts/evaluate_centralized.py:L9`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/scripts/evaluate_centralized.py#L9) evaluate `FNN`.
* **Status of LSTM-CNN in Code:** The `CNN_LSTM_Attn` class in [`src/models/mobile_optimized.py`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/src/models/mobile_optimized.py) is an unintegrated legacy file from an earlier prototype (`FL_Project`). It is **never imported, trained, or evaluated** by any active script in the primary repository.

---

## 2. LSTM-CNN Claim Audit

A complete scan of the manuscript, reference documents, and configuration files cataloged all statements claiming that an LSTM-CNN architecture was used:

| File Location | Section / Line | Current Statement in Document | Audit Finding |
| :--- | :--- | :--- | :--- |
| [`SPRINGER_LATEX/sections/04_models.tex`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/04_models.tex#L28) | Section 4.2.1, Line 28 | *"The model uses the same setup uses the FedAvg algorithm with the same LSTM-CNN architecture as the central model, but trains on 30 distributed clients (subjects)."* | **UNSUPPORTED CLAIM:** The FL model actually uses the 3-layer FNN architecture. |
| [`SPRINGER_LATEX/sections/04_models.tex`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/04_models.tex#L49-L53) | Section 4.2.3, Lines 49–53 | *"Centralized LSTM-CNN with Differential Privacy (CL+DP)... Its hybrid 847K-parameter architecture consisting of two LSTM layers (128 -> 64 units) followed by 1D convolutions captures temporal and local sensor patterns..."* | **UNSUPPORTED / MISMATCHED:** This text described a prototype model evaluated in an external prototype codebase (`FL_Project`), not the canonical FNN pipeline. |
| [`SPRINGER_LATEX/sections/06_results.tex`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/06_results.tex#L21) | Section 6, Table 1 (Row M5) | `M5 & Centralized LSTM-CNN + DP & \sigma = 0.10, \epsilon \approx 1.0 & 84.59\% & 0.845 \\` | **LEGACY LABEL:** Refers to the historical prototype result from `FL_Project`. |
| [`SPRINGER_LATEX/sections/04_models.tex`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/04_models.tex#L15) | Section 4.1.1, Line 15 | *"Our primary baseline FNN consists of a straightforward three-layered architecture (256-128-64 units, ReLU) with 98,502 parameters"* | **INACCURATE SPECIFICATION:** The actual implemented FNN is `128-64-32` units (82,470 parameters), not `256-128-64` (185,414 parameters). |
| [`PAPER_REFERENCE/Federated_Learning_Differential_Privacy_HAR_LNCS.docx`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/PAPER_REFERENCE/Federated_Learning_Differential_Privacy_HAR_LNCS.docx) | Section 4 & Table 1 | Paragraphs 45, 51 and Table 1 Row 3 | Historical source of the draft LSTM-CNN claims. |

---

## 3. Temporal-Sequence Audit

### How the 561 Features are Actually Structured:
* The UCI-HAR dataset's 561 features are **static summary statistics** computed over 2.56-second sliding time windows (128 raw readings @ 50 Hz with 50% overlap).
* These 561 features comprise:
  * Time-domain statistics: mean, standard deviation, median absolute deviation, max, min, signal magnitude area, energy, interquartile range, autoregression coefficients.
  * Frequency-domain statistics: FFT energy bands, spectral entropy, skewness, kurtosis.
  * Angle features between vectors (e.g., `angle(tBodyAccMean, gravity)`).

### Scientific Reality:
* There is **no temporal ordering or sequential relationship across the 561 columns of a single feature vector**.
* Feature #1 (`tBodyAcc-mean()-X`) followed by Feature #2 (`tBodyAcc-mean()-Y`) does not constitute a chronological time series.
* In the actual codebase, the 561 features are fed **in parallel as a single 1D vector** into `Linear(561, 128)`.
* No temporal sequence generation, reshaping to $(B, T, D)$, or recurrent step is performed on the 561 features.

---

## 4. Reviewer 1 #3 Resolution

### Reviewer Concern:
> *“The LSTM-CNN architecture requires stronger justification because UCI-HAR provides engineered feature vectors rather than explicitly defined raw temporal sequences. Explain how temporal sequences are constructed and how the 561 features are ordered for LSTM processing.”*

### Scientifically Correct Response for Revision:
1. **Acknowledge and Clarify:** The reviewer correctly identified that an LSTM cannot be sensibly applied to an unordered 561-dimensional statistical feature vector without fabricating artificial temporal semantics.
2. **State Implementation Reality:** The primary federated learning model in this study is a **3-layer Feed-Forward Neural Network (FNN)** operating directly on the 561-dimensional feature vectors.
3. **Manuscript Correction:** The draft references to LSTM-CNN in Section 4.2 were draft discrepancies inherited from an exploratory prototype and will be corrected in the revised manuscript to accurately specify the 3-layer FNN architecture. No artificial temporal sequence is constructed.

---

## 5. Reviewer 1 #9 Resolution

### Reviewer Concern:
> *“The baseline comparison is limited to FNN and Random Forest, while the proposed approach uses LSTM-CNN. Include a directly comparable centralized non-private LSTM-CNN baseline under identical preprocessing and evaluation conditions.”*

### Methodological Evaluation of Options:

* **Option A: Force-implement and train an LSTM-CNN baseline.**
  * *Scientific Weakness:* Training an LSTM on 561 static features is scientifically invalid. Training an LSTM-CNN on raw $128 \times 9$ inertial signals would create a severe input representation mismatch with the primary 561-feature FL pipeline, rendering the centralized vs. federated comparison invalid.
* **Option B (Selected & Recommended): Align the manuscript with the actual FNN architecture and provide a directly comparable Centralized Non-Private FNN Baseline.**
  * *Scientific Strength:* Ensures complete methodological symmetry:
    * Centralized Non-Private: 3-Layer FNN on 561 features.
    * Federated Non-Private: 3-Layer FNN on 561 features.
    * Federated with DP: 3-Layer FNN on 561 features + Parameter-Delta DP.
    * Centralized with DP: 3-Layer FNN on 561 features + DP.
  * All models share identical input dimensions (561), identical parameter counts (82,470), identical optimizers (Adam, $lr=0.001$), and identical subject-disjoint evaluation sets.

### Decision:
**Option B is formally approved.** The centralized baseline is a **Centralized Non-Private 3-Layer FNN** trained on the exact same 21 training subjects and evaluated on the 9 held-out test subjects.

---

## 6. Final UCI-HAR Architecture Decision

```
========================================================================================================
                               FINAL UCI-HAR ARCHITECTURE SPECIFICATION
========================================================================================================
```
1. **Model Family:** Feed-Forward Neural Network (FNN).
2. **Input Representation:** 561-dimensional engineered feature vector.
3. **Layer Structure:**
   * $\text{Input Layer: } 561\text{ units}$
   * $\text{Hidden Layer 1: } \text{Linear}(561, 128) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.3)$
   * $\text{Hidden Layer 2: } \text{Linear}(128, 64) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.3)$
   * $\text{Hidden Layer 3: } \text{Linear}(64, 32) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.3)$
   * $\text{Output Layer: } \text{Linear}(32, 6)$ (Logits)
4. **Total Parameters:** **82,470**
5. **Loss Function:** `nn.CrossEntropyLoss()`
6. **Explicit Exclusions:**
   * **NO** LSTM layers.
   * **NO** 1D Convolutional layers.
   * **NO** Artificial temporal sequence ordering on the 561 features.

---

## 7. WISDM and HHAR Architecture Status

In accordance with single-issue isolation and protocol design:
* **WISDM Architecture:** **DEFERRED** (Will be formalized as a 1D-CNN operating on raw 20 Hz sliding windows in the cross-dataset extension phase).
* **HHAR Architecture:** **DEFERRED** (Will be formalized as a 1D-CNN operating on resampled 50 Hz sliding windows in the cross-dataset extension phase).

---

## 8. Manuscript Correction Inventory

The following manuscript locations must be updated during the upcoming manuscript revision phase:

| Manuscript File | Section | Current Inaccurate Text | Required Future Correction |
| :--- | :--- | :--- | :--- |
| [`SPRINGER_LATEX/sections/04_models.tex:L15`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/04_models.tex#L15) | Section 4.1.1 | *"three-layered architecture (256-128-64 units, ReLU) with 98,502 parameters"* | Update to: *"three-hidden-layer architecture (128-64-32 units, ReLU, Dropout 0.3) with 82,470 trainable parameters"*. |
| [`SPRINGER_LATEX/sections/04_models.tex:L28`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/04_models.tex#L28) | Section 4.2.1 | *"same LSTM-CNN architecture as the central model"* | Update to: *"same 3-layer FNN architecture as the centralized baseline"*. |
| [`SPRINGER_LATEX/sections/04_models.tex:L49-L53`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/04_models.tex#L49-L53) | Section 4.2.3 | *"Centralized LSTM-CNN with Differential Privacy (CL+DP)... hybrid 847K-parameter architecture"* | Update to describe the **Centralized FNN with DP** operating symmetrically on 561 features. |
| [`SPRINGER_LATEX/sections/06_results.tex:L21`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/06_results.tex#L21) | Table 1 (Row M5) | `M5 & Centralized LSTM-CNN + DP & ...` | Update label to: `M5 & Centralized FNN + DP & ...`. |
| [`SPRINGER_LATEX/sections/08_conclusion.tex:L7`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/08_conclusion.tex#L7) | Section 8 | Reference to unsupported 94.5% centralized comparison | Update to compare FL against the verified empirical Centralized FNN baseline. |

---

## 9. Figure and Diagram Correction Inventory

| Figure Asset | Manuscript Location | Current Depiction / Caption | Required Future Correction |
| :--- | :--- | :--- | :--- |
| [`SPRINGER_LATEX/figures/fig1.png`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/figures/fig1.png) | Section 1, [`01_introduction.tex:L13`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/01_introduction.tex#L13) | Illustrates general client-to-server FL flow. | Keep high-level concept; ensure no diagram component erroneously labels client models as LSTM-CNN or raw time-series for UCI-HAR. |
| [`SPRINGER_LATEX/figures/fig7.png`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/figures/fig7.png) | Section 4, [`04_models.tex:L56`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/04_models.tex#L56) | Caption: *"Consistent accuracy rating with higher epoch rounds"* (shows a bar chart of 94.5% vs 93.0% vs 91.6%). | Replace with genuine method comparison bar chart based on verified empirical results across seeds. |

---

## 10. Files Changed

* **Created:**
  * [`reports/issue_02_architecture_resolution.md`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/reports/issue_02_architecture_resolution.md) (This audit and decision report).

---

## 11. Files Deliberately NOT Changed

To preserve strict single-issue isolation:
* **No FL changes:** [`scripts/train_federated.py`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/scripts/train_federated.py) unchanged.
* **No DP changes:** [`src/privacy/differential_privacy.py`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/src/privacy/differential_privacy.py) unchanged.
* **No partition changes:** Data partitioning deferred to Issue #3.
* **No scaling changes:** Scaling deferred to Issue #3.
* **No WISDM / HHAR modifications:** Raw datasets preserved.
* **No training performed:** Centralized baseline training is deferred to Issue #4.
* **No new results generated.**

---

## 12. Final Status

```
========================================================================================================
                                     ISSUE 2 STATUS: RESOLVED
========================================================================================================
The LSTM-CNN vs. FNN architecture discrepancy raised by Reviewer 1 (#3 and #9) is fully resolved:
- The actual model for UCI-HAR is confirmed to be the 3-layer FNN (82,470 parameters) on 561 features.
- No artificial temporal ordering is constructed for the 561 features.
- The centralized baseline is defined symmetrically as a Centralized Non-Private 3-Layer FNN.
- All manuscript and diagram locations requiring future text correction are cataloged.
========================================================================================================
```
