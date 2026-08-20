# Issue 14 — Final Experimental Execution Matrix & Training Campaign Freeze

**Paper Title:** *Robust Human Activity Recognition through Federated Learning with Differential Privacy: A Comparison of Baseline and Centralized Models*  
**Venue:** Accepted for **ICI3T 2026** (Springer CCIS / LNCS Series)  
**Task:** Final Master Experimental Matrix, Protocol Freeze, and Pre-Training Verification  
**Date:** August 21, 2026  
**Status:** **ISSUE 14 STATUS: RESOLVED**

---

## 1. Final Datasets

Three Human Activity Recognition (HAR) datasets are included in the revision:

1. **UCI-HAR (Primary Benchmark):** 30 subjects, 10,299 samples, 561 engineered features, 6 activities.
2. **WISDM Smartphone & Smartwatch (Secondary Generalization):** 51 subjects, 18 activities (ambulatory, sedentary, and fine-motor gestures), smartphone & smartwatch accelerometer/gyroscope streams.
3. **HHAR Heterogeneity Benchmark (Device Diversity):** 9 users, 6 phone models, 2 smartwatch models, 6 activities, testing cross-device model robustness under client sampling.

---

## 2. Dataset Provenance and Citations

| Dataset | Canonical Source & DOI / URL | Key Citation | Primary Role in Paper |
| :--- | :--- | :--- | :--- |
| **UCI-HAR** | UCI Machine Learning Repository (`archive.ics.uci.edu/dataset/240`) | Anguita et al. (ESANN 2013) | **Primary Core Study:** FL vs. Centralized, Client-Level DP, XAI, Confusion Analysis |
| **WISDM** | IEEE Dataport / Fordham Laboratory (`doi:10.1109/ACCESS.2019.2940729`) | Weiss et al. (IEEE Access 2019) | **External Extension 1:** Multi-activity (18 classes) generalization across subjects |
| **HHAR** | ACM SenSys Archive (`doi:10.1145/2809695.2809718`) | Stisen et al. (ACM SenSys 2015) | **External Extension 2:** Hardware & sensor heterogeneity across 6 phone models |

---

## 3. Final Model Conditions (UCI-HAR Core)

```
                                  CORE 3-WAY CONTROLLED MATRIX
  ┌─────────────────────────────────┬──────────────────────────────────┬─────────────────────────────────┐
  │ Condition A: Centralized Non-DP │ Condition B: Federated Non-DP    │ Condition C: Federated DP-FL    │
  │ • Model: 3-Layer FNN            │ • Model: 3-Layer FNN             │ • Model: 3-Layer FNN            │
  │ • Architecture: 561-128-64-32-6 │ • Architecture: 561-128-64-32-6  │ • Architecture: 561-128-64-32-6 │
  │ • Parameters: 82,470            │ • Parameters: 82,470             │ • Parameters: 82,470            │
  │ • Budget: 200 Epochs            │ • Budget: 50 Rounds (E=10, K=10) │ • Budget: 50 Rounds (E=10, K=10)│
  │ • Privacy: None (Baseline)      │ • Privacy: None (FL Baseline)    │ • Privacy: Client DP (σ=1.00)   │
  └─────────────────────────────────┴──────────────────────────────────┴─────────────────────────────────┘
```

---

## 4. Frozen Training Configuration

* **Model Architecture:** 3-Layer Feed-Forward Neural Network (`Linear(561, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 6)`), 82,470 trainable parameters.
* **Optimizer:** Adam ($\eta = 0.001, \beta_1 = 0.9, \beta_2 = 0.999, \text{weight\_decay} = 1 \times 10^{-4}$).
* **Batch Size:** $B = 32$.
* **Local Epochs:** $E = 10$ local epochs per client per round.
* **Communication Rounds:** $R = 50$ global communication rounds.
* **Client Participation:** $K = 10$ clients sampled uniformly without replacement from $N = 21$ training clients per round ($q \approx 0.4762$).
* **Aggregation:**
  * Non-Private FL: Sample-weighted FedAvg over 80% local training data.
  * DP-FL: Unweighted FedAvg over clipped, noisy client parameter deltas.

---

## 5. Frozen Differential Privacy Configuration

* **Privacy Unit:** **Client-Level (Participant-Level) Differential Privacy**.
* **Global $L_2$ Delta Clipping:** $C = 1.0$ applied locally to client parameter updates $\Delta W_k$.
* **Noise Mechanism:** Local isotropic Gaussian noise injection $\Delta \widehat{W}_k = \Delta \widetilde{W}_k + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$.
* **Primary Noise Multiplier:** $\sigma = 1.00$ ($\sigma_{\text{global}} = \sigma \sqrt{10} \approx 3.1623$).
* **Target Delta ($\delta$):** $\delta = 1.0 \times 10^{-3}$ ($0.001$).
* **Accountant:** Analytical Subsampled Rényi Differential Privacy (RDP) Accountant.
* **Accounted Privacy Budget:** $\mathbf{\epsilon \approx 4.7861}$ ($\delta = 10^{-3}, R=50, q=10/21, C=1.0$).

---

## 6. Seed Policy

* **Predetermined Seeds:** **`42`**, **`123`**, and **`456`**.
* **Policy:** All primary metrics and tables report $\mathbf{\text{Mean} \pm \text{SD}}$ across the 3 independent runs. Zero seed cherry-picking permitted.

---

## 7. UCI-HAR Experiment Matrix

| Condition | Training Partition | Budget | Noise Multiplier $\sigma$ | Accounted $(\epsilon, \delta)$ | Seeds | Primary Outputs |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Centralized Baseline** | Pooled 21 Train (80%) | 200 Epochs | $0.00$ | Non-Private | `42, 123, 456` | Test Acc, F1, Loss, SHAP, LIME |
| **Federated No-DP** | 21 Clients (80% Train) | 50 Rounds | $0.00$ | Non-Private | `42, 123, 456` | Test Acc, F1, Client-Val, SHAP, LIME |
| **Federated Client-DP** | 21 Clients (80% Train) | 50 Rounds | $1.00$ | $\epsilon = 4.79, \delta = 10^{-3}$ | `42, 123, 456` | Test Acc, F1, Client-Val, SHAP, LIME |

---

## 8. WISDM Experiment Matrix

* **Representation:** 51 Subjects $\rightarrow$ 36 Training Clients / 15 Held-Out Test Subjects. Windowed statistical feature vectors (93 input features) mapped to 18 activity classes.
* **Model:** 3-Layer FNN (`93->128->64->32->18`).
* **Conditions:**
  1. Centralized Non-Private FNN (200 Epochs, Seeds `42, 123, 456`)
  2. Federated Non-Private FedAvg (50 Rounds, $K=12, E=10$, Seeds `42, 123, 456`)
  3. Federated Client-DP FedAvg ($\sigma=1.00$, Seeds `42, 123, 456`)

---

## 9. HHAR Experiment Matrix

* **Representation:** 9 Users $\rightarrow$ 6 Training Clients / 3 Held-Out Test Users across 6 device models. 2.56-sec window statistical features (60 input features, 6 activities).
* **Model:** 3-Layer FNN (`60->128->64->32->6`).
* **Conditions:**
  1. Centralized Non-Private FNN (200 Epochs, Seeds `42, 123, 456`)
  2. Federated Non-Private FedAvg (50 Rounds, $K=4, E=10$, Seeds `42, 123, 456`)
  3. Federated Client-DP FedAvg ($\sigma=1.00$, Seeds `42, 123, 456`)

---

## 10. Noise-Sensitivity Matrix (Pareto Curve)

Conducted on UCI-HAR across $R=50$ rounds and seeds `42, 123, 456`:

| Local $\sigma$ | Global $\sigma_{\text{global}}$ | Accounted $\epsilon$ ($\delta = 10^{-3}$) | Accounted $\epsilon$ ($\delta = 10^{-2}$) | Runs (Seeds) | Target Metric |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0.01** | 0.0316 | 49,932.71 | 49,930.41 | `42, 123, 456` | Test Accuracy, Macro-F1 |
| **0.05** | 0.1581 | 1,932.71 | 1,930.41 | `42, 123, 456` | Test Accuracy, Macro-F1 |
| **0.10** | 0.3162 | 432.72 | 430.42 | `42, 123, 456` | Test Accuracy, Macro-F1 |
| **0.20** | 0.6325 | 70.05 | 67.75 | `42, 123, 456` | Test Accuracy, Macro-F1 |
| **0.50** | 1.5811 | 12.19 | 9.89 | `42, 123, 456` | Test Accuracy, Macro-F1 |
| **1.00** | 3.1623 | **4.79** | **4.02** | `42, 123, 456` | Primary Model Evaluation |

---

## 11. Global Evaluation Matrix

* **Evaluation Cohort:** 9 Held-Out Test Subjects (`[2, 4, 9, 10, 12, 13, 18, 20, 24]`, 2,947 samples).
* **Computed Metrics:**
  * Overall Test Accuracy (\%)
  * Macro-Averaged F1-Score & Weighted-F1
  * Macro-Precision & Macro-Recall
  * Per-Class Precision, Recall, F1, and Support for all 6 classes
  * $6 \times 6$ Raw Integer & Row-Normalized Confusion Matrices
  * Ambulatory Diagnostic Metrics (Stairs vs. Walking)

---

## 12. Client-Level Evaluation Matrix

* **Evaluation Cohort:** 20% Local Validation Partitions of the 21 Training Clients (1,471 validation samples).
* **Computed Metrics:**
  * Macro-Client Mean Accuracy ($\mu_{\text{client}}$) $\pm$ Standard Deviation ($\text{SD}_{\text{client}}$)
  * Median Client Accuracy
  * Minimum Client Accuracy ($\min_k \text{Acc}_k$, worst-case participant)
  * Maximum Client Accuracy ($\max_k \text{Acc}_k$)
  * Per-Client DP Degradation Gap ($\Delta \text{Acc}_k = \text{Acc}_{\text{DP}, k} - \text{Acc}_{\text{NonDP}, k}$)

---

## 13. Convergence Experiment

* **Tracking:** Validation Loss and Accuracy logged at every round $r \in \{1, \dots, 50\}$ over the pooled client validation splits.
* **Leakage Prevention:** Test set is evaluated strictly once on the final Round 50 model.
* **Output:** 50-round convergence trajectories with $\text{Mean} \pm \text{SD}$ ribbon error bands.

---

## 14. SHAP Experiment

* **Cohort:** 120 stratified held-out test instances (20/class across 6 activities) under fixed seed `42`.
* **Conditions:** Centralized Non-Private vs. Federated Non-Private vs. Federated DP-FL.
* **Metrics:** Mean absolute SHAP importance ($I_f$), Top-20 ranking, Spearman rank correlation ($\rho_s$), and Top-20 Jaccard overlap ($J_{20}$).

---

## 15. LIME Experiment

* **Cohort:** 72 stratified held-out test instances (8 per subject across all 9 held-out subjects, 12 per activity).
* **Diagnostics:** Evaluates both correct predictions and ambulatory misclassifications (`WALKING_UPSTAIRS` $\rightarrow$ `WALKING`).
* **Metrics:** Top-10 Jaccard similarity ($J_{10}$), Within-Activity Consistency ($\overline{J}_{\text{act}}$), and Inter-Subject Robustness Ratio ($R_{\text{subj}}$).

---

## 16. Exact Experiment Count

```
========================================================================================================
                                     TOTAL EXPERIMENTAL RUNS
========================================================================================================
1. UCI-HAR Core Campaign:
   - Centralized Baseline:          3 Seeds x 1 Condition   =  3 runs
   - Federated Non-Private:         3 Seeds x 1 Condition   =  3 runs
   - Federated DP Noise Sweep:      3 Seeds x 6 Noise Levels = 18 runs
   ----------------------------------------------------------------
   UCI-HAR Subtotal:                                         24 runs

2. WISDM Multi-Activity Campaign:
   - Centralized Baseline:          3 Seeds x 1 Condition   =  3 runs
   - Federated Non-Private:         3 Seeds x 1 Condition   =  3 runs
   - Federated Client-DP (σ=1.00):  3 Seeds x 1 Condition   =  3 runs
   ----------------------------------------------------------------
   WISDM Subtotal:                                            9 runs

3. HHAR Device Heterogeneity Campaign:
   - Centralized Baseline:          3 Seeds x 1 Condition   =  3 runs
   - Federated Non-Private:         3 Seeds x 1 Condition   =  3 runs
   - Federated Client-DP (σ=1.00):  3 Seeds x 1 Condition   =  3 runs
   ----------------------------------------------------------------
   HHAR Subtotal:                                             9 runs
========================================================================================================
GRAND TOTAL:                                                 42 INDEPENDENT TRAINING RUNS
========================================================================================================
```

---

## 17. Computational Budget

* **Total FL Communication Rounds:** $(3 + 18 + 3 + 3 + 3 + 3) \times 50 = \mathbf{1,650\text{ rounds}}$.
* **Total Client Local Epochs:** $1,650 \text{ rounds} \times 10 \text{ clients} \times 10 \text{ epochs} = \mathbf{165,000\text{ client-epochs}}$.
* **Estimated Execution Time:** ~15–20 minutes on modern multi-core CPU / Apple Silicon MPS GPU.

---

## 18. Output Directory Structure

```
FINAL_FEDERATED_LEARNING/results/experiments/
├── uci_har/
│   ├── centralized/        (seed_42, seed_123, seed_456: metrics.json, manifest.json, history.csv)
│   ├── federated_nodp/     (seed_42, seed_123, seed_456)
│   ├── federated_dp/       (sigma_0.01 .. sigma_1.00 x 3 seeds)
│   ├── shap/               (shap_values.npy, feature_rankings.json)
│   ├── lime/               (lime_explanations.json, consistency_metrics.json)
│   └── client_analysis/    (client_metrics.json, heterogeneity_report.json)
├── wisdm/
│   ├── centralized/
│   ├── federated_nodp/
│   └── federated_dp/
├── hhar/
│   ├── centralized/
│   ├── federated_nodp/
│   └── federated_dp/
└── summary_tables/
    ├── table1_uci_har_results.csv
    ├── table2_noise_sensitivity.csv
    ├── table3_client_robustness.csv
    └── table4_external_datasets.csv
```

---

## 19. Experiment Manifest Schema

Every run directory will contain an immutable, machine-readable `manifest.json` recording the exact hyperparameter state, seed, git commit SHA, and accounted $(\epsilon, \delta)$.

---

## 20. Paper Tables Mapping

* **Table 1 (Core UCI-HAR Benchmark):** Centralized vs. Federated vs. DP-FL (Accuracy, Macro-F1, Weighted-F1, Macro-P, Macro-R, $\text{Mean} \pm \text{SD}$).
* **Table 2 (Differential Privacy Pareto Sweep):** $\sigma \in \{0.01, \dots, 1.00\}$ vs. Accounted $\epsilon$ vs. Empirical Accuracy & F1.
* **Table 3 (Client-Level Robustness & Heterogeneity):** Macro-Client Mean $\pm$ SD, Median, Min, Max, and $\Delta \text{Acc}_k$.
* **Table 4 (Cross-Dataset Generalization):** UCI-HAR vs. WISDM vs. HHAR benchmark comparison.
* **Table 5 (LIME Explanation Consistency):** Within-activity $J_{10}$, cross-subject $R_{\text{subj}}$, and error attribution shift.

---

## 21. Paper Figures Mapping

* **Figure 1 (Architecture):** Client-Level Differentially Private Federated Learning schematic (with *"DP Parameter Deltas"*).
* **Figure 5 (Noise-Utility Pareto Curve):** Empirical Test Accuracy vs. RDP-computed $\epsilon$ with $\text{Mean} \pm \text{SD}$ shaded bands.
* **Figure 6 (Convergence Ribbons):** 50-round validation loss & accuracy trajectories (Centralized vs. FL vs. DP-FL).
* **Figure 7 (Method Comparison):** Multi-seed bar chart of Accuracy and Macro-F1.
* **Figure 8 (Confusion Matrix Heatmap):** $6 \times 6$ normalized confusion matrix with ambulatory breakdown.
* **Figure 9 (SHAP Feature Importance):** Top-20 global feature attribution across Centralized, FL, and DP-FL.
* **Figure 10 (LIME Consistency & Diagnostics):** Cross-subject consistency boxplots and stair-climbing error case study.
* **Figure 11 (Client Performance Distribution):** Per-client validation accuracy boxplots across all 21 clients.

---

## 22. Reviewer Requirement Coverage Matrix

| Reviewer Item | Requirement Summary | Addressed By Experiment / Section |
| :--- | :--- | :--- |
| **Reviewer 1 #1** | Subject-disjoint partition & no train/test leakage | Issue #3: 21 Train Clients / 9 Held-Out Test Subjects |
| **Reviewer 1 #2** | 561 vs. 20 feature input dimension | Issue #1: 561 engineered input dimension locked |
| **Reviewer 1 #3** | FNN vs. LSTM-CNN temporal sequence | Issue #2: 3-Layer FNN locked (no artificial temporal ordering) |
| **Reviewer 1 #4** | DP reproducibility parameters & accountant | Issue #6 & #7: $C=1.0, B=32, q=10/21, R=50, \delta=10^{-3}$, RDP Accountant |
| **Reviewer 1 #5** | Reconcile $\sigma=0.1/\epsilon=1.0$ vs $\sigma=0.01/\epsilon\approx 100$ | Issue #7: Excised legacy claims; exact RDP sweep table |
| **Reviewer 1 #6** | Define privacy unit (Client vs. Record DP) | Issue #6: Client-Level (Participant-Level) DP locked |
| **Reviewer 1 #7** | Controlled comparison & training budget | Issue #4 & #5: 200 Centralized Epochs $\approx$ 50 FL Rounds |
| **Reviewer 1 #8** | Remove unsupported 94.5% claim | Issue #4: Excised from conclusion and Fig 7 |
| **Reviewer 1 #9** | Centralized FNN non-private baseline | Issue #4: Symmetric 82,470-parameter FNN baseline |
| **Reviewer 1 #10** | Ambulatory confusion, Macro-F1, Precision/Recall | Issue #10: Pure-Python metrics suite + $6 \times 6$ confusion matrix |
| **Reviewer 1 #11** | Repeated runs, seeds & variability | Issue #9: Seeds `42, 123, 456`, $\text{Mean} \pm \text{SD}$ |
| **Reviewer 1 #12** | Client-level performance variation & non-IID | Issue #11: 21 local validation evaluations, TVD audit |
| **Reviewer 1 #13** | SHAP controlled ablations & causal claims | Issue #12: Centralized vs. FL vs. DP-FL on fixed 120 samples |
| **Reviewer 1 #14** | LIME multi-subject, multi-activity consistency | Issue #13: 72 stratified instances across 9 subjects, $J_{10}, R_{\text{subj}}$ |
| **Reviewer 1 #15** | Clarify encryption vs. DP vs. SecAgg | Issue #8: Clarified Option A (FL + DP); removed "encrypted" |
| **Reviewer 2 #3** | Graphical comparative study | Issues #9, #10, #11, #12: Figures 5, 6, 7, 8, 9, 10, 11 |
| **Reviewer 2 #6** | Add more results and discussion | Issues #10, #11, #14: WISDM & HHAR external datasets |

---

## 23. Claim-to-Evidence Safety Mapping

* **Claim 1:** *"Federated learning preserves data sovereignty by training locally on user devices."* $\rightarrow$ **Evidence:** 21 independent client data loaders with strictly local gradient steps.
* **Claim 2:** *"The DP mechanism guarantees client-level differential privacy."* $\rightarrow$ **Evidence:** Parameter-delta $L_2$ clipping ($C=1.0$) + Gaussian noise analyzed via analytical RDP accountant ($\epsilon \approx 4.79, \delta=10^{-3}$).
* **Claim 3:** *"Federated models generalize to unseen human subjects."* $\rightarrow$ **Evidence:** Empirical accuracy on 9 completely held-out test subjects.
* **Claim 4:** *"Model performance is robust across heterogeneous participants."* $\rightarrow$ **Evidence:** Macro-Client Mean $\pm$ SD and minimum client validation statistics across all 21 clients.

---

## 24. Pre-Training Validation Checklist

* [x] UCI-HAR, WISDM, and HHAR directories exist and are verified.
* [x] 21 Training Clients and 9 Held-Out Test Subjects share 0.0% overlap.
* [x] Model architecture has exactly 82,470 trainable parameters.
* [x] Hyperparameters frozen: Adam ($\eta=10^{-3}, B=32, E=10, R=50, K=10/21$).
* [x] DP parameters frozen: $C=1.0, \sigma=1.00, \delta=10^{-3}, \text{Accountant}=\text{RDP}$.
* [x] Seeds frozen: `42, 123, 456`.
* [x] Output directories initialized.
* [x] [`validate_pre_training.py`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/scripts/validate_pre_training.py) passes with 100% success.

---

## 25. Git Checkpoint

* **Remote URL:** `https://github.com/DreamX55/FederatedLearning_Project.git`
* **Branch:** `main`
* **Pre-Experiment Checkpoint Commit:** `e137e35` (synchronizing Issues 1–5; updated through Issue 14).

---

## 26. Final Protocol Invariants

```
========================================================================================================
                                     FROZEN PROTOCOL INVARIANTS
========================================================================================================
- Dataset:              UCI-HAR (561 input features), WISDM (93 features), HHAR (60 features)
- Subject Partition:    21 Training Clients / 9 Held-Out Test Subjects (UCI-HAR)
- Primary Model:        3-Layer FNN (561-128-64-32-6, 82,470 parameters)
- Optimizer:            Adam (lr=0.001, beta1=0.9, beta2=0.999, weight_decay=1e-4)
- Optimization Budget:  Centralized = 200 Epochs; Federated = 50 Rounds (K=10, E=10, B=32)
- Privacy Mechanism:    Client-Level DP (C=1.0, sigma=1.00, delta=1e-3, epsilon ≈ 4.79)
- Seeds:                42, 123, 456 (Mean ± SD reporting)
- Total Campaign Runs:  42 Independent Training Runs
========================================================================================================
```

---

## 27. Training Status

```
========================================================================================================
                              NO FINAL TRAINING CAMPAIGN HAS BEEN EXECUTED
========================================================================================================
All experimental conditions, matrices, manifests, and validation gates are frozen and verified.
Execution will commence systematically in the next phase.
========================================================================================================
```

---

## 28. Final Status

```
========================================================================================================
                                     ISSUE 14 STATUS: RESOLVED
========================================================================================================
The master experimental execution matrix, protocol invariants, and pre-training gates are 100% frozen.
========================================================================================================
```
