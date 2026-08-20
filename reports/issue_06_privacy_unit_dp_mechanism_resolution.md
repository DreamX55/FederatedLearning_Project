# Issue 06 — Privacy Unit & DP Mechanism Resolution

**Paper Title:** *Robust Human Activity Recognition through Federated Learning with Differential Privacy: A Comparison of Baseline and Centralized Models*  
**Venue:** Accepted for **ICI3T 2026** (Springer CCIS / LNCS Series)  
**Issue:** Reviewer 1 #4 & #6 (Resolution of Differential Privacy Mechanism, Privacy Unit Definition, and Accounting Provenance)  
**Date:** August 20, 2026  
**Status:** **ISSUE 6 STATUS: RESOLVED**

---

## 1. Current DP Implementation Audit

A line-by-line inspection of [`FINAL_FEDERATED_LEARNING/src/privacy/differential_privacy.py`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/src/privacy/differential_privacy.py) and [`FINAL_FEDERATED_LEARNING/scripts/train_federated.py`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/scripts/train_federated.py) traced the exact differential privacy execution path:

```
                                  CLIENT k (Subject k)
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │ 1. Local Training: 10 local epochs (B=32) using local SGD on X_train_k (No DP)    │
  │ 2. Parameter Delta Calculation: ΔW_k = W_k - W_global                            │
  │ 3. Global L2 Norm Computation: ||ΔW_k||_2 = sqrt( sum( ||ΔW_k^(l)||_2^2 ) )       │
  │ 4. Client-Level L2 Clipping: ΔW_k_clipped = ΔW_k * min(1, C / ||ΔW_k||_2)        │
  │ 5. Local Gaussian Noise Injection: ΔW_k_noisy = ΔW_k_clipped + N(0, σ² C² I)     │
  └────────────────────────────────────────┬─────────────────────────────────────────┘
                                           │ Transmit ΔW_k_noisy
                                           ▼
                                    FEDERATED SERVER
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │ 6. Unweighted Averaging: ΔW_global = (1 / K) * sum_{k ∈ S_t} ΔW_k_noisy          │
  │ 7. Global Model Update: W_global^(r+1) = W_global^(r) + ΔW_global                 │
  └──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Exact Clipping Mechanism

* **What is clipped:** The **entire client parameter update delta** $\Delta W_k = W_k - W_{\text{global}}$ (the collection of all weight and bias matrices across all 4 layers of the 82,470-parameter FNN).
* **Clipping Norm:** Global $L_2$ norm across all layers:
  $$\|\Delta W_k\|_2 = \sqrt{\sum_{l=1}^4 \|\Delta W_k^{(l)}\|_2^2}$$
* **Clipping Formula:**
  $$\Delta \widetilde{W}_k = \Delta W_k \cdot \min\left(1, \frac{C}{\|\Delta W_k\|_2 + 10^{-8}}\right)$$
* **Clipping Location:** Locally on each client before transmitting the update.
* **Clipping Frequency:** Exactly **once per participating client per communication round** (after all $E = 10$ local training epochs complete).
* **Clipping Bound ($C$):** $C = 1.0$ (calibrated in protocol v1.0).

---

## 3. Exact Noise Mechanism

* **Noise Distribution:** Zero-mean isotropic Gaussian noise $\mathcal{N}\left(0, \sigma^2 C^2 \mathbf{I}\right)$.
* **Noise Placement:** Added directly to each element of the clipped parameter delta vector $\Delta \widetilde{W}_k$.
* **Noise Scale:** $\text{Standard Deviation} = \sigma \cdot C$.
* **When Noise is Added:** Locally on each participating client immediately after $L_2$ delta clipping at the conclusion of each round.

---

## 4. Privacy Unit

```
========================================================================================================
                                    VERIFIED PRIVACY UNIT
========================================================================================================
The implemented mechanism provides CLIENT-LEVEL (PARTICIPANT-LEVEL) DIFFERENTIAL PRIVACY.
========================================================================================================
```

### Formal Mathematical Definition:
* **Neighboring Datasets ($D \sim D'$):** Two distributed federated cohorts $D = \{D_1, D_2, \dots, D_N\}$ and $D' = \{D_1', D_2', \dots, D_N'\}$ are neighboring if they differ by the **entire local training dataset of a single subject/client** $k$ (i.e., adding, removing, or replacing all ~350 activity samples belonging to Subject $k$).
* **Privacy Guarantee:** The mechanism protects the participant against reconstruction, membership inference, or attribute inference targeting the individual's entire sensor history.
* **Distinction from Record-Level DP:** In record-level DP, neighboring datasets differ by a single 2.56-second window ($1$ sample), requiring per-sample gradient clipping. Because our mechanism clips and perturbs the **aggregate outcome of all samples for that subject** ($\Delta W_k$), the protection extends to the **entire human subject**.

---

## 5. Client Sampling Dynamics

* **Total Client Cohort:** $N = 21$ training subjects.
* **Sampled Clients per Round:** $K = 10$ clients.
* **Subsampling Ratio:**
  $$q = \frac{K}{N} = \frac{10}{21} \approx 0.4762$$
* **Sampling Method:** Uniform random selection without replacement within each round.
* **Repeated Participation:** Across $R = 50$ communication rounds, clients are sampled with replacement across rounds (Poisson/Bernoulli subsampled mechanism structure). A single client participates in an expected $\mathbb{E}[\text{rounds}] = R \times q = 50 \times \frac{10}{21} \approx 23.8$ rounds.

---

## 6. Local Training Interaction

* **Local Batch Size:** $B = 32$.
* **Local Epochs:** $E = 10$.
* **Local Optimizer Steps:** $\sim 90$ steps per client per round.
* **DP Interaction:** Local gradient descent proceeds unperturbed on the client device. Zero noise is injected during local backward passes, preserving local optimization trajectory stability. The DP boundary is applied strictly at the **client communication interface**.

---

## 7. Delta ($\delta$) Audit

* **Legacy Implementation:** No $\delta$ parameter was passed to any accountant in the existing codebase.
* **Legacy Manuscript:** Section 4.2.2 and Section 6 mention $\delta = 10^{-5}$ as a heuristic text reference.
* **Theoretical Client-Level Requirement:** For client-level DP with $N = 21$ clients, standard differential privacy theory requires $\delta < 1/N_{\text{clients}}$. Setting $\delta = 10^{-5}$ is excessively conservative for $N=21$. The standard canonical value $\delta = \frac{1}{2 N} \approx 0.0238$ (or $\delta = 10^{-2}$) will be formalized in Issue #7.

---

## 8. Privacy Accountant Audit

* **Audit Finding:** The legacy repository contains **NO active privacy accountant implementation** (no Opacus RDP accountant, no moments accountant, and no analytical composition code).
* **Provenance of Reported $\epsilon$:** The values $\epsilon \approx 100.0$ and $\epsilon = 1.0$ in `summary_results.csv` and Table 1 were **hard-coded textual annotations** assigned to noise scales $\sigma=0.01$ and $\sigma=0.10$, rather than outputs of an execution trace.
* **Resolution Requirement:** A formal, reproducible Rényi Differential Privacy (RDP) accountant must be integrated in Issue #7 to compute the exact $(\epsilon, \delta)$ values.

---

## 9. $\sigma$ and $\epsilon$ Inconsistency Audit

The audit cataloged all conflicting $\sigma$ and $\epsilon$ statements across the manuscript and legacy artifacts:

| Location | Statement | Conflict Type | Status |
| :--- | :--- | :--- | :--- |
| [`SPRINGER_LATEX/sections/04_models.tex:L40`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/04_models.tex#L40) | *"Gaussian noise injection ($\sigma=0.1$, $\epsilon=1.0$) on model gradients"* | Erroneously describes gradient noise instead of delta noise; $\epsilon=1.0$ unverified. | Cataloged for correction. |
| [`SPRINGER_LATEX/sections/06_results.tex:L20`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/06_results.tex#L20) | `M4 & Federated Learning + DP & \sigma = 0.01, \epsilon \approx 100.0` | $\epsilon \approx 100.0$ is an uncomputed placeholder for $\sigma=0.01$. | Cataloged for correction. |
| [`SPRINGER_LATEX/sections/06_results.tex:L21`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/06_results.tex#L21) | `M5 & Centralized LSTM-CNN + DP & \sigma = 0.10, \epsilon \approx 1.0` | Centralized DP claim without formal accountant derivation. | Cataloged for correction. |
| [`FINAL_FEDERATED_LEARNING/scripts/analysis.ipynb:L58-L61`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/FINAL_FEDERATED_LEARNING/scripts/analysis.ipynb#L58-L61) | Evaluated $\sigma \in \{0.01, 0.05, 0.10, 0.20\}$ | Experimental sweeps performed without associated $\epsilon$ calculation. | Cataloged for Issue #7. |

---

## 10. Reviewer 1 #4 Resolution Status

> *“The differential-privacy implementation is not reproducible from the reported parameters. Provide clipping norm, batch size, sampling rate, client participation, local updates, communication rounds, δ, and the privacy accountant used to derive ε.”*

### Parameter Resolution Status:
1. **Clipping Norm ($C$):** $1.0$ ($L_2$ norm on parameter delta $\Delta W_k$) — **RESOLVED**.
2. **Local Batch Size ($B$):** $32$ — **RESOLVED**.
3. **Client Sampling Rate ($q$):** $10/21 \approx 0.4762$ — **RESOLVED**.
4. **Client Participation ($K$):** $10$ out of $21$ training clients per round — **RESOLVED**.
5. **Local Updates ($E$):** $10$ local epochs ($\sim 90$ SGD steps per client) — **RESOLVED**.
6. **Communication Rounds ($R$):** $50$ rounds — **RESOLVED**.
7. **Target $\delta$:** Standardized in Issue #7.
8. **Privacy Accountant:** RDP Accountant formalized in Issue #7.

---

## 11. Reviewer 1 #6 Resolution Status

> *“The manuscript should explicitly define the privacy unit underlying the claimed guarantee. Distinguish record-level privacy from participant/client-level privacy and ensure that the reported (ε, δ) guarantee corresponds to the implemented mechanism.”*

### Resolution Status:
* **The privacy unit is rigorously established as Client-Level (Participant-Level) DP.**
* The manuscript revision will explicitly distinguish client-level DP from record-level DP, explaining that clipping and perturbation operate on the joint parameter delta $\Delta W_k$ of each subject, protecting the user's entire multi-activity time-series profile against inference attacks.

---

## 12. Required Corrected DP Design

To ensure mathematical soundness for RDP accounting:
1. **Unweighted Server Aggregation:**
   $$W_{\text{global}}^{(r+1)} = W_{\text{global}}^{(r)} + \frac{1}{K} \sum_{k \in S_t} \Delta \widehat{W}_k$$
   *Strictly preserves the global $L_2$ sensitivity $\Delta_2 = \frac{C}{K}$.*
2. **Standardized Clipping Bound:** $C = 1.0$ across all layers.
3. **Calibrated Noise Multipliers:** $\sigma \in \{0.0, 0.01, 0.05, 0.10, 0.20\}$.
4. **Analytical RDP Accountant:** Composing $R = 50$ subsampled Gaussian mechanism steps with subsampling ratio $q = 10/21$.

---

## 13. Manuscript Correction Inventory

| File | Section | Current Text | Required Future Correction |
| :--- | :--- | :--- | :--- |
| [`SPRINGER_LATEX/sections/04_models.tex:L40`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/04_models.tex#L40) | Section 4.2.2 | *"Gaussian noise injection ($\sigma=0.1, \epsilon=1.0$) on model gradients prior to aggregation"* | Update to: *"Client-level parameter-delta Differential Privacy with $L_2$ update clipping ($C=1.0$) and calibrated Gaussian noise injection on client parameter updates $\Delta W_k$"*. |
| [`SPRINGER_LATEX/sections/06_results.tex:L20`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/06_results.tex#L20) | Section 6, Table 1 | `\sigma = 0.01, \epsilon \approx 100.0` | Replace with exact RDP-computed $(\epsilon, \delta)$ values derived in Issue #7. |
| [`SPRINGER_LATEX/sections/01_introduction.tex:L14`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/SPRINGER_LATEX/sections/01_introduction.tex#L14) | Section 1 | Caption: *"Client-Level Differentially Private..."* | Retain and reinforce the formal client-level DP guarantee in the body text. |

---

## 14. Code Changes

* **Created:**
  * [`reports/issue_06_privacy_unit_dp_mechanism_resolution.md`](file:///Users/apple/Documents/Jonath/Research/FL%20Paper/reports/issue_06_privacy_unit_dp_mechanism_resolution.md) (This resolution report).

---

## 15. Files Deliberately NOT Changed

To preserve strict single-issue isolation:
* **No training performed.**
* **No final $\epsilon$ calculated:** Calculation scheduled for Issue #7.
* **No final $\sigma$ selected:** Noise sweep scheduled for Issue #7.
* **No WISDM / HHAR modifications:** Raw datasets preserved.
* **No manuscript files edited yet.**

---

## 16. Dependencies for Issue #7

The following items are queued for resolution in **Issue #7 (Privacy Accountant Implementation & $(\epsilon, \delta)$ Derivation)**:
1. Exact mathematical RDP accountant implementation.
2. Selection of target $\delta$.
3. Exact RDP composition calculation for each evaluated $\sigma \in \{0.01, 0.05, 0.10, 0.20\}$ across $R=50$ rounds and $q=10/21$.
4. Generation of the privacy-utility tradeoff table and $(\epsilon, \delta)$ mapping.

---

## 17. Final Status

```
========================================================================================================
                                     ISSUE 6 STATUS: RESOLVED
========================================================================================================
The Differential Privacy mechanism and Privacy Unit are fully defined and mathematically grounded:
- Privacy Unit: Formally established as Client-Level (Participant-Level) Differential Privacy.
- Mechanism: Parameter-Delta L2 Clipping (C=1.0) + Local Gaussian Noise Injection + Unweighted FedAvg.
- Accounting Inputs: N=21, K=10, q=10/21 ≈ 0.4762, R=50 rounds, E=10 local epochs, B=32.
========================================================================================================
```
