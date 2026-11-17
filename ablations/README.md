# STELLAR-ARM Ablation Studies

This directory contains the complete set of **7 ablation experiments** for **STELLAR-ARM** evaluated on the frozen 750-pair Held-Out Test Benchmark (`arm_test_benchmark.parquet`).

The experiments cover both **architectural component ablations** and **data-distribution ablations**.

---

## Master Ablation Summary

All models use 4-bit NF4 QLoRA on `Qwen/Qwen3-8B` ($r=16$, $\alpha=32$) trained on Shard 1 (7,500 pairs, 469 planned steps) on 2× NVIDIA Tesla T4:

| # | Variant / Notebook | Category | Overall Acc | Dim MAE | Level-C Viol | Veto $\lambda$ | R1 (General) | R3 (Code) | R4 (IFEval) | R5 (Trunc) | R6 (Off-Diag) | R7 (Safety) |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | [`ablations-stellar-arm-independent-6d.ipynb`](ablations-stellar-arm-independent-6d.ipynb) | Architecture | **93.73%** | 0.0755 | 0.13% | 0.000 | 75.8% | **100.0%** | 90.0% | **100.0%** | 95.0% | 97.0% |
| **2** | [`ablations-stellar-arm-no-veto.ipynb`](ablations-stellar-arm-no-veto.ipynb) | Architecture | **93.73%** | 0.0788 | **0.00%** | 0.000 | **77.5%** | **100.0%** | 89.0% | **100.0%** | 95.0% | 96.0% |
| **3** | [`ablations-stellar-in-backbone-only.ipynb`](ablations-stellar-in-backbone-only.ipynb) | Architecture | **93.07%** | 0.0820 | **0.00%** | 2.020 | 70.8% | **100.0%** | 90.0% | **100.0%** | **96.0%** | 97.0% |
| **4** | [`ablations-stellar-no-cf-loss.ipynb`](ablations-stellar-no-cf-loss.ipynb) | Architecture | **92.80%** | **0.0736** | **0.00%** | 2.020 | 70.0% | **100.0%** | 90.0% | **100.0%** | 94.0% | **98.0%** |
| **5** | [`ablations-stellar-scalar-rm.ipynb`](ablations-stellar-scalar-rm.ipynb) | Architecture | **92.67%** | 0.4481 | **42.93%** | 0.000 | **77.5%** | 98.2% | 84.0% | 99.0% | 95.0% | 96.0% |
| **6** | [`ablations-scalar-rm-standard-data.ipynb`](ablations-scalar-rm-standard-data.ipynb) | Data Dist. | **88.93%** | 0.3995 | 0.07% | 0.000 | 74.2% | 90.0% | 88.0% | 96.0% | 88.0% | 87.0% |
| **7** | [`ablations-stellar-arm-standard-data.ipynb`](ablations-stellar-arm-standard-data.ipynb) | Data Dist. | **88.53%** | 0.2837 | **0.00%** | 2.046 | 69.2% | 90.9% | 89.0% | 87.0% | 92.0% | 93.0% |

*(Note: In regime R2 Math Reasoning, all models across both data settings scored 99.2%–100.0%.)*

---

## Notebook Descriptions

```text
ablations/
├── README.md
├── ablations-stellar-scalar-rm.ipynb          # Pure Bradley-Terry scalar baseline (no 6D supervision, no DAG)
├── ablations-stellar-arm-independent-6d.ipynb  # Diagonal 6D multi-head baseline (no causal DAG mixer)
├── ablations-stellar-in-backbone-only.ipynb    # In-backbone causal bias only (no post-backbone B2 mixer)
├── ablations-stellar-no-cf-loss.ipynb          # Without C2 counterfactual activation patching loss
├── ablations-stellar-arm-no-veto.ipynb         # Without A1 LRSV non-compensatory gatekeeper veto
├── ablations-scalar-rm-standard-data.ipynb     # Pure Bradley-Terry scalar RM on uncurated standard data
└── ablations-stellar-arm-standard-data.ipynb   # STELLAR In-Backbone RM on uncurated standard data
```

---

## Key Findings

1. **Hierarchical Reward Hacking Prevention (Level-C Violations):**
   - The scalar baseline (`ablations-stellar-scalar-rm.ipynb`) suffers a **42.93% Level-C violation rate**, allowing superficial presentation to compensate for broken reasoning or factual flaws.
   - All STELLAR variants with C1 Monotonicity constraints achieve strictly **0.00% violations** across both curated and uncurated data distributions.
   - Independent multi-objective heads (`ablations-stellar-arm-independent-6d.ipynb`) without causal routing allow violations to re-emerge (**0.13%**).

2. **Data Pipeline Impact:**
   - Training on the curated alignment pipeline yields a **+4.5% overall accuracy gain**, a **+13.0% gain on truncated responses**, a **+10.0% gain on safety refusals**, and a **3.5× reduction in dimensional MAE** compared to standard natural preference data.

3. **Disentangled Dimensional Accuracy:**
   - Multi-objective ARM heads achieve an average Dim MAE of **~0.075**, compared to **~0.40–0.45** for scalar models.
