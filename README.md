# Responsible AI for Geospatial Intelligence

A four-phase methodology for embedding Privacy, Fairness, Transparency, and Explainability into geospatial AI development, validated across two structurally distinct experiments using TSViT as the shared model backbone.

---

## Project Structure

```
rai_geospatial/
├── configs/
│   └── config.yaml              # All hyperparameters, acceptance thresholds, and PARC settings
├── data/
│   ├── dataset.py               # GeospatialDataset — loads parcels or pixels
│   ├── preprocessing.py         # Phase 1 data cleaning and fusion pipelines
│   ├── augmentation.py          # Spatial, spectral, and rotation augmentation
│   └── sampler.py               # Stratified subgroup sampler for fair batching
├── models/
│   ├── tsvit.py                 # TSViT with dual temporal and spatial attention
│   └── losses.py                # Fairness-penalised cross-entropy loss
├── phases/
│   ├── parc.py                  # PARC — Pareto-Adaptive R-AI Compliance (pre-phase calibration)
│   ├── phase1_privacy.py        # Planar Laplace, spatial k-anonymity, DCR, MIA
│   ├── phase2_training.py       # Fair training with DP-SGD support via Opacus
│   ├── phase3_transparency.py   # Metamorphic testing (MR1–MR4) and DCS audit
│   └── phase4_explainability.py # GradCAM++, Integrated Gradients, TAC metric
├── evaluation/
│   └── assurance.py             # Cross-phase RAI assurance and report generation
├── explainability/
│   └── visualiser.py            # Publication-quality figures for all four phases
├── utils/
│   ├── logger.py                # Structured logging to stdout and file
│   ├── metrics.py               # mIoU, macro-F1, ROC-AUC, per-class metrics
│   ├── seed.py                  # Reproducibility seed setting
│   └── checkpoint.py            # Model checkpoint save/load
├── outputs/                     # Generated reports, checkpoints, and figures
├── run_experiments.py           # Main entry point — runs PARC then both experiments end-to-end
└── requirements.txt
```

---

## Experiments

**Experiment 1 — Publicly Reproducible Validation**
Crop type classification using PASTIS/EuroCrops/Sentinel-2 across 9 crop classes. All 12 RAI acceptance criteria are evaluated against an unintervened TSViT baseline.

**Experiment 2 — Real-World Generalisability**
Urban flood risk classification for a UK Sustainable Drainage Systems (SuDS) pilot. Six heterogeneous geospatial sources are fused into a 14-channel input. Five ordinal risk levels are classified across five urban zone types.

---

## PARC — Pareto-Adaptive R-AI Compliance

PARC runs **once** on the target dataset before the four phases begin. It replaces heuristic thresholds with statistically grounded values, makes R-AI characteristic interactions empirically measurable, and selects a configuration that is robust to threshold perturbation.

| Step | What it does | Mechanism |
|------|-------------|-----------|
| 0 — Threshold derivation | Anchors each τᵢ to a reference distribution | Bootstrap (500 replicates); one-sided (1−α) quantile per criterion |
| 1 — Interdependency profiling | Fits GP surrogates, computes **R-AI Interdependency Matrix M** | Latin-hypercube sampling (K=12), Matérn-5/2 GPs, central finite differences at θ₀ |
| 2 — Adaptive phase reordering | Finds π* minimising destructive inter-phase interference | Exhaustive enumeration of all 4! = 24 orderings |
| 3 — Max-min slack selection | Selects θ* and emits a **slack certificate** | Differential evolution (~200 evaluations on GP surrogate) |

PARC outputs drive all four phases: derived thresholds replace the fixed values in `config.yaml`, the optimal configuration (ε, λ₁, λ₂) is forwarded to Phase 1–2, and the phase ordering determines evaluation priority. If no feasible configuration exists, PARC returns an **infeasibility certificate** identifying the binding criterion conflict.

Configure PARC in `configs/config.yaml` under the `parc:` key:

```yaml
parc:
  probe_budget: 12      # K — LHS configurations for GP fitting
  alpha: 0.05           # one-sided significance level for τ derivation
  n_bootstrap: 500      # bootstrap replicates
  de_max_iter: 200      # differential-evolution iterations
  random_state: 42
  dataset_stats: {}     # optional anchors, e.g. {gradcam_mean: 0.62, gradcam_std: 0.03}
```

---

## Four-Phase Methodology

| Phase | Characteristic | Key Techniques | Acceptance Criteria |
|-------|---------------|----------------|---------------------|
| 1 — Data Collection & Preprocessing | Privacy | Planar Laplace (geo-indistinguishability), spatial k-anonymity (k≥5), DCR auditing | MIA ≤ 0.52, ε ≤ 5.0, δ ≤ 1e-5, DCR ≥ 5σ |
| 2 — Model Training & Development | Fairness | Fairness-penalised loss (λ₁, λ₂), stratified subgroup sampling, DP-SGD via Opacus | ΔTPR ≤ 0.05, Dem. Parity ≥ 0.80, SDS ≤ 15% |
| 3 — Model Evaluation & Transparency Verification | Transparency | Metamorphic Testing (MR1–MR4, 5000 pairs each), Documentation Completeness Score | MT Violation Rate ≤ 10%, DCS ≥ 90% |
| 4 — Deployment & Explainability | Explainability | GradCAM++ (WHERE), Integrated Gradients (WHICH BANDS), TAC (WHEN) | GradCAM++ IoU ≥ 0.60, NIR+Red ≥ 50%, TAC ≥ 0.80 |

---

## Installation

```bash
pip install -r requirements.txt
```

GPU training requires CUDA 11.8+ and a compatible PyTorch build. The implementation runs on CPU for testing but an NVIDIA A100 or equivalent is recommended for full training.

---

## Running the Experiments

```bash
cd rai_geospatial
python run_experiments.py
```

This runs PARC first, then both experiments sequentially. The execution order is:

1. **PARC** — derives statistically grounded thresholds, computes the Interdependency Matrix, determines the optimal phase ordering, and selects the best configuration (ε, λ₁, λ₂).
2. **Experiment 1** (PASTIS/EuroCrops) — four phases run with PARC-derived thresholds and configuration.
3. **Experiment 2** (SuDS Flood Risk) — same, reusing the same PARC result.

PARC output is saved to `outputs/parc_result.json`. Each experiment then executes all four phases, runs the cross-phase assurance check, saves a JSON report to `outputs/`, and writes figures to `outputs/figures/`.

To modify hyperparameters, acceptance thresholds, or PARC settings, edit `configs/config.yaml`.

---

## Configuration

Key parameters in `configs/config.yaml`:

```yaml
phase1:
  epsilon_exp1: 4.1        # Planar Laplace privacy budget — Experiment 1
  epsilon_exp2: 4.8        # Planar Laplace privacy budget — Experiment 2
  k_anonymity: 5           # Minimum spatial neighbourhood size
  mia_threshold: 0.52      # Maximum acceptable MIA attack accuracy

phase2:
  lambda1: 0.3             # TPR disparity penalty coefficient
  lambda2: 0.2             # Spatial disparity penalty coefficient
  delta_tpr_threshold: 0.05

phase4:
  gradcam_iou_threshold: 0.60
  tac_threshold: 0.80
  phenological_windows:
    winter_wheat: [60, 90, 150, 175]
    maize: [190, 210, 250, 275]
```

---

## Outputs

After a successful run, the `outputs/` directory contains:

- `parc_result.json` — PARC outputs: derived thresholds, Interdependency Matrix, phase ordering, optimal config, slack certificate
- `assurance_report_pastis_eurocrops.json` — full acceptance criteria results for Experiment 1
- `assurance_report_suds_flood_risk.json` — full acceptance criteria results for Experiment 2
- `checkpoints/` — best model weights for each experiment
- `figures/pastis_eurocrops/` — GradCAM++ IoU, TAC, MT violation rate, fairness plots
- `figures/suds_flood_risk/` — same set for Experiment 2
- `run.log` — full training and evaluation log

---

## Key Novel Contributions

**PARC (Pareto-Adaptive R-AI Compliance)**, implemented in `phases/parc.py`, contributes three elements absent from prior R-AI pipelines:

1. **Statistically grounded thresholds** — each τᵢ is derived from a reference distribution on the target dataset, removing dependence on heuristic values that may not generalise across datasets.
2. **R-AI Interdependency Matrix M** — the first empirical measurement of how tightening one RAI criterion (privacy, fairness, transparency, explainability) affects the others, computed from Gaussian Process surrogates fitted to Latin-hypercube probe data.
3. **Slack certificate** — threshold sensitivity is transformed from a post-hoc check into part of the optimisation objective, producing a certificate that records the margin by which each threshold is satisfied.

The **Temporal Attribution Consistency (TAC)** metric, implemented in `phases/phase4_explainability.py`, measures whether TSViT's temporal attention concentrates on agronomically or hydrologically validated key dates rather than data artefacts such as cloud-free acquisition scheduling patterns. TAC is computed per class against expert-validated phenological or hydrological windows defined in `config.yaml`.

---

## Citation

If you use this implementation, please cite:

```
Hochbauer, Basheer, Frincu, Islam. "Responsible AI for Geospatial Intelligence:
Empirical Validation of Privacy, Fairness, Transparency, and Explainability
in Multi-Temporal Satellite Classification." Ai4MultiGIS Project,
EU Horizon Europe, 2025.
```

---

## Dependencies

- **PyTorch ≥ 2.1** — model training and inference
- **Opacus ≥ 1.4** — DP-SGD differential privacy enforcement
- **Captum ≥ 0.7** — Integrated Gradients attribution
- **scikit-learn** — fairness metrics and evaluation
- **scipy** — spatial distance computations for k-anonymity and DCR
- **matplotlib** — figure generation
