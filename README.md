# Responsible AI for Geospatial Intelligence

A four-phase methodology for embedding Privacy, Fairness, Transparency, and Explainability into geospatial AI development, validated across two structurally distinct experiments using TSViT as the shared model backbone.

---

## Project Structure

```
rai_geospatial/
├── configs/
│   └── config.yaml              # All hyperparameters and acceptance thresholds
├── data/
│   ├── dataset.py               # GeospatialDataset — loads parcels or pixels
│   ├── preprocessing.py         # Phase 1 data cleaning and fusion pipelines
│   ├── augmentation.py          # Spatial, spectral, and rotation augmentation
│   └── sampler.py               # Stratified subgroup sampler for fair batching
├── models/
│   ├── tsvit.py                 # TSViT with dual temporal and spatial attention
│   └── losses.py                # Fairness-penalised cross-entropy loss
├── phases/
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
├── run_experiments.py           # Main entry point — runs both experiments end-to-end
└── requirements.txt
```

---

## Experiments

**Experiment 1 — Publicly Reproducible Validation**
Crop type classification using PASTIS/EuroCrops/Sentinel-2 across 9 crop classes. All 12 RAI acceptance criteria are evaluated against an unintervened TSViT baseline.

**Experiment 2 — Real-World Generalisability**
Urban flood risk classification for a UK Sustainable Drainage Systems (SuDS) pilot. Six heterogeneous geospatial sources are fused into a 14-channel input. Five ordinal risk levels are classified across five urban zone types.

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

This runs both experiments sequentially. Each experiment executes all four phases, runs the cross-phase assurance check, saves a JSON report to `outputs/`, and writes figures to `outputs/figures/`.

To modify hyperparameters, acceptance thresholds, or privacy budgets, edit `configs/config.yaml`. All parameters are documented inline in that file.

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

- `assurance_report_pastis_eurocrops.json` — full acceptance criteria results for Experiment 1
- `assurance_report_suds_flood_risk.json` — full acceptance criteria results for Experiment 2
- `checkpoints/` — best model weights for each experiment
- `figures/pastis_eurocrops/` — GradCAM++ IoU, TAC, MT violation rate, fairness plots
- `figures/suds_flood_risk/` — same set for Experiment 2
- `run.log` — full training and evaluation log

---

## Key Novel Contribution

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
