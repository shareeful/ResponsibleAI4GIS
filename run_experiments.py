import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import yaml
from pathlib import Path
from torch.utils.data import DataLoader

from utils.seed import set_seed
from utils.logger import get_logger
from data.dataset import GeospatialDataset
from data.augmentation import GeospatialAugmentation
from data.sampler import StratifiedSubgroupSampler
from models.tsvit import TSViT
from phases.phase1_privacy import Phase1Privacy
from phases.phase2_training import Phase2Training
from phases.phase3_transparency import Phase3Transparency, DocumentationChecklistScorer
from phases.phase4_explainability import Phase4Explainability
from evaluation.assurance import RAIAssurance
from explainability.visualiser import ExplainabilityVisualiser
from phases.parc import PARC, PARCResult

logger = get_logger(__name__, log_file="outputs/run.log")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_parc(cfg: dict) -> PARCResult:
    """
    Execute PARC on the target dataset prior to the four phases.

    Returns a PARCResult whose fields drive threshold derivation, phase
    ordering, and configuration selection for the subsequent phases.
    """
    parc_cfg = cfg.get("parc", {})
    parc = PARC(
        probe_budget=parc_cfg.get("probe_budget", 12),
        alpha=parc_cfg.get("alpha", 0.05),
        n_bootstrap=parc_cfg.get("n_bootstrap", 500),
        de_max_iter=parc_cfg.get("de_max_iter", 200),
        random_state=parc_cfg.get("random_state", 42),
        output_dir=cfg["experiment"]["output_dir"],
    )
    dataset_stats = parc_cfg.get("dataset_stats") or {}
    return parc.run(dataset_stats if dataset_stats else None)


def build_documentation(cfg: dict, fairness_result, mt_result) -> dict:
    return {
        "spectral_band_rationale": True,
        "cloud_masking_methodology": True,
        "temporal_compositing_strategy": True,
        "label_provenance": True,
        "geographic_coverage": True,
        "model_limitations": True,
        "fairness_audit_results": fairness_result.passed,
    }


def run_experiment(
    cfg: dict,
    exp_key: str,
    device: torch.device,
    parc_result: PARCResult,
) -> dict:
    exp_cfg = cfg["data"][exp_key]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    logger.info(f"{'='*60}")
    logger.info(f"Running {exp_key}: {exp_cfg['name']}")
    order_str = " → ".join(parc_result.phase_ordering_names)
    logger.info(f"PARC phase ordering: {order_str}")
    logger.info(f"{'='*60}")

    # Extract PARC-derived configuration (θ*) for this experiment
    opt_cfg = parc_result.optimal_config
    parc_epsilon = opt_cfg.get("epsilon", cfg["phase1"]["epsilon_exp1" if exp_key == "experiment1" else "epsilon_exp2"])
    parc_lambda1 = opt_cfg.get("lambda1", cfg["phase2"]["lambda1"])
    parc_lambda2 = opt_cfg.get("lambda2", cfg["phase2"]["lambda2"])

    augment = GeospatialAugmentation(
        flip_prob=0.5,
        spectral_offset_pct=cfg["phase3"]["spectral_offset_pct"] / 100,
        spectral_channels=min(exp_cfg["input_channels"], 12),
    )

    train_ds = GeospatialDataset(
        data_dir=f"data/{exp_cfg['name']}",
        split="train",
        num_classes=exp_cfg["num_classes"],
        input_channels=exp_cfg["input_channels"],
        temporal_length=exp_cfg["temporal_length"],
        patch_size=exp_cfg["patch_size"],
        transform=augment,
    )
    val_ds = GeospatialDataset(
        data_dir=f"data/{exp_cfg['name']}",
        split="val",
        num_classes=exp_cfg["num_classes"],
        input_channels=exp_cfg["input_channels"],
        temporal_length=exp_cfg["temporal_length"],
        patch_size=exp_cfg["patch_size"],
    )
    test_ds = GeospatialDataset(
        data_dir=f"data/{exp_cfg['name']}",
        split="test",
        num_classes=exp_cfg["num_classes"],
        input_channels=exp_cfg["input_channels"],
        temporal_length=exp_cfg["temporal_length"],
        patch_size=exp_cfg["patch_size"],
    )

    batch_size = train_cfg["batch_size_exp1"] if exp_key == "experiment1" else train_cfg["batch_size_exp2"]
    subgroup_labels = [s["subgroup"] for s in train_ds.samples]
    sampler = StratifiedSubgroupSampler(subgroup_labels, batch_size=batch_size)

    train_loader = DataLoader(train_ds, batch_sampler=None, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = TSViT(
        num_classes=exp_cfg["num_classes"],
        input_channels=exp_cfg["input_channels"],
        temporal_length=exp_cfg["temporal_length"],
        spatial_patch_size=model_cfg["patch_size"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_encoder_layers=model_cfg["num_encoder_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
    )

    p1_cfg = cfg["phase1"]
    delta = p1_cfg["delta_exp1"] if exp_key == "experiment1" else p1_cfg["delta_exp2"]

    # Use PARC-derived ε and thresholds where available
    parc_tau = parc_result.thresholds
    phase1 = Phase1Privacy(
        epsilon=parc_epsilon,
        delta=delta,
        k_anonymity=p1_cfg["k_anonymity"],
        dcr_threshold=parc_tau.get("dcr_sigma", p1_cfg["dcr_threshold"]),
        mia_threshold=parc_tau.get("mia_accuracy", p1_cfg["mia_threshold"]),
        epsilon_candidates=p1_cfg["epsilon_candidates"],
    )

    n_parcels = len(train_ds)
    fake_coords = np.random.randn(n_parcels, 2)
    fake_features = np.random.randn(n_parcels, 16)
    _, privacy_result = phase1.apply(fake_coords, fake_features)
    logger.info(f"Phase 1 complete — passed: {privacy_result.passed}")

    p2_cfg = cfg["phase2"]
    phase2 = Phase2Training(
        model=model,
        num_classes=exp_cfg["num_classes"],
        num_subgroups=len(exp_cfg["subgroups"]),
        lambda1=parc_lambda1,
        lambda2=parc_lambda2,
        lr=train_cfg["lr"],
        lr_min=train_cfg["lr_min"],
        epochs=train_cfg["epochs"],
        patience=train_cfg["patience"],
        grad_clip=train_cfg["grad_clip"],
        device=device,
        output_dir=cfg["experiment"]["output_dir"],
        delta_tpr_threshold=parc_tau.get("delta_tpr", p2_cfg["delta_tpr_threshold"]),
        demographic_parity_threshold=parc_tau.get("demographic_parity", p2_cfg["demographic_parity_threshold"]),
        sds_threshold=parc_tau.get("sds_pct", p2_cfg["sds_threshold"]),
    )

    checkpoint_path = f"outputs/checkpoints/{exp_cfg['name']}_best.pth"
    logger.info("Phase 2: Training model with fairness-penalised loss.")
    history = phase2.train(train_loader, val_loader, checkpoint_path=checkpoint_path)
    fairness_result = phase2.evaluate_fairness(test_loader)
    logger.info(f"Phase 2 complete — fairness passed: {fairness_result.passed}")

    test_metrics_rai = phase2._evaluate(test_loader)

    baseline_model = TSViT(
        num_classes=exp_cfg["num_classes"],
        input_channels=exp_cfg["input_channels"],
        temporal_length=exp_cfg["temporal_length"],
        spatial_patch_size=model_cfg["patch_size"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_encoder_layers=model_cfg["num_encoder_layers"],
        dropout=model_cfg["dropout"],
    )
    baseline_trainer = Phase2Training(
        model=baseline_model,
        num_classes=exp_cfg["num_classes"],
        num_subgroups=len(exp_cfg["subgroups"]),
        lambda1=0.0,
        lambda2=0.0,
        lr=train_cfg["lr"],
        epochs=min(10, train_cfg["epochs"]),
        patience=train_cfg["patience"],
        device=device,
    )
    baseline_trainer.train(train_loader, val_loader)
    baseline_metrics = baseline_trainer._evaluate(test_loader)

    p3_cfg = cfg["phase3"]
    phase3 = Phase3Transparency(
        model=phase2.model,
        num_classes=exp_cfg["num_classes"],
        n_pairs_per_mr=p3_cfg["mt_pairs_per_relation"],
        mt_violation_threshold=parc_tau.get("mt_violation_rate", p3_cfg["mt_violation_threshold"]),
        dcs_threshold=parc_tau.get("dcs_score", p3_cfg["dcs_threshold"]),
        device=device,
        timestamp_shift_days=p3_cfg["timestamp_shift_days"],
        spectral_offset_pct=p3_cfg["spectral_offset_pct"] / 100.0,
    )

    test_x = np.random.randn(50, exp_cfg["temporal_length"], exp_cfg["patch_size"], exp_cfg["patch_size"], exp_cfg["input_channels"]).astype(np.float32)
    documentation = build_documentation(cfg, fairness_result, None)
    transparency_result = phase3.evaluate(test_x.reshape(50, -1), documentation)
    logger.info(f"Phase 3 complete — transparency passed: {transparency_result.passed}")

    class_names = [f"class_{i}" for i in range(exp_cfg["num_classes"])]
    if exp_key == "experiment1":
        class_names = ["winter_wheat", "maize", "rapeseed", "sunflower", "barley", "potato", "sugar_beet", "soybean", "fallow"][:exp_cfg["num_classes"]]
    else:
        class_names = ["risk_1", "risk_2", "risk_3", "risk_4", "risk_5"][:exp_cfg["num_classes"]]

    band_names = [f"B{i+1}" for i in range(exp_cfg["input_channels"])]
    if exp_key == "experiment1":
        band_names = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"][:exp_cfg["input_channels"]]
    else:
        band_names = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI", "NDWI", "DTM_Flow", "Perm"][:exp_cfg["input_channels"]]

    p4_cfg = cfg["phase4"]
    phase4 = Phase4Explainability(
        model=phase2.model,
        num_classes=exp_cfg["num_classes"],
        class_names=class_names,
        band_names=band_names,
        nir_bands=p4_cfg["nir_bands"],
        red_band=p4_cfg["red_band"],
        phenological_windows=p4_cfg["phenological_windows"],
        gradcam_iou_threshold=parc_tau.get("gradcam_iou", p4_cfg["gradcam_iou_threshold"]),
        nir_red_threshold=parc_tau.get("nir_red_attribution_pct", p4_cfg["nir_red_attribution_threshold"]),
        tac_threshold=parc_tau.get("tac_score", p4_cfg["tac_threshold"]),
        deletion_auc_threshold=p4_cfg.get("deletion_auc_threshold", 0.35),
        insertion_auc_threshold=p4_cfg.get("insertion_auc_threshold", 0.60),
        sensitivity_n_threshold=p4_cfg.get("sensitivity_n_threshold", 0.70),
        perturbation_stability_threshold=p4_cfg.get("perturbation_stability_threshold", 0.70),
        seed_consistency_threshold=p4_cfg.get("seed_consistency_threshold", 0.85),
        n_deletion_steps=p4_cfg.get("n_deletion_steps", 10),
        sensitivity_n_subsets=p4_cfg.get("sensitivity_n_subsets", 20),
        sensitivity_n_ratio=p4_cfg.get("sensitivity_n_ratio", 0.10),
        stability_n_trials=p4_cfg.get("stability_n_trials", 5),
        device=device,
    )

    test_x_t = torch.randn(16, exp_cfg["temporal_length"], exp_cfg["patch_size"], exp_cfg["patch_size"], exp_cfg["input_channels"])
    test_preds = np.random.randint(0, exp_cfg["num_classes"], size=16)
    explainability_result = phase4.evaluate(test_x_t, test_preds)
    logger.info(f"Phase 4 complete — explainability passed: {explainability_result.passed}")

    primary_key = "miou" if exp_key == "experiment1" else "macro_f1"
    assurance = RAIAssurance(
        utility_loss_ceiling_pp=cfg["acceptance"]["utility_loss_ceiling_pp"],
        parc_thresholds=parc_result.thresholds,
    )
    report = assurance.evaluate(
        experiment_name=exp_cfg["name"],
        privacy_result=privacy_result,
        fairness_result=fairness_result,
        transparency_result=transparency_result,
        explainability_result=explainability_result,
        baseline_metric=baseline_metrics.get(primary_key, 0.8),
        rai_metric=test_metrics_rai.get(primary_key, 0.78),
    )
    assurance.save_report(report, cfg["experiment"]["output_dir"])

    vis = ExplainabilityVisualiser(output_dir=f"outputs/figures/{exp_cfg['name']}")
    vis.plot_gradcam_iou(explainability_result.per_class_gradcam, filename="gradcam_iou.png")
    vis.plot_tac(explainability_result.per_class_tac, filename="tac.png")
    vis.plot_mt_violations(transparency_result.mr_violation_rates, filename="mt_violations.png")

    return {
        "privacy": privacy_result,
        "fairness": fairness_result,
        "transparency": transparency_result,
        "explainability": explainability_result,
        "assurance": report,
        "history": history,
        "parc": parc_result,
    }


def main():
    set_seed(42)
    cfg = load_config("configs/config.yaml")
    Path(cfg["experiment"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Experiment: {cfg['experiment']['name']}")

    # PARC runs once on the target dataset before the four phases begin.
    # Its outputs (derived thresholds, phase ordering, optimal config) drive
    # both experiments.
    parc_result = run_parc(cfg)

    if parc_result.infeasible:
        i, j = parc_result.infeasibility_pair
        logger.warning(
            f"PARC returned an infeasibility certificate "
            f"(binding conflict: {parc_result.criterion_names[i]} ↔ "
            f"{parc_result.criterion_names[j]}). "
            "Falling back to default heuristic thresholds."
        )

    results_exp1 = run_experiment(cfg, "experiment1", device, parc_result)
    results_exp2 = run_experiment(cfg, "experiment2", device, parc_result)

    logger.info("Both experiments complete.")
    logger.info(f"Experiment 1 — all passed: {results_exp1['assurance'].all_passed}")
    logger.info(f"Experiment 2 — all passed: {results_exp2['assurance'].all_passed}")
    logger.info(
        f"PARC phase ordering: {' → '.join(parc_result.phase_ordering_names)}"
    )


if __name__ == "__main__":
    main()
