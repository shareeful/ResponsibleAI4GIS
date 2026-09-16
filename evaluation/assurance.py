import numpy as np
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional
from pathlib import Path
from utils.logger import get_logger
from utils.metrics import compute_miou, compute_macro_f1, compute_roc_auc

logger = get_logger(__name__)


@dataclass
class AssuranceReport:
    experiment_name: str
    privacy_passed: bool
    fairness_passed: bool
    transparency_passed: bool
    explainability_passed: bool
    all_passed: bool
    utility_cost_pp: float
    utility_within_ceiling: bool
    metrics: Dict = field(default_factory=dict)
    acceptance_criteria: Dict = field(default_factory=dict)
    summary: str = ""


class RAIAssurance:
    # Default heuristic thresholds — overridden by PARC-derived values when available
    THRESHOLDS = {
        "mia_accuracy":            ("<=", 0.52),
        "dp_epsilon":              ("<=", 5.0),
        "dp_delta":                ("<=", 1e-5),
        "dcr_sigma":               (">=", 5.0),
        "delta_tpr":               ("<=", 0.05),
        "demographic_parity":      (">=", 0.80),
        "sds_pct":                 ("<=", 15.0),
        "mt_violation_rate":       ("<=", 10.0),
        "dcs_score":               (">=", 90.0),
        "gradcam_iou":             (">=", 0.60),
        "nir_red_attribution_pct": (">=", 50.0),
        "tac_score":               (">=", 0.80),
    }

    # Operator polarity (needed when applying custom thresholds from PARC)
    _THRESHOLD_OPS = {
        "mia_accuracy":            "<=",
        "dp_epsilon":              "<=",
        "dp_delta":                "<=",
        "dcr_sigma":               ">=",
        "delta_tpr":               "<=",
        "demographic_parity":      ">=",
        "sds_pct":                 "<=",
        "mt_violation_rate":       "<=",
        "dcs_score":               ">=",
        "gradcam_iou":             ">=",
        "nir_red_attribution_pct": ">=",
        "tac_score":               ">=",
    }

    def __init__(
        self,
        utility_loss_ceiling_pp: float = 5.0,
        parc_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.utility_loss_ceiling_pp = utility_loss_ceiling_pp
        # Apply PARC-derived thresholds if provided, keeping operator polarity
        if parc_thresholds:
            self.THRESHOLDS = {
                key: (self._THRESHOLD_OPS[key], parc_thresholds[key])
                for key in self.THRESHOLDS
                if key in parc_thresholds
            }
            logger.info("RAIAssurance: using PARC-derived thresholds.")

    def evaluate(
        self,
        experiment_name: str,
        privacy_result,
        fairness_result,
        transparency_result,
        explainability_result,
        baseline_metric: float,
        rai_metric: float,
    ) -> AssuranceReport:
        utility_cost = (baseline_metric - rai_metric) * 100.0

        metrics = {
            "mia_accuracy": privacy_result.mia_accuracy,
            "dp_epsilon": privacy_result.epsilon,
            "dp_delta": privacy_result.delta,
            "dcr_sigma": privacy_result.dcr_sigma,
            "delta_tpr": fairness_result.delta_tpr,
            "demographic_parity": fairness_result.demographic_parity,
            "sds_pct": fairness_result.sds,
            "mt_violation_rate": transparency_result.mt_violation_rate,
            "dcs_score": transparency_result.dcs_score,
            "gradcam_iou": explainability_result.gradcam_iou,
            "nir_red_attribution_pct": explainability_result.nir_red_attribution_pct,
            "tac_score": explainability_result.tac_score,
            # Fidelity diagnostics (not acceptance criteria — reported for reviewer validation)
            "deletion_auc": explainability_result.deletion_auc,
            "insertion_auc": explainability_result.insertion_auc,
            "sensitivity_n": explainability_result.sensitivity_n,
            "spectral_stability": explainability_result.spectral_stability,
            "seed_consistency": explainability_result.seed_consistency,
        }

        criteria_pass = self._check_criteria(metrics)
        all_passed = all(criteria_pass.values())
        utility_ok = utility_cost <= self.utility_loss_ceiling_pp

        report = AssuranceReport(
            experiment_name=experiment_name,
            privacy_passed=privacy_result.passed,
            fairness_passed=fairness_result.passed,
            transparency_passed=transparency_result.passed,
            explainability_passed=explainability_result.passed,
            all_passed=all_passed,
            utility_cost_pp=utility_cost,
            utility_within_ceiling=utility_ok,
            metrics=metrics,
            acceptance_criteria=criteria_pass,
        )

        report.summary = self._generate_summary(report)
        self._log_report(report)

        if not utility_ok:
            logger.warning(
                f"Utility cost {utility_cost:.2f}pp exceeds ceiling {self.utility_loss_ceiling_pp}pp. "
                "Re-calibrate intervention parameters."
            )

        return report

    def _check_criteria(self, metrics: Dict) -> Dict[str, bool]:
        results = {}
        for key, (op, threshold) in self.THRESHOLDS.items():
            value = metrics.get(key, None)
            if value is None:
                results[key] = False
                continue
            if op == "<=":
                results[key] = value <= threshold
            elif op == ">=":
                results[key] = value >= threshold
        return results

    def _generate_summary(self, report: AssuranceReport) -> str:
        status = "PASSED" if report.all_passed else "FAILED"
        failed = [k for k, v in report.acceptance_criteria.items() if not v]
        lines = [
            f"RAI Assurance [{status}] — {report.experiment_name}",
            f"Privacy: {'PASS' if report.privacy_passed else 'FAIL'} | "
            f"Fairness: {'PASS' if report.fairness_passed else 'FAIL'} | "
            f"Transparency: {'PASS' if report.transparency_passed else 'FAIL'} | "
            f"Explainability: {'PASS' if report.explainability_passed else 'FAIL'}",
            f"Utility cost: {report.utility_cost_pp:.2f}pp ({'within' if report.utility_within_ceiling else 'exceeds'} ceiling)",
        ]
        if failed:
            lines.append(f"Failed criteria: {', '.join(failed)}")
        return "\n".join(lines)

    def _log_report(self, report: AssuranceReport) -> None:
        logger.info("=" * 60)
        for line in report.summary.split("\n"):
            logger.info(line)
        logger.info("Detailed metrics:")
        for k, v in report.metrics.items():
            passed = report.acceptance_criteria.get(k, True)
            logger.info(f"  {k:35s}: {v:.4f}  {'PASS' if passed else 'FAIL'}")
        logger.info("=" * 60)

    def save_report(self, report: AssuranceReport, output_dir: str) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / f"assurance_report_{report.experiment_name}.json"
        with open(path, "w") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info(f"Assurance report saved to {path}")
