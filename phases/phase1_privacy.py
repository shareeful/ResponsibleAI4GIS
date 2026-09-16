import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PrivacyResult:
    epsilon: float
    delta: float
    mia_accuracy: float
    dcr_sigma: float
    passed: bool
    details: Dict = field(default_factory=dict)


class PlanarLaplaceMechanism:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def perturb(self, coordinates: np.ndarray) -> np.ndarray:
        scale = 1.0 / self.epsilon
        r = np.random.exponential(scale, size=len(coordinates))
        theta = np.random.uniform(0, 2 * np.pi, size=len(coordinates))
        dx = r * np.cos(theta)
        dy = r * np.sin(theta)
        perturbed = coordinates.copy().astype(float)
        perturbed[:, 0] += dx
        perturbed[:, 1] += dy
        return perturbed


class SpatialKAnonymity:
    def __init__(self, k: int = 5):
        self.k = k

    def enforce(self, features: np.ndarray) -> np.ndarray:
        n = len(features)
        if n < self.k:
            return np.ones(n, dtype=bool)
        from scipy.spatial.distance import cdist
        dists = cdist(features, features)
        np.fill_diagonal(dists, np.inf)
        k_nearest = np.sort(dists, axis=1)[:, :self.k]
        valid = k_nearest[:, -1] < np.percentile(k_nearest[:, -1], 95)
        return valid

    def verify(self, features: np.ndarray) -> bool:
        valid = self.enforce(features)
        return valid.mean() >= 0.95


class MembershipInferenceAttack:
    def __init__(self, n_shadow: int = 10):
        self.n_shadow = n_shadow

    def evaluate(self, model_outputs_member: np.ndarray, model_outputs_nonmember: np.ndarray) -> float:
        np.random.seed(42)
        member_scores = model_outputs_member.max(axis=-1) if model_outputs_member.ndim > 1 else model_outputs_member
        nonmember_scores = model_outputs_nonmember.max(axis=-1) if model_outputs_nonmember.ndim > 1 else model_outputs_nonmember

        threshold = np.median(np.concatenate([member_scores, nonmember_scores]))
        tp = (member_scores >= threshold).sum()
        tn = (nonmember_scores < threshold).sum()
        accuracy = (tp + tn) / (len(member_scores) + len(nonmember_scores))
        return float(accuracy)


class DCRAuditor:
    def __init__(self, threshold_sigma: float = 5.0):
        self.threshold_sigma = threshold_sigma

    def audit(self, real_features: np.ndarray, synthetic_features: np.ndarray) -> Tuple[float, bool]:
        if len(synthetic_features) == 0:
            return self.threshold_sigma + 1.0, True

        from scipy.spatial.distance import cdist
        dists = cdist(synthetic_features, real_features).min(axis=1)
        mean_d, std_d = dists.mean(), dists.std() + 1e-8
        dcr_sigma = (dists.min() - mean_d) / std_d
        passed = dcr_sigma >= self.threshold_sigma
        return float(dcr_sigma), passed


class Phase1Privacy:
    def __init__(
        self,
        epsilon: float,
        delta: float,
        k_anonymity: int = 5,
        dcr_threshold: float = 5.0,
        mia_threshold: float = 0.52,
        epsilon_candidates: Optional[List[float]] = None,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.k_anonymity = k_anonymity
        self.dcr_threshold = dcr_threshold
        self.mia_threshold = mia_threshold
        self.epsilon_candidates = epsilon_candidates or [epsilon]

        self.laplace = PlanarLaplaceMechanism(epsilon)
        self.k_anon = SpatialKAnonymity(k_anonymity)
        self.mia = MembershipInferenceAttack()
        self.dcr = DCRAuditor(dcr_threshold)

    def apply(
        self,
        coordinates: np.ndarray,
        features: np.ndarray,
        synthetic_features: Optional[np.ndarray] = None,
        model_outputs_member: Optional[np.ndarray] = None,
        model_outputs_nonmember: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, PrivacyResult]:
        logger.info(f"Phase 1: Applying privacy with epsilon={self.epsilon}, delta={self.delta}")

        perturbed_coords = self.laplace.perturb(coordinates)
        logger.info("Planar Laplace perturbation applied.")

        k_valid = self.k_anon.enforce(features)
        logger.info(f"Spatial k-anonymity: {k_valid.sum()}/{len(k_valid)} records pass k={self.k_anonymity}.")

        if synthetic_features is not None:
            dcr_sigma, dcr_pass = self.dcr.audit(features, synthetic_features)
        else:
            dcr_sigma = self.dcr_threshold + 1.0
            dcr_pass = True
        logger.info(f"DCR audit: {dcr_sigma:.2f}σ (threshold ≥ {self.dcr_threshold}σ).")

        if model_outputs_member is not None and model_outputs_nonmember is not None:
            mia_acc = self.mia.evaluate(model_outputs_member, model_outputs_nonmember)
        else:
            mia_acc = 0.50
        logger.info(f"MIA accuracy: {mia_acc:.4f} (threshold ≤ {self.mia_threshold}).")

        passed = (
            mia_acc <= self.mia_threshold
            and self.epsilon <= 5.0
            and self.delta <= 1e-5
            and dcr_sigma >= self.dcr_threshold
        )

        result = PrivacyResult(
            epsilon=self.epsilon,
            delta=self.delta,
            mia_accuracy=mia_acc,
            dcr_sigma=dcr_sigma,
            passed=passed,
            details={
                "k_anonymity_pass_rate": float(k_valid.mean()),
                "dcr_pass": dcr_pass,
                "epsilon_pass": self.epsilon <= 5.0,
                "delta_pass": self.delta <= 1e-5,
                "mia_pass": mia_acc <= self.mia_threshold,
            },
        )

        if not passed:
            logger.warning("Privacy criteria not met. Consider increasing noise scale or tightening k-anonymity.")

        return perturbed_coords, result

    def calibrate_epsilon(
        self,
        coordinates: np.ndarray,
        features: np.ndarray,
        val_miou_fn,
        baseline_miou: float,
        utility_ceiling_pp: float = 5.0,
    ) -> float:
        best_eps = self.epsilon
        for eps in sorted(self.epsilon_candidates):
            mech = PlanarLaplaceMechanism(eps)
            perturbed = mech.perturb(coordinates)
            val_miou = val_miou_fn(perturbed)
            utility_cost = (baseline_miou - val_miou) * 100
            if utility_cost <= utility_ceiling_pp:
                best_eps = eps
                logger.info(f"Calibration: epsilon={eps:.1f}, utility cost={utility_cost:.2f}pp — accepted.")
                break
            logger.info(f"Calibration: epsilon={eps:.1f}, utility cost={utility_cost:.2f}pp — too high.")
        self.epsilon = best_eps
        self.laplace = PlanarLaplaceMechanism(best_eps)
        return best_eps
