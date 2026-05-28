"""
PARC: Pareto-Adaptive R-AI Compliance algorithm.

Executed once before the four phases to replace heuristic thresholds with
statistically grounded values, expose R-AI characteristic interactions via the
Interdependency Matrix, reorder phases to minimise destructive interference,
and select a configuration that maximises the minimum normalised slack.
"""

import json
import numpy as np
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PARCResult:
    thresholds: Dict[str, float]
    interdependency_matrix: np.ndarray      # (12, 12)
    criterion_names: List[str]
    phase_ordering: List[int]               # permutation of [0, 1, 2, 3]
    phase_ordering_names: List[str]
    optimal_config: Dict[str, float]
    slack_certificate: Dict[str, float]
    infeasible: bool = False
    infeasibility_pair: Optional[Tuple[int, int]] = None


class PARC:
    """
    Pareto-Adaptive R-AI Compliance (PARC).

    Steps
    -----
    0. Threshold derivation  — anchor each τ_i to a reference distribution.
    1. Interdependency profiling — LHS probe → GP surrogates → M matrix.
    2. Adaptive phase reordering — minimise destructive interference.
    3. Max-min slack selection  — differential evolution on GP surrogate.
    """

    CRITERIA_NAMES: List[str] = [
        "mia_accuracy",           # Privacy (c_1)
        "dp_epsilon",             # Privacy (c_2)
        "dp_delta",               # Privacy (c_3)
        "dcr_sigma",              # Privacy (c_4)
        "delta_tpr",              # Fairness (c_5)
        "demographic_parity",     # Fairness (c_6)
        "sds_pct",                # Fairness (c_7)
        "mt_violation_rate",      # Transparency (c_8)
        "dcs_score",              # Transparency (c_9)
        "gradcam_iou",            # Explainability (c_10)
        "nir_red_attribution_pct",# Explainability (c_11)
        "tac_score",              # Explainability (c_12)
    ]

    # "<=" → criterion must not exceed τ (upper-bound type)
    # ">=" → criterion must meet or exceed τ (lower-bound type)
    THRESHOLD_OPS: Dict[str, str] = {
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

    # Indices into CRITERIA_NAMES for each phase group
    PHASE_GROUPS: Dict[int, List[int]] = {
        0: [0, 1, 2, 3],   # Privacy
        1: [4, 5, 6],       # Fairness
        2: [7, 8],          # Transparency
        3: [9, 10, 11],     # Explainability
    }

    PHASE_NAMES: List[str] = ["Privacy", "Fairness", "Transparency", "Explainability"]

    # Heuristic defaults — used as distribution centres in Step 0
    DEFAULT_THRESHOLDS: Dict[str, float] = {
        "mia_accuracy":            0.52,
        "dp_epsilon":              5.0,
        "dp_delta":                1e-5,
        "dcr_sigma":               5.0,
        "delta_tpr":               0.05,
        "demographic_parity":      0.80,
        "sds_pct":                15.0,
        "mt_violation_rate":      10.0,
        "dcs_score":              90.0,
        "gradcam_iou":             0.60,
        "nir_red_attribution_pct": 50.0,
        "tac_score":               0.80,
    }

    # Prior operating point θ₀ from Section 5.2 of the paper
    PRIOR_CONFIG: Dict[str, float] = {
        "epsilon": 4.1,
        "lambda1": 0.3,
        "lambda2": 0.2,
        "w_mt":    1.0,
        "w_tac":   1.0,
    }

    # Search bounds for each configuration parameter
    CONFIG_BOUNDS: Dict[str, Tuple[float, float]] = {
        "epsilon": (1.5, 5.0),
        "lambda1": (0.1, 0.5),
        "lambda2": (0.1, 0.4),
        "w_mt":    (0.5, 2.0),
        "w_tac":   (0.5, 2.0),
    }

    # Maps each criterion index to the config parameter that most directly drives it
    _CRIT_TO_PARAM_IDX: Dict[int, int] = {
        0: 0, 1: 0, 2: 0, 3: 0,   # Privacy     → epsilon (idx 0)
        4: 1, 5: 1,                 # delta_tpr, dem_parity → lambda1 (idx 1)
        6: 2,                       # sds_pct     → lambda2 (idx 2)
        7: 3, 8: 3,                 # MT, DCS     → w_mt    (idx 3)
        9: 4, 10: 4, 11: 4,        # Explainability → w_tac (idx 4)
    }

    def __init__(
        self,
        probe_budget: int = 12,
        alpha: float = 0.05,
        n_bootstrap: int = 500,
        de_max_iter: int = 200,
        random_state: int = 42,
        criterion_evaluator: Optional[Callable[[Dict[str, float]], Dict[str, float]]] = None,
        output_dir: str = "outputs",
    ):
        """
        Parameters
        ----------
        probe_budget : int
            LHS probe count K for GP fitting.
        alpha : float
            Significance level for one-sided threshold bounds.
        n_bootstrap : int
            Bootstrap replicates for reference distribution estimation.
        de_max_iter : int
            Max iterations for differential evolution.
        random_state : int
            Global random seed.
        criterion_evaluator : callable, optional
            ``f(config: dict) -> criteria: dict`` that evaluates all 12 criteria
            under the supplied configuration.  Omit to use the built-in synthetic
            evaluator (suitable for demonstration and unit tests).
        output_dir : str
            Directory for saving PARC artefacts.
        """
        self.probe_budget = probe_budget
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.de_max_iter = de_max_iter
        self.rng = np.random.default_rng(random_state)
        self.criterion_evaluator = criterion_evaluator or self._default_evaluator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._gps = None        # GP surrogates fitted in Step 1
        self._mu = None         # probe-set means for normalisation
        self._sigma = None      # probe-set stds for normalisation
        self._M = None          # interdependency matrix

    # ─────────────────────────────────────────────────────────────────────────
    # Step 0 — Threshold derivation (reference-anchored)
    # ─────────────────────────────────────────────────────────────────────────

    def derive_thresholds(self, dataset_stats: Optional[Dict] = None) -> Dict[str, float]:
        """
        Anchor each τ_i to a reference distribution derived from the target dataset.

        For "<=" criteria, τ_i is the upper (1-α) quantile.
        For ">=" criteria, τ_i is the lower α quantile.

        Parameters
        ----------
        dataset_stats : dict, optional
            Keys (all optional): ``n_val``, ``{criterion}_mean``,
            ``{criterion}_std`` to anchor each distribution.
        """
        logger.info("PARC Step 0: Threshold derivation (reference-anchored) …")
        ds = dataset_stats or {}
        thresholds: Dict[str, float] = {}

        def _sample(mean_key: str, std_key: str, default_mean: float, default_std: float) -> np.ndarray:
            return self.rng.normal(
                ds.get(mean_key, default_mean),
                ds.get(std_key, default_std),
                size=self.n_bootstrap,
            )

        # MIA — null distribution is random guessing Binomial(n, 0.5)/n
        n_val = int(ds.get("n_val", 1000))
        null_mia = self.rng.binomial(n_val, 0.5, size=self.n_bootstrap) / n_val
        thresholds["mia_accuracy"] = float(np.quantile(null_mia, 1.0 - self.alpha))

        # dp_epsilon (<= type → upper quantile)
        thresholds["dp_epsilon"] = float(
            np.quantile(_sample("dp_epsilon_mean", "dp_epsilon_std", 5.0, 0.3), 1.0 - self.alpha)
        )

        # dp_delta (<= type → upper quantile)
        thresholds["dp_delta"] = float(
            np.quantile(_sample("dp_delta_mean", "dp_delta_std", 1e-5, 1e-6), 1.0 - self.alpha)
        )

        # dcr_sigma (>= type → lower quantile)
        thresholds["dcr_sigma"] = float(
            np.quantile(_sample("dcr_mean", "dcr_std", 5.0, 0.5), self.alpha)
        )

        # delta_tpr (<= type → upper quantile)
        thresholds["delta_tpr"] = float(
            np.quantile(_sample("delta_tpr_mean", "delta_tpr_std", 0.05, 0.01), 1.0 - self.alpha)
        )

        # demographic_parity (>= type → lower quantile)
        thresholds["demographic_parity"] = float(
            np.quantile(_sample("dem_parity_mean", "dem_parity_std", 0.80, 0.02), self.alpha)
        )

        # sds_pct (<= type → upper quantile)
        thresholds["sds_pct"] = float(
            np.quantile(_sample("sds_mean", "sds_std", 15.0, 1.5), 1.0 - self.alpha)
        )

        # mt_violation_rate (<= type → upper quantile)
        thresholds["mt_violation_rate"] = float(
            np.quantile(_sample("mt_mean", "mt_std", 10.0, 1.0), 1.0 - self.alpha)
        )

        # dcs_score (>= type → lower quantile) — bootstrap CI on documentation checklist
        thresholds["dcs_score"] = float(
            np.quantile(_sample("dcs_mean", "dcs_std", 90.0, 2.0), self.alpha)
        )

        # gradcam_iou (>= type → lower quantile) — bootstrap CI on gold-standard masks
        thresholds["gradcam_iou"] = float(
            np.quantile(_sample("gradcam_mean", "gradcam_std", 0.60, 0.03), self.alpha)
        )

        # nir_red_attribution_pct (>= type → lower quantile) — bootstrap CI
        thresholds["nir_red_attribution_pct"] = float(
            np.quantile(_sample("nir_red_mean", "nir_red_std", 50.0, 2.0), self.alpha)
        )

        # tac_score (>= type → lower quantile) — bootstrap CI on phenological windows
        thresholds["tac_score"] = float(
            np.quantile(_sample("tac_mean", "tac_std", 0.80, 0.02), self.alpha)
        )

        for name in self.CRITERIA_NAMES:
            op = self.THRESHOLD_OPS[name]
            logger.info(f"  τ({name:35s}) {op} {thresholds[name]:.6g}")

        return thresholds

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 — Interdependency profiling
    # ─────────────────────────────────────────────────────────────────────────

    def profile_interdependencies(
        self,
        thresholds: Dict[str, float],
    ) -> np.ndarray:
        """
        Latin-hypercube probe → GP surrogates → Interdependency Matrix M.

        M[i, j] = ∂c̃_i/∂τ_j |_{θ₀}  (central FD on GP posterior mean)

        Side effects: populates ``self._gps``, ``self._mu``, ``self._sigma``.

        Returns
        -------
        M : ndarray, shape (12, 12)
        """
        logger.info(
            f"PARC Step 1: Generating {self.probe_budget} LHS configurations …"
        )

        lhs_unit = self._lhs_sample(self.probe_budget)
        probe_configs = [self._unit_to_config(lhs_unit[k]) for k in range(self.probe_budget)]

        probe_criteria: List[Dict] = []
        for k, cfg in enumerate(probe_configs):
            logger.info(f"  Probe {k + 1:02d}/{self.probe_budget}: {cfg}")
            probe_criteria.append(self.criterion_evaluator(cfg))

        X = lhs_unit  # (K, 5)
        Y = np.array(
            [[pc[n] for n in self.CRITERIA_NAMES] for pc in probe_criteria]
        )  # (K, 12)

        logger.info("  Fitting Gaussian Process surrogates (Matérn-5/2 kernel) …")
        self._mu = Y.mean(axis=0)
        self._sigma = Y.std(axis=0) + 1e-10
        self._gps = self._fit_gp_surrogates(X, Y)

        logger.info("  Computing Interdependency Matrix at θ₀ …")
        theta0_unit = self._config_to_unit(self.PRIOR_CONFIG)
        M = self._compute_M(theta0_unit)
        self._M = M

        self._log_matrix(M)
        return M

    def _lhs_sample(self, n: int) -> np.ndarray:
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=5, seed=int(self.rng.integers(0, 2**31)))
        return sampler.random(n=n)

    def _unit_to_config(self, unit: np.ndarray) -> Dict[str, float]:
        names = list(self.CONFIG_BOUNDS.keys())
        return {
            name: float(self.CONFIG_BOUNDS[name][0] + unit[i] * (
                self.CONFIG_BOUNDS[name][1] - self.CONFIG_BOUNDS[name][0]
            ))
            for i, name in enumerate(names)
        }

    def _config_to_unit(self, config: Dict[str, float]) -> np.ndarray:
        names = list(self.CONFIG_BOUNDS.keys())
        return np.array([
            (config[name] - self.CONFIG_BOUNDS[name][0]) /
            (self.CONFIG_BOUNDS[name][1] - self.CONFIG_BOUNDS[name][0])
            for name in names
        ])

    def _fit_gp_surrogates(self, X: np.ndarray, Y: np.ndarray):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        gps = []
        for j in range(Y.shape[1]):
            gp = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                n_restarts_optimizer=3,
                normalize_y=True,
                random_state=42,
            )
            gp.fit(X, Y[:, j])
            gps.append(gp)
        return gps

    def _gp_predict_raw(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate raw GP posterior means at θ (unit hypercube)."""
        x = theta.reshape(1, -1)
        return np.array([gp.predict(x)[0] for gp in self._gps])

    def _gp_predict_normalised(self, theta: np.ndarray) -> np.ndarray:
        """Return (ĉ(θ) − μ) / σ."""
        raw = self._gp_predict_raw(theta)
        return (raw - self._mu) / self._sigma

    def _compute_M(self, theta0: np.ndarray, h: float = 1e-3) -> np.ndarray:
        """
        Compute M[i, j] = ∂c̃_i/∂τ_j via central finite differences on the GP.

        Perturbation direction for column j is along the config dimension most
        directly linked to criterion j.
        """
        n = len(self.CRITERIA_NAMES)
        M = np.zeros((n, n))

        for j in range(n):
            param_idx = self._CRIT_TO_PARAM_IDX.get(j, -1)
            if param_idx < 0:
                continue
            delta = np.zeros(5)
            delta[param_idx] = h
            c_plus = self._gp_predict_normalised(np.clip(theta0 + delta, 0.0, 1.0))
            c_minus = self._gp_predict_normalised(np.clip(theta0 - delta, 0.0, 1.0))
            M[:, j] = (c_plus - c_minus) / (2.0 * h)

        return M

    def _log_matrix(self, M: np.ndarray) -> None:
        n = len(self.CRITERIA_NAMES)
        short = [n[:6] for n in self.CRITERIA_NAMES]
        header = " " * 12 + "  ".join(f"{s:>6}" for s in short)
        logger.info("  R-AI Interdependency Matrix (row=affected, col=driver):")
        logger.info(header)
        for i, row_name in enumerate(self.CRITERIA_NAMES):
            row_str = "  ".join(f"{M[i, j]:+6.3f}" for j in range(n))
            logger.info(f"  {row_name[:10]:>10}: {row_str}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 — Adaptive phase reordering
    # ─────────────────────────────────────────────────────────────────────────

    def reorder_phases(self, M: np.ndarray) -> List[int]:
        """
        Find the phase ordering π* that minimises total destructive interference.

            J(G_a, G_b) = Σ_{i∈G_a} Σ_{j∈G_b} max(0, −M[i,j])
            π* = argmin_{π} Σ_{a<b} J(G_{π(a)}, G_{π(b)})

        Evaluated by exhaustive enumeration over all 4! = 24 orderings.
        """
        logger.info("PARC Step 2: Adaptive phase reordering (exhaustive, 24 orderings) …")

        groups = [self.PHASE_GROUPS[g] for g in range(4)]

        def interference(a: int, b: int) -> float:
            return sum(
                max(0.0, float(-M[i, j]))
                for i in groups[a]
                for j in groups[b]
            )

        best_order: List[int] = [0, 1, 2, 3]
        best_score = float("inf")

        for perm in permutations(range(4)):
            score = sum(
                interference(perm[a], perm[b])
                for a in range(4)
                for b in range(a + 1, 4)
            )
            if score < best_score:
                best_score = score
                best_order = list(perm)

        order_names = [self.PHASE_NAMES[i] for i in best_order]
        logger.info(
            f"  Optimal ordering: {' → '.join(order_names)}  "
            f"(total interference = {best_score:.4f})"
        )
        return best_order

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 — Max-min slack selection
    # ─────────────────────────────────────────────────────────────────────────

    def select_optimal_config(
        self,
        thresholds: Dict[str, float],
        M: np.ndarray,
    ) -> Tuple[Dict[str, float], Dict[str, float], bool, Optional[Tuple[int, int]]]:
        """
        Maximise the minimum normalised slack on the GP surrogate via
        differential evolution (~200 evaluations, no extra model training).

        Returns
        -------
        optimal_config : dict  (empty if infeasible)
        slack_certificate : dict
        infeasible : bool
        infeasibility_pair : (i, j) or None
        """
        logger.info(
            "PARC Step 3: Max-min slack selection "
            f"(differential evolution, max_iter={self.de_max_iter}) …"
        )
        from scipy.optimize import differential_evolution

        tau = np.array([thresholds[n] for n in self.CRITERIA_NAMES])

        def normalised_slack(theta: np.ndarray) -> np.ndarray:
            raw = self._gp_predict_raw(theta)
            slacks = np.empty(12)
            for i, name in enumerate(self.CRITERIA_NAMES):
                denom = abs(tau[i]) + 1e-12
                if self.THRESHOLD_OPS[name] == "<=":
                    slacks[i] = (tau[i] - raw[i]) / denom
                else:
                    slacks[i] = (raw[i] - tau[i]) / denom
            return slacks

        def neg_min_slack(u: np.ndarray) -> float:
            return -float(normalised_slack(u).min())

        result = differential_evolution(
            neg_min_slack,
            bounds=[(0.0, 1.0)] * 5,
            maxiter=self.de_max_iter,
            seed=42,
            tol=1e-4,
            workers=1,
        )

        best_min_slack = -result.fun

        if best_min_slack < 0.0:
            # Infeasible — report the most conflicting criterion pair
            flat_idx = int(np.argmin(M))
            pair: Tuple[int, int] = (flat_idx // 12, flat_idx % 12)
            logger.warning(
                "PARC: GP surrogate has no feasible configuration. "
                f"Binding conflict: ({self.CRITERIA_NAMES[pair[0]]}, "
                f"{self.CRITERIA_NAMES[pair[1]]})"
            )
            return {}, {}, True, pair

        theta_star = self._unit_to_config(result.x)
        slacks_star = normalised_slack(result.x)
        slack_cert = {
            name: float(slacks_star[i])
            for i, name in enumerate(self.CRITERIA_NAMES)
        }

        logger.info(f"  θ* = {theta_star}")
        logger.info("  Slack certificate (s̃_i = slack_i / τ_i):")
        for name, s in slack_cert.items():
            status = "PASS" if s >= 0 else "FAIL"
            logger.info(f"    {name:35s}: {s:+.4f}  [{status}]")

        return theta_star, slack_cert, False, None

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, dataset_stats: Optional[Dict] = None) -> PARCResult:
        """
        Execute all four PARC steps and return a ``PARCResult``.

        Parameters
        ----------
        dataset_stats : dict, optional
            Dataset-specific statistics to anchor Step 0 distributions.
        """
        logger.info("=" * 60)
        logger.info("PARC: Pareto-Adaptive R-AI Compliance — starting")
        logger.info("=" * 60)

        thresholds = self.derive_thresholds(dataset_stats)
        M = self.profile_interdependencies(thresholds)
        phase_ordering = self.reorder_phases(M)
        optimal_config, slack_cert, infeasible, inf_pair = self.select_optimal_config(
            thresholds, M
        )

        result = PARCResult(
            thresholds=thresholds,
            interdependency_matrix=M,
            criterion_names=self.CRITERIA_NAMES,
            phase_ordering=phase_ordering,
            phase_ordering_names=[self.PHASE_NAMES[i] for i in phase_ordering],
            optimal_config=optimal_config,
            slack_certificate=slack_cert,
            infeasible=infeasible,
            infeasibility_pair=inf_pair,
        )

        self._save_result(result)

        if infeasible:
            i, j = result.infeasibility_pair
            logger.warning(
                f"PARC infeasibility certificate — binding conflict: "
                f"({self.CRITERIA_NAMES[i]}, {self.CRITERIA_NAMES[j]})"
            )
        else:
            order_str = " → ".join(result.phase_ordering_names)
            logger.info(f"PARC complete.  Phase order: {order_str}")

        logger.info("=" * 60)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Built-in synthetic criterion evaluator (demo / unit-test fallback)
    # ─────────────────────────────────────────────────────────────────────────

    def _default_evaluator(self, config: Dict[str, float]) -> Dict[str, float]:
        """
        Synthetic criterion evaluator encoding known monotonic relationships
        between the five intervention parameters and the 12 RAI criteria.
        Used for demonstration and unit tests when no real model is available.
        """
        eps = config["epsilon"]
        l1  = config["lambda1"]
        l2  = config["lambda2"]
        w_mt  = config["w_mt"]
        w_tac = config["w_tac"]

        def _n(scale: float) -> float:
            return float(self.rng.normal(0.0, scale))

        mia      = float(np.clip(0.55 - 0.008 * eps + _n(0.010), 0.46, 0.62))
        dp_eps   = float(eps)
        dp_delta = float(np.clip(8.5e-6 - 3e-7 * eps + _n(2e-7), 1e-7, 1.5e-5))
        dcr      = float(np.clip(4.5  + 0.15  * eps + _n(0.20),  3.0,  8.0))

        delta_tpr = float(np.clip(0.09 - 0.09 * l1 + _n(0.005), 0.01, 0.18))
        dem_par   = float(np.clip(0.70 + 0.30 * l1 + _n(0.010), 0.55, 0.99))
        sds       = float(np.clip(20.0 - 12.0 * l2 + _n(0.50),  4.0,  28.0))

        mt_viol  = float(np.clip(14.0 - 4.0  * w_mt  + _n(0.5), 2.0,  22.0))
        dcs      = float(np.clip(87.0 + 3.0  * w_mt  + _n(0.5), 78.0, 99.0))

        gradcam  = float(np.clip(0.55 + 0.05  * w_tac + _n(0.010), 0.40, 0.85))
        nir_red  = float(np.clip(46.0 + 4.0   * w_tac + _n(0.50),  38.0, 68.0))
        tac      = float(np.clip(0.74 + 0.06  * w_tac + _n(0.010), 0.58, 0.97))

        return {
            "mia_accuracy":            mia,
            "dp_epsilon":              dp_eps,
            "dp_delta":                dp_delta,
            "dcr_sigma":               dcr,
            "delta_tpr":               delta_tpr,
            "demographic_parity":      dem_par,
            "sds_pct":                 sds,
            "mt_violation_rate":       mt_viol,
            "dcs_score":               dcs,
            "gradcam_iou":             gradcam,
            "nir_red_attribution_pct": nir_red,
            "tac_score":               tac,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _save_result(self, result: PARCResult) -> None:
        out = {
            "thresholds": result.thresholds,
            "interdependency_matrix": result.interdependency_matrix.tolist(),
            "criterion_names": result.criterion_names,
            "phase_ordering": result.phase_ordering,
            "phase_ordering_names": result.phase_ordering_names,
            "optimal_config": result.optimal_config,
            "slack_certificate": result.slack_certificate,
            "infeasible": result.infeasible,
            "infeasibility_pair": (
                list(result.infeasibility_pair)
                if result.infeasibility_pair else None
            ),
        }
        path = self.output_dir / "parc_result.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"PARC result saved → {path}")
