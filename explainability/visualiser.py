import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


class ExplainabilityVisualiser:
    def __init__(self, output_dir: str = "outputs/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_gradcam_iou(
        self,
        per_class_iou: Dict[str, float],
        threshold: float = 0.60,
        title: str = "GradCAM++ IoU per Class",
        filename: str = "gradcam_iou.png",
    ) -> None:
        classes = list(per_class_iou.keys())
        values = list(per_class_iou.values())
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#4a90c4" if v >= threshold else "#d9534f" for v in values]
        ax.bar(classes, values, color=colors, alpha=0.85)
        ax.axhline(threshold, color="orange", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("GradCAM++ IoU")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_spectral_attribution(
        self,
        band_shares: Dict[str, float],
        nir_bands: List[str],
        red_band: str,
        threshold: float = 50.0,
        title: str = "Spectral Attribution (Integrated Gradients)",
        filename: str = "spectral_attribution.png",
    ) -> None:
        bands = list(band_shares.keys())
        values = list(band_shares.values())
        nir_red = set(nir_bands + [red_band])
        colors = ["#2ecc71" if b in nir_red else "#95a5a6" for b in bands]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(bands, values, color=colors, alpha=0.85)
        total_nir_red = sum(band_shares.get(b, 0) for b in nir_red)
        ax.set_ylabel("Attribution Share (%)")
        ax.set_title(f"{title}\nNIR+Red total: {total_nir_red:.1f}% (threshold ≥ {threshold}%)")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_tac(
        self,
        per_class_tac: Dict[str, float],
        threshold: float = 0.80,
        title: str = "Temporal Attribution Consistency (TAC)",
        filename: str = "tac.png",
    ) -> None:
        classes = list(per_class_tac.keys())
        values = list(per_class_tac.values())
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#4a90c4" if v >= threshold else "#d9534f" for v in values]
        ax.barh(classes, values, color=colors, alpha=0.85)
        ax.axvline(threshold, color="orange", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("TAC Score")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_mt_violations(
        self,
        mr_rates: Dict[str, float],
        threshold: float = 10.0,
        title: str = "Metamorphic Testing Violation Rates",
        filename: str = "mt_violations.png",
    ) -> None:
        mrs = [k for k in mr_rates if k != "overall"]
        values = [mr_rates[k] for k in mrs]
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#d9534f" if v > threshold else "#2ecc71" for v in values]
        ax.bar(mrs, values, color=colors, alpha=0.85)
        ax.axhline(threshold, color="orange", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold}%)")
        ax.set_ylabel("Violation Rate (%)")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_fairness(
        self,
        subgroup_accuracies_baseline: Dict[str, float],
        subgroup_accuracies_rai: Dict[str, float],
        threshold: float = 80.0,
        title: str = "Per-Subgroup Accuracy Before and After RAI",
        filename: str = "fairness.png",
    ) -> None:
        groups = list(subgroup_accuracies_baseline.keys())
        base_vals = [subgroup_accuracies_baseline[g] * 100 for g in groups]
        rai_vals = [subgroup_accuracies_rai.get(g, 0) * 100 for g in groups]
        x = np.arange(len(groups))
        w = 0.35
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x - w / 2, base_vals, w, color="#d9534f", alpha=0.85, label="Baseline")
        ax.bar(x + w / 2, rai_vals, w, color="#2ecc71", alpha=0.85, label="RAI Approach")
        ax.axhline(threshold, color="orange", linestyle="--", linewidth=1.5, label=f"Equity threshold ({threshold}%)")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=15, ha="right")
        ax.set_ylabel("Classification Accuracy (%)")
        ax.set_ylim(55, 100)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_radar(
        self,
        metrics_baseline: Dict[str, float],
        metrics_rai: Dict[str, float],
        title: str = "RAI Compliance Radar Chart",
        filename: str = "radar.png",
    ) -> None:
        labels = list(metrics_baseline.keys())
        N = len(labels)
        if N < 3:
            return

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        base_vals = list(metrics_baseline.values()) + [list(metrics_baseline.values())[0]]
        rai_vals = list(metrics_rai.values()) + [list(metrics_rai.values())[0]]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.plot(angles, [1.0] * len(angles), "k--", linewidth=0.8, alpha=0.4)
        ax.plot(angles, base_vals, "o-", color="#d9534f", linewidth=1.8, markersize=4, label="Baseline")
        ax.fill(angles, base_vals, alpha=0.12, color="#d9534f")
        ax.plot(angles, rai_vals, "s-", color="#2c5f8a", linewidth=1.8, markersize=4, label="RAI Approach")
        ax.fill(angles, rai_vals, alpha=0.18, color="#4a90c4")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0, 1.6)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
