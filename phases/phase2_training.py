import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from utils.logger import get_logger
from utils.metrics import compute_miou, compute_macro_f1
from utils.checkpoint import save_checkpoint

logger = get_logger(__name__)


@dataclass
class FairnessResult:
    delta_tpr: float
    demographic_parity: float
    sds: float
    subgroup_accuracies: Dict[str, float]
    passed: bool
    details: Dict = field(default_factory=dict)


class Phase2Training:
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        num_subgroups: int,
        lambda1: float = 0.3,
        lambda2: float = 0.2,
        lr: float = 1e-4,
        lr_min: float = 1e-6,
        epochs: int = 150,
        patience: int = 15,
        grad_clip: float = 1.0,
        device: torch.device = None,
        output_dir: str = "outputs",
        use_dp: bool = False,
        dp_epsilon: float = 4.1,
        dp_delta: float = 8.3e-6,
        delta_tpr_threshold: float = 0.05,
        demographic_parity_threshold: float = 0.80,
        sds_threshold: float = 15.0,
    ):
        self.model = model
        self.num_classes = num_classes
        self.num_subgroups = num_subgroups
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lr = lr
        self.lr_min = lr_min
        self.epochs = epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.use_dp = use_dp
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.delta_tpr_threshold = delta_tpr_threshold
        self.demographic_parity_threshold = demographic_parity_threshold
        self.sds_threshold = sds_threshold

        self.model = self.model.to(self.device)
        from models.losses import FairnessPenalisedLoss
        self.criterion = FairnessPenalisedLoss(num_classes, num_subgroups, lambda1, lambda2)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=lr_min)

        if use_dp:
            self._wrap_dp()

    def _wrap_dp(self):
        try:
            from opacus import PrivacyEngine
            privacy_engine = PrivacyEngine()
            self.model, self.optimizer, _ = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=None,
                target_epsilon=self.dp_epsilon,
                target_delta=self.dp_delta,
                epochs=self.epochs,
                max_grad_norm=self.grad_clip,
            )
            logger.info(f"DP-SGD enabled: epsilon={self.dp_epsilon}, delta={self.dp_delta}")
        except ImportError:
            logger.warning("Opacus not installed. Training without DP-SGD.")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_path: str = None,
    ) -> Dict:
        best_val_metric = -1.0
        patience_counter = 0
        history = {"train_loss": [], "val_miou": [], "val_f1": []}

        for epoch in range(self.epochs):
            train_loss = self._train_epoch(train_loader)
            val_metrics = self._evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_miou"].append(val_metrics["miou"])
            history["val_f1"].append(val_metrics["macro_f1"])

            self.scheduler.step()

            val_primary = val_metrics["miou"]
            if val_primary > best_val_metric:
                best_val_metric = val_primary
                patience_counter = 0
                if checkpoint_path:
                    save_checkpoint({
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "val_miou": best_val_metric,
                    }, checkpoint_path)
            else:
                patience_counter += 1

            logger.info(
                f"Epoch {epoch+1:03d}/{self.epochs} | loss={train_loss:.4f} | "
                f"val_miou={val_metrics['miou']:.4f} | val_f1={val_metrics['macro_f1']:.4f}"
            )

            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}.")
                break

        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for x, y, subgroups in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            subgroups = subgroups.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            y_flat = y.view(-1) if y.dim() > 1 else y
            logits_2d = logits if logits.dim() == 2 else logits.view(-1, self.num_classes)
            loss = self.criterion(logits_2d, y_flat, subgroups)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    def _evaluate(self, loader: DataLoader) -> Dict:
        self.model.eval()
        all_preds, all_targets, all_subgroups = [], [], []
        with torch.no_grad():
            for x, y, subgroups in loader:
                x = x.to(self.device)
                logits = self.model(x)
                preds = logits.argmax(dim=-1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.numpy())
                all_subgroups.append(subgroups.numpy())

        preds_np = np.concatenate(all_preds).ravel()
        targets_np = np.concatenate(all_targets).ravel()
        subgroups_np = np.concatenate(all_subgroups).ravel()

        return {
            "miou": compute_miou(preds_np, targets_np, self.num_classes),
            "macro_f1": compute_macro_f1(preds_np, targets_np),
            "preds": preds_np,
            "targets": targets_np,
            "subgroups": subgroups_np,
        }

    def evaluate_fairness(self, loader: DataLoader) -> FairnessResult:
        metrics = self._evaluate(loader)
        preds = metrics["preds"]
        targets = metrics["targets"]
        subgroups = metrics["subgroups"]

        subgroup_accs = {}
        tprs = []
        for g in range(self.num_subgroups):
            mask = subgroups == g
            if mask.sum() == 0:
                continue
            acc = (preds[mask] == targets[mask]).mean()
            subgroup_accs[f"group_{g}"] = float(acc)
            tprs.append(float(acc))

        delta_tpr = max(tprs) - min(tprs) if len(tprs) >= 2 else 0.0
        dem_parity = min(tprs) / max(tprs) if max(tprs) > 0 else 1.0
        global_acc = (preds == targets).mean()
        sds = max(0.0, (global_acc - min(tprs)) * 100) if tprs else 0.0

        passed = (
            delta_tpr <= self.delta_tpr_threshold
            and dem_parity >= self.demographic_parity_threshold
            and sds <= self.sds_threshold
        )

        logger.info(f"Fairness — ΔTPR={delta_tpr:.4f}, DemParity={dem_parity:.4f}, SDS={sds:.2f}%")

        return FairnessResult(
            delta_tpr=delta_tpr,
            demographic_parity=dem_parity,
            sds=sds,
            subgroup_accuracies=subgroup_accs,
            passed=passed,
            details={"global_accuracy": float(global_acc)},
        )
