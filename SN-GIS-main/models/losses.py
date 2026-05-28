import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class FairnessPenalisedLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_subgroups: int,
        lambda1: float = 0.3,
        lambda2: float = 0.2,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_subgroups = num_subgroups
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        subgroups: torch.Tensor,
    ) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        tpr_penalty = self._tpr_disparity_penalty(logits, targets, subgroups)
        sds_penalty = self._spatial_disparity_penalty(logits, targets, subgroups)
        return ce_loss + self.lambda1 * tpr_penalty + self.lambda2 * sds_penalty

    def _tpr_disparity_penalty(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        subgroups: torch.Tensor,
    ) -> torch.Tensor:
        preds = logits.argmax(dim=1)
        tprs = []
        for g in range(self.num_subgroups):
            mask = subgroups == g
            if mask.sum() == 0:
                continue
            g_preds = preds[mask].float()
            g_targets = targets[mask].float()
            correct = (g_preds == g_targets).float().mean()
            tprs.append(correct)
        if len(tprs) < 2:
            return torch.tensor(0.0, device=logits.device)
        tprs_t = torch.stack(tprs)
        return tprs_t.max() - tprs_t.min()

    def _spatial_disparity_penalty(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        subgroups: torch.Tensor,
    ) -> torch.Tensor:
        preds = logits.argmax(dim=1)
        global_acc = (preds == targets).float().mean()
        accs = []
        for g in range(self.num_subgroups):
            mask = subgroups == g
            if mask.sum() == 0:
                continue
            acc = (preds[mask] == targets[mask]).float().mean()
            accs.append(acc)
        if not accs:
            return torch.tensor(0.0, device=logits.device)
        min_acc = torch.stack(accs).min()
        return F.relu(global_acc - min_acc - 0.15)
