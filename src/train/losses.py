"""Multi-objective loss definitions for IG-DDG."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossConfig:
    w_reg: float = 1.0
    w_rank: float = 0.25
    w_sign: float = 0.1
    w_reverse: float = 0.1
    w_cath_aux: float = 0.05
    w_uncertainty: float = 0.05
    ranking_margin: float = 0.0


def _pairwise_ranking_loss(pred: torch.Tensor, target: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    loss = pred.new_tensor(0.0)
    pairs = 0
    for g in torch.unique(group):
        idx = torch.where(group == g)[0]
        if idx.numel() < 2:
            continue
        p = pred[idx]
        t = target[idx]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                if t[i] == t[j]:
                    continue
                sign = torch.sign(t[i] - t[j])
                loss = loss + F.softplus(-(p[i] - p[j]) * sign)
                pairs += 1
    if pairs == 0:
        return pred.new_tensor(0.0)
    return loss / pairs


def compute_loss(outputs: dict, batch: dict, cfg: LossConfig | None = None) -> dict[str, torch.Tensor]:
    cfg = cfg or LossConfig()

    pred = outputs["ddg_mean"]
    logvar = outputs["ddg_logvar"]
    sign_logit = outputs["sign_logit"]

    target = batch["ddg"].float()
    ppi_group = batch["ppi_group"].long()
    sign_target = batch["sign_target"].float()

    reg = F.smooth_l1_loss(pred, target)
    rank = _pairwise_ranking_loss(pred, target, ppi_group)
    sign = F.binary_cross_entropy_with_logits(sign_logit, sign_target)

    # Reverse-consistency optional hook.
    reverse = pred.new_tensor(0.0)
    rev = batch.get("reverse_ddg_mean", None)
    if rev is not None:
        reverse = F.smooth_l1_loss(pred, -rev.float())

    cath_aux = pred.new_tensor(0.0)
    cath_label = batch.get("cath_label", None)
    cath_logit = outputs["aux_outputs"].get("cath_logit", None)
    if cath_label is not None and cath_logit is not None:
        cath_aux = F.cross_entropy(cath_logit, cath_label.long())

    uncertainty = (0.5 * torch.exp(-logvar) * (pred - target) ** 2 + 0.5 * logvar).mean()

    total = (
        cfg.w_reg * reg
        + cfg.w_rank * rank
        + cfg.w_sign * sign
        + cfg.w_reverse * reverse
        + cfg.w_cath_aux * cath_aux
        + cfg.w_uncertainty * uncertainty
    )

    return {
        "loss": total,
        "reg": reg.detach(),
        "rank": rank.detach(),
        "sign": sign.detach(),
        "reverse": reverse.detach(),
        "cath_aux": cath_aux.detach(),
        "uncertainty": uncertainty.detach(),
    }
