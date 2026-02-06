"""Training and evaluation loops."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PatchDataset, collate_batch
from src.eval.metrics import evaluate
from src.models.ig_ddg import IGDDG, IGDDGConfig
from src.train.losses import LossConfig, compute_loss
from src.utils import ensure_dir, set_seed, torch_device


@dataclass
class TrainConfig:
    train_jsonl: str
    val_jsonl: str
    out_dir: str
    seed: int = 42
    epochs: int = 120
    batch_size: int = 4
    grad_accum_steps: int = 6
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    amp: bool = True
    patience: int = 20
    foldx_dim: int = 32
    use_foldx_branch: bool = True
    use_inter_chain_attention: bool = True
    use_film: bool = True


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _build_loader(path: str, batch_size: int, shuffle: bool) -> DataLoader:
    ds = PatchDataset(path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch, num_workers=0)


def _run_eval(model: IGDDG, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    preds = []
    labels = []
    groups = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            out = model(batch)
            pred = out["ddg_mean"].detach().cpu().numpy()
            y = batch["ddg"].detach().cpu().numpy()
            g = batch["ppi_group"].detach().cpu().numpy()
            preds.extend(pred.tolist())
            labels.extend(y.tolist())
            groups.extend(g.tolist())
    return evaluate(np.asarray(preds), np.asarray(labels), np.asarray(groups))


def train_model(cfg: TrainConfig, loss_cfg: LossConfig | None = None) -> dict:
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)
    device = torch_device(prefer_cuda=True)

    train_loader = _build_loader(cfg.train_jsonl, cfg.batch_size, shuffle=True)
    val_loader = _build_loader(cfg.val_jsonl, cfg.batch_size, shuffle=False)

    model_cfg = IGDDGConfig(
        foldx_dim=cfg.foldx_dim,
        use_foldx_branch=cfg.use_foldx_branch,
        use_inter_chain_attention=cfg.use_inter_chain_attention,
        use_film=cfg.use_film,
    )
    model = IGDDG(model_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, math.ceil(len(train_loader) / cfg.grad_accum_steps) * cfg.epochs)
    warmup_steps = max(1, int(total_steps * cfg.warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    best_metric = -1e9
    best_epoch = -1
    no_improve = 0
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}"), start=1):
            batch = _move_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                out = model(batch)
                loss_dict = compute_loss(out, batch, cfg=loss_cfg)
                loss = loss_dict["loss"] / cfg.grad_accum_steps

            scaler.scale(loss).backward()
            running += float(loss_dict["loss"].detach().cpu())

            if step % cfg.grad_accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

        val_metrics = _run_eval(model, val_loader, device)
        score = float(val_metrics["PearsonR"])
        if not np.isfinite(score):
            score = -1e9

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val": val_metrics,
            "cfg": cfg.__dict__,
        }
        torch.save(ckpt, Path(cfg.out_dir) / "last.pt")

        if score > best_metric or best_epoch == -1:
            best_metric = score
            best_epoch = epoch
            no_improve = 0
            torch.save(ckpt, Path(cfg.out_dir) / "best.pt")
        else:
            no_improve += 1

        if no_improve >= cfg.patience:
            break

    result = {
        "best_epoch": best_epoch,
        "best_val_pearson": best_metric,
    }
    with open(Path(cfg.out_dir) / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def evaluate_checkpoint(checkpoint: str, jsonl_path: str, batch_size: int = 8) -> dict:
    device = torch_device(prefer_cuda=True)
    ckpt = torch.load(checkpoint, map_location=device)
    cfg_dict = ckpt.get("cfg", {})
    model_cfg = IGDDGConfig(
        foldx_dim=cfg_dict.get("foldx_dim", 32),
        use_foldx_branch=cfg_dict.get("use_foldx_branch", True),
        use_inter_chain_attention=cfg_dict.get("use_inter_chain_attention", True),
        use_film=cfg_dict.get("use_film", True),
    )
    model = IGDDG(model_cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    loader = _build_loader(jsonl_path, batch_size=batch_size, shuffle=False)
    metrics = _run_eval(model, loader, device)
    return metrics
