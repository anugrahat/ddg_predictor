from __future__ import annotations

import json
from pathlib import Path

from src.train.engine import TrainConfig, evaluate_checkpoint, train_model
from tests.helpers import write_dummy_dataset


def test_training_smoke_and_checkpoint_eval(tmp_path: Path) -> None:
    train_jsonl, val_jsonl = write_dummy_dataset(tmp_path, n=12)
    out_dir = tmp_path / "run1"
    cfg = TrainConfig(
        train_jsonl=str(train_jsonl),
        val_jsonl=str(val_jsonl),
        out_dir=str(out_dir),
        seed=11,
        epochs=1,
        batch_size=2,
        grad_accum_steps=1,
        patience=1,
        amp=False,
        foldx_dim=3,
        use_foldx_branch=True,
    )
    result = train_model(cfg)
    assert "best_val_pearson" in result
    assert (out_dir / "best.pt").exists()

    metrics = evaluate_checkpoint(str(out_dir / "best.pt"), str(val_jsonl), batch_size=2)
    assert "PearsonR" in metrics


def test_fixed_seed_regression_band(tmp_path: Path) -> None:
    train_jsonl, val_jsonl = write_dummy_dataset(tmp_path, n=12)

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    cfg_a = TrainConfig(
        train_jsonl=str(train_jsonl),
        val_jsonl=str(val_jsonl),
        out_dir=str(out_a),
        seed=22,
        epochs=1,
        batch_size=2,
        grad_accum_steps=1,
        patience=1,
        amp=False,
        foldx_dim=3,
        use_foldx_branch=True,
    )
    cfg_b = TrainConfig(
        train_jsonl=str(train_jsonl),
        val_jsonl=str(val_jsonl),
        out_dir=str(out_b),
        seed=22,
        epochs=1,
        batch_size=2,
        grad_accum_steps=1,
        patience=1,
        amp=False,
        foldx_dim=3,
        use_foldx_branch=True,
    )

    r1 = train_model(cfg_a)
    r2 = train_model(cfg_b)

    d = abs(float(r1["best_val_pearson"]) - float(r2["best_val_pearson"]))
    assert d <= 1e-6
