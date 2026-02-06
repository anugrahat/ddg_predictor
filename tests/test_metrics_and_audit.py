from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.prepare import leakage_audit
from src.eval.metrics import evaluate


def test_leakage_audit_detects_overlap(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    heldout = tmp_path / "heldout.jsonl"

    train.write_text(
        "\n".join(
            [
                json.dumps({"cath_superfamily": "1.2.3.4"}),
                json.dumps({"cath_superfamily": "2.3.4.5"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    heldout.write_text(
        "\n".join(
            [
                json.dumps({"cath_superfamily": "2.3.4.5"}),
                json.dumps({"cath_superfamily": "7.8.9.0"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rep = leakage_audit(str(train), str(heldout))
    assert rep["overlap_count"] == 1


def test_metrics_reproduce_expected_values() -> None:
    y = np.asarray([0.1, 0.3, -0.2, 0.5, -0.4, 0.6], dtype=np.float64)
    p = np.asarray([0.11, 0.28, -0.18, 0.49, -0.35, 0.59], dtype=np.float64)
    g = np.asarray([0, 0, 1, 1, 2, 2])
    m = evaluate(p, y, g)

    assert 0.98 <= m["PearsonR"] <= 1.0
    assert 0.94 <= m["SpearmanR"] <= 1.0
    assert m["RMSE"] < 0.06
    assert m["MAE"] < 0.05
