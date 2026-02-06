from __future__ import annotations

import torch

from src.data.dataset import collate_batch
from src.models.ig_ddg import IGDDG, IGDDGConfig
from tests.helpers import make_dummy_patch


def _make_batch(with_foldx: bool) -> dict:
    items = []
    for i in range(3):
        items.append(
            {
                "wt": make_dummy_patch(seed=i),
                "mt": make_dummy_patch(seed=i + 100),
                "ddg": float(i) * 0.1,
                "ppi_id": f"ppi_{i // 2}",
                "record_id": f"rec_{i}",
                "cath_label": i % 3,
                "foldx": [0.1, 0.2, 0.3] if with_foldx else None,
                "split_tag": "train",
                "cath_superfamily": "x",
            }
        )
    return collate_batch(items)


def test_forward_without_foldx() -> None:
    b = _make_batch(with_foldx=False)
    m = IGDDG(IGDDGConfig(use_foldx_branch=False, num_layers=2, hidden_dim=128))
    out = m(b)
    assert out["ddg_mean"].shape[0] == 3
    assert out["ddg_logvar"].shape[0] == 3
    assert out["sign_logit"].shape[0] == 3


def test_forward_with_foldx() -> None:
    b = _make_batch(with_foldx=True)
    m = IGDDG(IGDDGConfig(use_foldx_branch=True, num_layers=2, hidden_dim=128, foldx_dim=3))
    out = m(b)
    assert out["ddg_mean"].shape[0] == 3
    assert "foldx_gate" in out["aux_outputs"]


def test_reverse_pair_delta_is_antisymmetric() -> None:
    a = make_dummy_patch(seed=7)
    b = make_dummy_patch(seed=8)

    batch_fwd = collate_batch(
        [{
            "wt": a,
            "mt": b,
            "ddg": 0.3,
            "ppi_id": "ppi_0",
            "record_id": "fwd",
            "cath_label": 0,
            "foldx": None,
            "split_tag": "train",
            "cath_superfamily": "a",
        }]
    )
    batch_rev = collate_batch(
        [{
            "wt": b,
            "mt": a,
            "ddg": -0.3,
            "ppi_id": "ppi_0",
            "record_id": "rev",
            "cath_label": 0,
            "foldx": None,
            "split_tag": "train",
            "cath_superfamily": "a",
        }]
    )

    m = IGDDG(IGDDGConfig(use_foldx_branch=False, num_layers=2, hidden_dim=128))
    out_f = m(batch_fwd)
    out_r = m(batch_rev)

    df = out_f["aux_outputs"]["delta_pooled"]
    dr = out_r["aux_outputs"]["delta_pooled"]
    assert torch.allclose(df, -dr, atol=1e-4, rtol=1e-3)
