from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def make_dummy_patch(n: int = 24, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    aa_ids = rng.integers(0, 21, size=(n,), dtype=np.int64)
    esm = rng.normal(0, 1, size=(n, 1280)).astype(np.float32)
    scalars = rng.normal(0, 1, size=(n, 6)).astype(np.float32)
    atom_summary = rng.normal(0, 1, size=(n, 8)).astype(np.float32)
    mutation_mask = np.zeros((n,), dtype=np.float32)
    mutation_mask[:2] = 1.0
    interface_mask = np.zeros((n,), dtype=np.float32)
    interface_mask[: n // 2] = 1.0
    first_shell_mask = np.zeros((n,), dtype=np.float32)
    first_shell_mask[2:8] = 1.0
    chain_ids = np.zeros((n,), dtype=np.int64)
    chain_ids[n // 2 :] = 1

    edges = []
    for i in range(n):
        j = (i + 1) % n
        for t in [0, 1, 2]:
            feat = np.zeros((25,), dtype=np.float32)
            feat[:16] = rng.random(16)
            feat[16:19] = rng.normal(0, 1, size=(3,))
            feat[19:22] = [abs(i - j) / 50.0, float(chain_ids[i] != chain_ids[j]), 1.0]
            feat[22 + t] = 1.0
            edges.append((i, j, feat))

    edge_index = np.asarray([[e[0] for e in edges], [e[1] for e in edges]], dtype=np.int64)
    edge_feat = np.asarray([e[2] for e in edges], dtype=np.float32)

    return {
        "aa_ids": aa_ids,
        "esm": esm,
        "scalars": scalars,
        "atom_summary": atom_summary,
        "mutation_mask": mutation_mask,
        "interface_mask": interface_mask,
        "first_shell_mask": first_shell_mask,
        "chain_ids": chain_ids,
        "edge_index": edge_index,
        "edge_feat": edge_feat,
    }


def write_dummy_dataset(root: Path, n: int = 8) -> tuple[Path, Path]:
    patches = root / "patches"
    patches.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n):
        wt = make_dummy_patch(seed=i)
        mt = make_dummy_patch(seed=i + 100)
        wt_path = patches / f"{i}_wt.npz"
        mt_path = patches / f"{i}_mt.npz"
        np.savez_compressed(wt_path, **wt)
        np.savez_compressed(mt_path, **mt)
        rows.append(
            {
                "pdb_id": f"{i}_DUMY",
                "chain_id": "A",
                "residue_idx": 10 + i,
                "wt_aa": "A",
                "mt_aa": "V",
                "ddg": float((-1) ** i * (0.1 + i * 0.05)),
                "split_tag": "train" if i < int(0.75 * n) else "val",
                "ppi_id": f"ppi_{i // 2}",
                "cath_superfamily": f"{i // 3}",
                "wt_patch_path": str(wt_path),
                "mt_patch_path": str(mt_path),
                "record_id": f"{i}_DUMY",
                "foldx": [0.1, -0.2, 0.3],
                "cath_label": i % 4,
            }
        )

    train = [r for r in rows if r["split_tag"] == "train"]
    val = [r for r in rows if r["split_tag"] == "val"]

    train_jsonl = root / "train.jsonl"
    val_jsonl = root / "val.jsonl"
    with train_jsonl.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with val_jsonl.open("w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r) + "\n")

    return train_jsonl, val_jsonl
