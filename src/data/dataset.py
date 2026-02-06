"""Torch dataset and collate utilities for IG-DDG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import load_jsonl


class PatchDataset(Dataset):
    def __init__(self, jsonl_path: str) -> None:
        self.rows = load_jsonl(jsonl_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        wt = np.load(Path(row["wt_patch_path"]))
        mt = np.load(Path(row["mt_patch_path"]))
        foldx = row.get("foldx", None)

        sample = {
            "wt": {k: wt[k] for k in wt.files},
            "mt": {k: mt[k] for k in mt.files},
            "ddg": float(row["ddg"]),
            "ppi_id": row.get("ppi_id", row.get("pdb_id", "")),
            "record_id": row.get("record_id", row.get("pdb_id", "")),
            "cath_label": row.get("cath_label", None),
            "foldx": np.asarray(foldx, dtype=np.float32) if foldx is not None else None,
            "split_tag": row.get("split_tag", "train"),
            "cath_superfamily": row.get("cath_superfamily", "unknown"),
        }
        return sample


def _pad_array(arr: np.ndarray, shape: tuple[int, ...], fill: float = 0.0, dtype=None) -> np.ndarray:
    out = np.full(shape, fill, dtype=dtype or arr.dtype)
    slices = tuple(slice(0, s) for s in arr.shape)
    out[slices] = arr
    return out


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    B = len(batch)
    max_n = max(int(x["wt"]["aa_ids"].shape[0]) for x in batch)
    max_e = max(int(x["wt"]["edge_index"].shape[1]) for x in batch)

    def build_patch(key: str) -> dict[str, torch.Tensor]:
        aa_ids = np.zeros((B, max_n), dtype=np.int64)
        esm = np.zeros((B, max_n, 1280), dtype=np.float32)
        scalars_dim = int(batch[0][key]["scalars"].shape[1])
        atom_dim = int(batch[0][key]["atom_summary"].shape[1])
        edge_dim = int(batch[0][key]["edge_feat"].shape[1]) if max_e > 0 else 25
        scalars = np.zeros((B, max_n, scalars_dim), dtype=np.float32)
        atom_summary = np.zeros((B, max_n, atom_dim), dtype=np.float32)
        mutation_mask = np.zeros((B, max_n), dtype=np.float32)
        interface_mask = np.zeros((B, max_n), dtype=np.float32)
        first_shell_mask = np.zeros((B, max_n), dtype=np.float32)
        chain_ids = np.zeros((B, max_n), dtype=np.int64)
        node_mask = np.zeros((B, max_n), dtype=np.float32)

        edge_index = np.full((B, 2, max_e), -1, dtype=np.int64)
        edge_feat = np.zeros((B, max_e, edge_dim), dtype=np.float32)
        edge_mask = np.zeros((B, max_e), dtype=np.float32)

        for i, sample in enumerate(batch):
            p = sample[key]
            n = p["aa_ids"].shape[0]
            e = p["edge_index"].shape[1]
            aa_ids[i, :n] = p["aa_ids"]
            esm[i, :n] = p["esm"]
            scalars[i, :n] = p["scalars"]
            atom_summary[i, :n] = p["atom_summary"]
            mutation_mask[i, :n] = p["mutation_mask"]
            interface_mask[i, :n] = p["interface_mask"]
            first_shell_mask[i, :n] = p["first_shell_mask"]
            chain_ids[i, :n] = p["chain_ids"]
            node_mask[i, :n] = 1.0

            if e > 0:
                edge_index[i, :, :e] = p["edge_index"]
                edge_feat[i, :e] = p["edge_feat"]
                edge_mask[i, :e] = 1.0

        return {
            "aa_ids": torch.from_numpy(aa_ids),
            "esm": torch.from_numpy(esm),
            "scalars": torch.from_numpy(scalars),
            "atom_summary": torch.from_numpy(atom_summary),
            "mutation_mask": torch.from_numpy(mutation_mask),
            "interface_mask": torch.from_numpy(interface_mask),
            "first_shell_mask": torch.from_numpy(first_shell_mask),
            "chain_ids": torch.from_numpy(chain_ids),
            "node_mask": torch.from_numpy(node_mask),
            "edge_index": torch.from_numpy(edge_index),
            "edge_feat": torch.from_numpy(edge_feat),
            "edge_mask": torch.from_numpy(edge_mask),
        }

    wt_patch = build_patch("wt")
    mt_patch = build_patch("mt")

    ppi_to_id: dict[str, int] = {}
    ppi_group: list[int] = []
    for x in batch:
        ppi = str(x["ppi_id"])
        if ppi not in ppi_to_id:
            ppi_to_id[ppi] = len(ppi_to_id)
        ppi_group.append(ppi_to_id[ppi])

    ddg = torch.tensor([float(x["ddg"]) for x in batch], dtype=torch.float32)
    sign_target = (ddg > 0).float()

    has_foldx = any(x["foldx"] is not None for x in batch)
    foldx = None
    foldx_mask = None
    if has_foldx:
        dim = max((len(x["foldx"]) for x in batch if x["foldx"] is not None), default=0)
        arr = np.zeros((B, dim), dtype=np.float32)
        m = np.zeros((B,), dtype=np.float32)
        for i, x in enumerate(batch):
            if x["foldx"] is None:
                continue
            fx = x["foldx"]
            arr[i, : len(fx)] = fx
            m[i] = 1.0
        foldx = torch.from_numpy(arr)
        foldx_mask = torch.from_numpy(m)

    cath_vals = [x.get("cath_label", None) for x in batch]
    cath_label = None
    if all(v is not None for v in cath_vals):
        cath_label = torch.tensor(cath_vals, dtype=torch.long)

    return {
        "wt_patch": wt_patch,
        "mt_patch": mt_patch,
        "ddg": ddg,
        "ppi_group": torch.tensor(ppi_group, dtype=torch.long),
        "sign_target": sign_target,
        "foldx": foldx,
        "foldx_mask": foldx_mask,
        "cath_label": cath_label,
        "reverse_ddg_mean": None,
        "metadata": {
            "record_ids": [x["record_id"] for x in batch],
            "ppi_ids": [x["ppi_id"] for x in batch],
            "split_tags": [x["split_tag"] for x in batch],
            "cath_superfamilies": [x["cath_superfamily"] for x in batch],
        },
    }
