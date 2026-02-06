"""Shared type definitions for IG-DDG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class MutationRecord:
    pdb_id: str
    chain_id: str
    residue_idx: int
    wt_aa: str
    mt_aa: str
    ddg: float
    split_tag: str
    ppi_id: str = ""
    cath_superfamily: str = ""
    wt_patch_path: str = ""
    mt_patch_path: str = ""
    foldx: list[float] | None = None
    cath_label: int | None = None
    reverse_id: str | None = None
    record_id: str = ""


@dataclass
class InterfacePatch:
    aa_ids: np.ndarray
    esm: np.ndarray
    scalars: np.ndarray
    atom_summary: np.ndarray
    mutation_mask: np.ndarray
    interface_mask: np.ndarray
    first_shell_mask: np.ndarray
    chain_ids: np.ndarray
    edge_index: np.ndarray
    edge_feat: np.ndarray


@dataclass
class BatchTensors:
    wt_patch: dict[str, torch.Tensor]
    mt_patch: dict[str, torch.Tensor]
    ddg: torch.Tensor
    ppi_group: torch.Tensor
    sign_target: torch.Tensor
    foldx: torch.Tensor | None = None
    foldx_mask: torch.Tensor | None = None
    cath_label: torch.Tensor | None = None
    reverse_ddg_mean: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
