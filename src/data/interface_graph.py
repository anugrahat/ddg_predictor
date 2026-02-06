"""Interface-centric graph patch builder."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from src.constants import AA_TO_ID
from src.data.pdb_utils import Structure, infer_interface_mask, residue_cb
from src.types import InterfacePatch

_MUT_RE = re.compile(r"^([A-Z])([A-Za-z])(-?\d+)([A-Z])$")


@dataclass
class ParsedMutation:
    wt: str
    chain: str
    pos: int
    mt: str


def parse_mutation_string(mutation: str) -> list[ParsedMutation]:
    items: list[ParsedMutation] = []
    for tok in [x.strip() for x in mutation.split(",") if x.strip()]:
        m = _MUT_RE.match(tok)
        if not m:
            continue
        wt, chain, pos, mt = m.groups()
        items.append(ParsedMutation(wt=wt, chain=chain, pos=int(pos), mt=mt))
    return items


def _rbf(distance: np.ndarray, num_kernels: int = 16, dmax: float = 20.0) -> np.ndarray:
    centers = np.linspace(0.0, dmax, num_kernels, dtype=np.float32)
    gamma = 1.0 / (centers[1] - centers[0] + 1e-6) ** 2
    d = distance[..., None] - centers[None, ...]
    return np.exp(-gamma * d * d)


def _atom_summary(coords: np.ndarray, center: np.ndarray) -> np.ndarray:
    rel = coords - center[None, :]
    mean = rel.mean(axis=0)
    std = rel.std(axis=0)
    radius = float(np.sqrt(np.sum(rel * rel, axis=-1)).max()) if len(rel) else 0.0
    count = float(coords.shape[0]) / 20.0
    return np.asarray([*mean.tolist(), *std.tolist(), count, radius / 10.0], dtype=np.float32)


def build_interface_patch(structure: Structure, mutation: str, max_nodes: int = 256) -> InterfacePatch:
    residues = structure.residues
    n_res = len(residues)
    if n_res == 0:
        raise ValueError("Empty structure: no residues parsed")

    muts = parse_mutation_string(mutation)
    interface_mask_full = infer_interface_mask(residues)

    cb_full = np.zeros((n_res, 3), dtype=np.float32)
    cb_missing = np.zeros(n_res, dtype=np.float32)
    for i, r in enumerate(residues):
        cb, miss = residue_cb(r)
        cb_full[i] = cb
        cb_missing[i] = miss

    mutation_idx = {
        i for i, r in enumerate(residues)
        for m in muts if r.chain_id == m.chain and r.resseq == m.pos
    }
    anchor_idx = set(mutation_idx) | {i for i, v in enumerate(interface_mask_full) if v > 0.5}
    if not anchor_idx:
        anchor_idx = set(range(n_res))

    anchor_coords = cb_full[sorted(anchor_idx)]
    d2 = ((cb_full[:, None, :] - anchor_coords[None, :, :]) ** 2).sum(axis=-1)
    dmin = np.sqrt(d2.min(axis=1))

    ordered = sorted(
        range(n_res),
        key=lambda i: (
            float(dmin[i]),
            residues[i].chain_id,
            residues[i].resseq,
            residues[i].icode,
        ),
    )
    keep = ordered[: min(max_nodes, n_res)]
    keep_set = set(keep)

    local_map = {old: new for new, old in enumerate(keep)}
    N = len(keep)

    cb = cb_full[keep]
    chain_vocab = {c: idx for idx, c in enumerate(sorted({residues[i].chain_id for i in keep}))}

    aa_ids = np.zeros(N, dtype=np.int64)
    esm = np.zeros((N, 1280), dtype=np.float32)
    scalars = np.zeros((N, 6), dtype=np.float32)
    atom_summary = np.zeros((N, 8), dtype=np.float32)
    mutation_mask = np.zeros(N, dtype=np.float32)
    interface_mask = interface_mask_full[keep].astype(np.float32)
    first_shell_mask = np.zeros(N, dtype=np.float32)
    chain_ids = np.zeros(N, dtype=np.int64)

    mut_cb = cb[[local_map[x] for x in mutation_idx if x in keep_set]] if mutation_idx else np.zeros((0, 3), dtype=np.float32)

    for j, old_i in enumerate(keep):
        r = residues[old_i]
        aa_ids[j] = AA_TO_ID.get(r.one_letter, AA_TO_ID["X"])
        chain_ids[j] = chain_vocab[r.chain_id]
        if old_i in mutation_idx:
            mutation_mask[j] = 1.0
        if len(mut_cb) > 0:
            d = np.sqrt(((cb[j][None, :] - mut_cb) ** 2).sum(axis=-1)).min()
            if d <= 8.0:
                first_shell_mask[j] = 1.0
        avg_b = float(r.atom_b.mean()) if len(r.atom_b) else 0.0
        scalars[j] = np.asarray([
            avg_b / 100.0,
            cb_missing[old_i],
            0.0,
            0.0,
            1.0,
            (j + 1) / max(N, 1),
        ], dtype=np.float32)
        atom_summary[j] = _atom_summary(r.atom_coords, cb[j])

    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_feat: list[np.ndarray] = []

    seen: set[tuple[int, int, int]] = set()

    dist = np.sqrt(((cb[:, None, :] - cb[None, :, :]) ** 2).sum(axis=-1))

    def add_edge(i: int, j: int, edge_type: int) -> None:
        key = (i, j, edge_type)
        if i == j or key in seen:
            return
        seen.add(key)
        dij = float(dist[i, j])
        rbf = _rbf(np.asarray([dij], dtype=np.float32))[0]
        vec = cb[j] - cb[i]
        norm = float(np.linalg.norm(vec)) + 1e-8
        orient = vec / norm
        seq_gap = abs(keep[i] - keep[j]) / 50.0
        cross = 1.0 if chain_ids[i] != chain_ids[j] else 0.0
        iface = 1.0 if (interface_mask[i] > 0 and interface_mask[j] > 0) else 0.0
        et = np.zeros(3, dtype=np.float32)
        et[edge_type] = 1.0
        f = np.concatenate(
            [rbf, orient.astype(np.float32), np.asarray([seq_gap, cross, iface], dtype=np.float32), et],
            axis=0,
        )
        edge_src.append(i)
        edge_dst.append(j)
        edge_feat.append(f)

    # Intra-spatial kNN edges.
    k = min(20, max(N - 1, 1))
    for i in range(N):
        nbr = np.argsort(dist[i])
        picked = [j for j in nbr if j != i][:k]
        for j in picked:
            add_edge(i, j, 0)

    # Sequential edges.
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if chain_ids[i] != chain_ids[j]:
                continue
            if abs(residues[keep[i]].resseq - residues[keep[j]].resseq) <= 2:
                add_edge(i, j, 1)

    # Inter-chain contact edges.
    for i in range(N):
        cand = [
            (float(dist[i, j]), j)
            for j in range(N)
            if i != j and chain_ids[i] != chain_ids[j] and float(dist[i, j]) <= 8.0
        ]
        cand.sort(key=lambda x: x[0])
        for _, j in cand[:12]:
            add_edge(i, j, 2)

    if edge_feat:
        edge_index = np.asarray([edge_src, edge_dst], dtype=np.int64)
        edge_feat_arr = np.asarray(edge_feat, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_feat_arr = np.zeros((0, 25), dtype=np.float32)

    return InterfacePatch(
        aa_ids=aa_ids,
        esm=esm,
        scalars=scalars,
        atom_summary=atom_summary,
        mutation_mask=mutation_mask,
        interface_mask=interface_mask,
        first_shell_mask=first_shell_mask,
        chain_ids=chain_ids,
        edge_index=edge_index,
        edge_feat=edge_feat_arr,
    )
