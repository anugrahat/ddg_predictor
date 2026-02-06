"""Minimal PDB parsing and residue helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass
class Residue:
    chain_id: str
    resseq: int
    icode: str
    resname: str
    atom_names: list[str]
    atom_coords: np.ndarray
    atom_b: np.ndarray

    @property
    def one_letter(self) -> str:
        return THREE_TO_ONE.get(self.resname.upper(), "X")


@dataclass
class Structure:
    residues: list[Residue]


def _flush_residue(store: dict | None, out: list[Residue]) -> None:
    if store is None:
        return
    coords = np.asarray(store["coords"], dtype=np.float32)
    b = np.asarray(store["b"], dtype=np.float32)
    out.append(
        Residue(
            chain_id=store["chain_id"],
            resseq=store["resseq"],
            icode=store["icode"],
            resname=store["resname"],
            atom_names=list(store["atoms"]),
            atom_coords=coords,
            atom_b=b,
        )
    )


def parse_pdb_text(pdb_text: str) -> Structure:
    residues: list[Residue] = []
    current: dict | None = None
    current_key: tuple[str, int, str] | None = None

    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom = line[12:16].strip()
        altloc = line[16:17].strip()
        if altloc and altloc != "A":
            continue
        resname = line[17:20].strip()
        chain = line[21:22].strip() or "_"
        try:
            resseq = int(line[22:26].strip())
        except ValueError:
            continue
        icode = line[26:27].strip() or ""
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            b = float(line[60:66]) if line[60:66].strip() else 0.0
        except ValueError:
            continue

        key = (chain, resseq, icode)
        if current_key != key:
            _flush_residue(current, residues)
            current = {
                "chain_id": chain,
                "resseq": resseq,
                "icode": icode,
                "resname": resname,
                "atoms": [],
                "coords": [],
                "b": [],
            }
            current_key = key

        assert current is not None
        current["atoms"].append(atom)
        current["coords"].append([x, y, z])
        current["b"].append(b)

    _flush_residue(current, residues)
    return Structure(residues=residues)


def residue_atom_coord(residue: Residue, atom_name: str) -> np.ndarray | None:
    for i, name in enumerate(residue.atom_names):
        if name == atom_name:
            return residue.atom_coords[i]
    return None


def residue_cb(residue: Residue) -> tuple[np.ndarray, float]:
    cb = residue_atom_coord(residue, "CB")
    if cb is not None:
        return cb, 0.0
    ca = residue_atom_coord(residue, "CA")
    if ca is not None:
        return ca, 1.0
    return residue.atom_coords.mean(axis=0), 1.0


def pairwise_min_atom_distance(a: Residue, b: Residue) -> float:
    d = a.atom_coords[:, None, :] - b.atom_coords[None, :, :]
    return float(np.sqrt(np.sum(d * d, axis=-1)).min())


def infer_interface_mask(residues: Iterable[Residue], threshold: float = 10.0) -> np.ndarray:
    residues = list(residues)
    n = len(residues)
    out = np.zeros(n, dtype=np.float32)
    if n == 0:
        return out

    cb = np.zeros((n, 3), dtype=np.float32)
    for i, r in enumerate(residues):
        cb[i], _ = residue_cb(r)

    chains = np.asarray([r.chain_id for r in residues])
    d = cb[:, None, :] - cb[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    for i in range(n):
        cross = chains != chains[i]
        if np.any(cross & (dist[i] <= threshold)):
            out[i] = 1.0
    return out
