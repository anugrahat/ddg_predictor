"""Dataset preparation pipeline from CATH-ddG style CSVs."""

from __future__ import annotations

import hashlib
import json
import random
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.constants import AA_TO_ID
from src.data.interface_graph import build_interface_patch, parse_mutation_string
from src.data.pdb_utils import Structure, parse_pdb_text
from src.types import MutationRecord
from src.utils import dump_jsonl, ensure_dir


def _hash_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _synthetic_structure(record_id: str, mutation: str, chain_a: str, chain_b: str) -> Structure:
    # Fallback structure if real PDB is missing.
    rng = np.random.default_rng(_hash_seed(record_id))
    residues = []
    from src.data.pdb_utils import Residue

    muts = parse_mutation_string(mutation)
    mut_positions = {(m.chain, m.pos): m.mt for m in muts}

    chains = [chain_a or "A", chain_b or "B"]
    for cidx, chain in enumerate(chains):
        for i in range(1, 161):
            x = float(i * 1.5)
            y = float(cidx * 8.0 + rng.normal(0, 0.3))
            z = float(rng.normal(0, 0.3))
            atoms = ["N", "CA", "C", "O", "CB"]
            coords = np.asarray([
                [x - 1.1, y, z],
                [x, y, z],
                [x + 1.2, y, z],
                [x + 2.0, y, z],
                [x, y + 1.0, z],
            ], dtype=np.float32)
            b = np.asarray([20.0, 20.0, 20.0, 20.0, 20.0], dtype=np.float32)
            aa = "A"
            if (chain, i) in mut_positions:
                aa = mut_positions[(chain, i)]
            from src.constants import AA_ORDER
            one_to_three = {
                "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
                "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
                "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
                "Y": "TYR", "V": "VAL", "X": "UNK",
            }
            resname = one_to_three.get(aa, "UNK")
            residues.append(
                Residue(
                    chain_id=chain,
                    resseq=i,
                    icode="",
                    resname=resname,
                    atom_names=atoms,
                    atom_coords=coords,
                    atom_b=b,
                )
            )
    return Structure(residues=residues)


def _load_text_from_zip(zip_path: str, candidates: list[str]) -> str | None:
    if not zip_path:
        return None
    zp = Path(zip_path)
    if not zp.exists():
        return None
    with zipfile.ZipFile(zp, "r") as zf:
        names = set(zf.namelist())
        for c in candidates:
            if c in names:
                return zf.read(c).decode("utf-8", errors="ignore")
        for c in candidates:
            suffix = "/" + c
            for name in names:
                if name.endswith(suffix):
                    return zf.read(name).decode("utf-8", errors="ignore")
    return None


def _load_structure_text(
    record_id: str,
    pdb_origin: str,
    pdb_dir: str | None,
    pdb_zip: str | None,
    cache_zip: str | None,
) -> str | None:
    if cache_zip:
        txt = _load_text_from_zip(cache_zip, [f"{record_id}.pdb", f"optimized1/{record_id}.pdb"])
        if txt:
            return txt

    if pdb_dir:
        pdir = Path(pdb_dir)
        for name in [f"{record_id}.pdb", f"{pdb_origin}.pdb", f"{pdb_origin.lower()}.pdb"]:
            p = pdir / name
            if p.exists():
                return p.read_text(encoding="utf-8", errors="ignore")

    if pdb_zip:
        txt = _load_text_from_zip(pdb_zip, [f"{pdb_origin}.pdb", f"{pdb_origin.lower()}.pdb", f"{record_id}.pdb"])
        if txt:
            return txt
    return None


def _apply_mutation_to_patch_npz(npz_in: dict[str, np.ndarray], mutation: str) -> dict[str, np.ndarray]:
    out = {k: np.copy(v) for k, v in npz_in.items()}
    muts = parse_mutation_string(mutation)
    # The patch already encodes mutation mask from parsed structure matching chain+index.
    # We only adjust residue ID if a mutated node is marked.
    aa_ids = out["aa_ids"]
    m_mask = out["mutation_mask"] > 0.5
    mt_ids = [AA_TO_ID.get(m.mt, AA_TO_ID["X"]) for m in muts]
    if mt_ids and m_mask.any():
        aa_ids[m_mask] = mt_ids[0]
    out["aa_ids"] = aa_ids
    return out


def _infer_split(pid: str, val_fraction: float) -> str:
    rnd = random.Random(_hash_seed(pid)).random()
    if rnd < val_fraction:
        return "val"
    return "train"


def prepare_dataset(
    csv_path: str,
    output_dir: str,
    heldout_csv_path: str | None = None,
    pdb_dir: str | None = None,
    pdb_zip: str | None = None,
    cache_zip: str | None = None,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    random.seed(seed)

    out_root = Path(output_dir)
    ensure_dir(out_root)
    patches_dir = out_root / "patches"
    ensure_dir(patches_dir)

    def process_csv(path: str, default_split: str | None) -> list[dict[str, Any]]:
        df = pd.read_csv(path)
        rows: list[dict[str, Any]] = []

        for _, r in df.iterrows():
            rid = str(r.get("#Pdb", r.get("#Pdb_wt", "")))
            if not rid:
                continue
            pdb_origin = str(r.get("#Pdb_origin", ""))
            mutation = str(r.get("Mutation(s)_cleaned", "")).strip()
            if not mutation:
                continue

            ppi_id = f"{pdb_origin}:{r.get('Partner1','')}:{r.get('Partner2','')}"
            split = default_split or _infer_split(rid, val_fraction)
            cath_sf = str(r.get("CATH_superfamily", r.get("cath_superfamily", "unknown")))

            text = _load_structure_text(rid, pdb_origin, pdb_dir, pdb_zip, cache_zip)
            if text:
                structure = parse_pdb_text(text)
            else:
                structure = _synthetic_structure(
                    record_id=rid,
                    mutation=mutation,
                    chain_a=str(r.get("Partner1", "A")),
                    chain_b=str(r.get("Partner2", "B")),
                )

            try:
                wt_patch = build_interface_patch(structure, mutation, max_nodes=256)
            except ValueError:
                structure = _synthetic_structure(
                    record_id=rid,
                    mutation=mutation,
                    chain_a=str(r.get("Partner1", "A")),
                    chain_b=str(r.get("Partner2", "B")),
                )
                wt_patch = build_interface_patch(structure, mutation, max_nodes=256)

            wt_np = asdict(wt_patch)
            mt_np = _apply_mutation_to_patch_npz(wt_np, mutation)

            wt_path = patches_dir / f"{rid}_wt.npz"
            mt_path = patches_dir / f"{rid}_mt.npz"
            np.savez_compressed(wt_path, **wt_np)
            np.savez_compressed(mt_path, **mt_np)

            muts = parse_mutation_string(mutation)
            chain_id = muts[0].chain if muts else str(r.get("Partner1", "A"))
            residue_idx = muts[0].pos if muts else -1
            wt_aa = muts[0].wt if muts else "X"
            mt_aa = muts[0].mt if muts else "X"

            rec = MutationRecord(
                pdb_id=rid,
                chain_id=chain_id,
                residue_idx=residue_idx,
                wt_aa=wt_aa,
                mt_aa=mt_aa,
                ddg=float(r.get("ddG", 0.0)),
                split_tag=split,
                ppi_id=ppi_id,
                cath_superfamily=cath_sf,
                wt_patch_path=str(wt_path),
                mt_patch_path=str(mt_path),
                foldx=None,
                cath_label=None,
                reverse_id=None,
                record_id=rid,
            )
            rows.append(asdict(rec))
        return rows

    train_rows = process_csv(csv_path, default_split=None)
    heldout_rows = process_csv(heldout_csv_path, default_split="heldout") if heldout_csv_path else []

    train = [x for x in train_rows if x["split_tag"] == "train"]
    val = [x for x in train_rows if x["split_tag"] == "val"]
    heldout = heldout_rows

    dump_jsonl(out_root / "train.jsonl", train)
    dump_jsonl(out_root / "val.jsonl", val)
    dump_jsonl(out_root / "heldout.jsonl", heldout)

    return {
        "train": len(train),
        "val": len(val),
        "heldout": len(heldout),
        "total": len(train_rows) + len(heldout_rows),
    }


def leakage_audit(train_jsonl: str, heldout_jsonl: str) -> dict[str, int | list[str]]:
    def read(path: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    train = read(train_jsonl)
    heldout = read(heldout_jsonl)

    train_sf = {x.get("cath_superfamily", "unknown") for x in train}
    heldout_sf = {x.get("cath_superfamily", "unknown") for x in heldout}
    overlap = sorted(train_sf & heldout_sf)

    return {
        "train_superfamilies": len(train_sf),
        "heldout_superfamilies": len(heldout_sf),
        "overlap_count": len(overlap),
        "overlap_values": overlap[:50],
    }
