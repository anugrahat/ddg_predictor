# ddg_predictor

IG-DDG (Interface-Gated DeltaG Graph Network) training and evaluation scaffold for robust mutation effect prediction on PPIs.

## Quick start

```bash
python3 -m pip install -e .[dev]
```

## CLI

```bash
ddg-cli prepare-data --help
ddg-cli train-baseline --help
ddg-cli train-igddg --help
ddg-cli eval-heldout --help
ddg-cli run-ablations --help
ddg-cli leakage-audit --help
```

## Using the downloaded CATH-ddG bundle

If you extracted:
`/Users/felarof99/Downloads/datasets-20260206T181848Z-1-002.zip`

into:
`/Users/felarof99/Documents/New project/data/raw/cath_ddg/`

you can run:

```bash
python3 -m src.cli prepare-data \
  --csv data/raw/cath_ddg/datasets/SKEMPI2.csv \
  --output-dir data/processed/skempi2 \
  --pdb-zip data/raw/cath_ddg/datasets/PDBs.zip \
  --seed 42
```

Quick smoke train:

```bash
python3 -m src.cli train-igddg \
  --train-jsonl data/processed/skempi2/train.jsonl \
  --val-jsonl data/processed/skempi2/val.jsonl \
  --out-dir runs/igddg_smoke \
  --epochs 1 \
  --batch-size 2 \
  --grad-accum-steps 1 \
  --no-amp
```

## Data format (processed)

A processed dataset is a JSONL file where each row contains a mutation record and paths to `.npz` patches for WT/MT structures.

Patch `.npz` keys:
- `aa_ids` `(N,) int64`
- `esm` `(N,1280) float32`
- `scalars` `(N,S) float32`
- `atom_summary` `(N,A) float32`
- `mutation_mask` `(N,) float32`
- `interface_mask` `(N,) float32`
- `chain_ids` `(N,) int64`
- `edge_index` `(2,E) int64`
- `edge_feat` `(E,F) float32`
- `first_shell_mask` `(N,) float32`

## Note

This repo includes a complete implementation scaffold and deterministic tests. Real benchmark replication requires SKEMPI/CATH data assets and trained runs.

# ddg_predictor
