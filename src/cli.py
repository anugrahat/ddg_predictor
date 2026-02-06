"""Project CLI entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.prepare import leakage_audit, prepare_dataset
from src.train.engine import TrainConfig, evaluate_checkpoint, train_model
from src.train.losses import LossConfig
from src.utils import ensure_dir


def cmd_prepare_data(args: argparse.Namespace) -> None:
    stats = prepare_dataset(
        csv_path=args.csv,
        heldout_csv_path=args.heldout_csv,
        output_dir=args.output_dir,
        pdb_dir=args.pdb_dir,
        pdb_zip=args.pdb_zip,
        cache_zip=args.cache_zip,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2))


def _train_common(args: argparse.Namespace, mode: str) -> None:
    use_inter = True
    use_film = True
    use_foldx = args.use_foldx

    if mode == "baseline":
        use_inter = False
        use_film = False
        use_foldx = False

    cfg = TrainConfig(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        out_dir=args.out_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_frac=args.warmup_frac,
        amp=not args.no_amp,
        patience=args.patience,
        foldx_dim=args.foldx_dim,
        use_foldx_branch=use_foldx,
        use_inter_chain_attention=use_inter,
        use_film=use_film,
    )
    lcfg = LossConfig()
    result = train_model(cfg, lcfg)
    print(json.dumps(result, indent=2))


def cmd_train_baseline(args: argparse.Namespace) -> None:
    _train_common(args, mode="baseline")


def cmd_train_igddg(args: argparse.Namespace) -> None:
    _train_common(args, mode="igddg")


def cmd_eval_heldout(args: argparse.Namespace) -> None:
    metrics = evaluate_checkpoint(args.checkpoint, args.jsonl, batch_size=args.batch_size)
    print(json.dumps(metrics, indent=2))


def cmd_run_ablations(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)
    variants = [
        ("variant_a_no_foldx", {"use_foldx": False, "mode": "igddg"}),
        ("variant_b_with_foldx", {"use_foldx": True, "mode": "igddg"}),
        ("variant_c_no_inter", {"use_foldx": True, "mode": "custom", "use_inter": False, "use_film": True}),
        ("variant_d_no_film", {"use_foldx": True, "mode": "custom", "use_inter": True, "use_film": False}),
    ]

    summaries = {}
    for name, spec in variants:
        out_dir = Path(args.out_dir) / name
        ensure_dir(out_dir)
        if spec["mode"] == "custom":
            cfg = TrainConfig(
                train_jsonl=args.train_jsonl,
                val_jsonl=args.val_jsonl,
                out_dir=str(out_dir),
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum_steps,
                lr=args.lr,
                weight_decay=args.weight_decay,
                warmup_frac=args.warmup_frac,
                amp=not args.no_amp,
                patience=args.patience,
                foldx_dim=args.foldx_dim,
                use_foldx_branch=spec["use_foldx"],
                use_inter_chain_attention=spec["use_inter"],
                use_film=spec["use_film"],
            )
            result = train_model(cfg, LossConfig())
        else:
            tmp = argparse.Namespace(**vars(args))
            tmp.out_dir = str(out_dir)
            tmp.use_foldx = spec["use_foldx"]
            result = train_model(
                TrainConfig(
                    train_jsonl=tmp.train_jsonl,
                    val_jsonl=tmp.val_jsonl,
                    out_dir=tmp.out_dir,
                    seed=tmp.seed,
                    epochs=tmp.epochs,
                    batch_size=tmp.batch_size,
                    grad_accum_steps=tmp.grad_accum_steps,
                    lr=tmp.lr,
                    weight_decay=tmp.weight_decay,
                    warmup_frac=tmp.warmup_frac,
                    amp=not tmp.no_amp,
                    patience=tmp.patience,
                    foldx_dim=tmp.foldx_dim,
                    use_foldx_branch=tmp.use_foldx,
                    use_inter_chain_attention=True,
                    use_film=True,
                ),
                LossConfig(),
            )
        summaries[name] = result

    summary_path = Path(args.out_dir) / "ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(json.dumps(summaries, indent=2))


def cmd_leakage_audit(args: argparse.Namespace) -> None:
    report = leakage_audit(args.train_jsonl, args.heldout_jsonl)
    print(json.dumps(report, indent=2))


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--foldx-dim", type=int, default=32)
    p.add_argument("--use-foldx", action="store_true")
    p.add_argument("--no-amp", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ddg-cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare-data")
    p.add_argument("--csv", required=True)
    p.add_argument("--heldout-csv", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--pdb-dir", default=None)
    p.add_argument("--pdb-zip", default=None)
    p.add_argument("--cache-zip", default=None)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_prepare_data)

    p = sub.add_parser("train-baseline")
    _add_train_args(p)
    p.set_defaults(func=cmd_train_baseline)

    p = sub.add_parser("train-igddg")
    _add_train_args(p)
    p.set_defaults(func=cmd_train_igddg)

    p = sub.add_parser("eval-heldout")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.set_defaults(func=cmd_eval_heldout)

    p = sub.add_parser("run-ablations")
    _add_train_args(p)
    p.set_defaults(func=cmd_run_ablations)

    p = sub.add_parser("leakage-audit")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--heldout-jsonl", required=True)
    p.set_defaults(func=cmd_leakage_audit)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
