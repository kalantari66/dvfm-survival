"""Re-aggregate the semi-synthetic results after rerunning a single model.

Only the named model's seed jobs need to have been rerun; every other model's
per-model CSVs are read from disk and carried through unchanged. This performs
the same three aggregation steps as the GWF workflow, but invoking them
directly avoids GWF's dependency resolution, which would otherwise schedule
every other model's targets because the shared experiment config is an input to
all of them.

    python scripts/reaggregate_semi_synthetic_model.py --model deepsurv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
RESULT_FILES = ("results_raw.csv", "results_mean.csv", "results_std.csv")


def _summary(path: Path, model: str) -> pd.Series | None:
    """Mean oracle C-index per dataset for one model, or None if unavailable."""
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    column = next(
        (name for name in ("oracle_ci", "CI Oracle") if name in frame.columns), None
    )
    if column is None or "Model" not in frame.columns:
        return None
    rows = frame[frame["Model"].astype(str).str.lower() == model.lower()]
    if rows.empty:
        return None
    return rows.groupby("Dataset")[column].mean()


def _relative(part: str) -> str:
    try:
        return Path(part).relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return part


def _run(command: list[str], dry_run: bool) -> None:
    print("  $ python " + " ".join(_relative(part) for part in command[1:]), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "semi_synthetic.yaml")
    parser.add_argument("--result-root", type=Path, default=PROJECT_ROOT / "results" / "semi-synthetic")
    parser.add_argument("--model", default="deepsurv", help="The model whose seed jobs were rerun")
    parser.add_argument("--dataset", action="append", help="Optional dataset name; may be repeated")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = args.model.lower()
    enabled = [str(name).lower() for name in config["models"]["enabled"]]
    if model not in enabled:
        raise SystemExit(f"{model!r} is not in models.enabled: {enabled}")
    names = [str(item["name"]) for item in config["data"]["datasets"]]
    if args.dataset:
        wanted = {name.lower() for name in args.dataset}
        names = [name for name in names if name.lower() in wanted]
        if len(names) != len(wanted):
            raise SystemExit("--dataset must name configured semi-synthetic datasets")
    seeds = config["seeds"]
    seed_count = len(seeds) if isinstance(seeds, list) else len(seeds["model"])

    # Fail before writing anything if a seed job is missing or did not finish,
    # so a partial rerun cannot be silently aggregated into the reported means.
    missing: list[str] = []
    for name in names:
        for index in range(seed_count):
            seed_dir = args.result_root / name / model / f"seed_{index}"
            if not (seed_dir / "_SUCCESS").is_file():
                missing.append(f"{seed_dir} (no _SUCCESS)")
            elif not (seed_dir / "results_raw.csv").stat().st_size:
                missing.append(f"{seed_dir} (empty results_raw.csv)")
    if missing:
        raise SystemExit(
            f"{len(missing)} {model} seed job(s) are missing or incomplete:\n  "
            + "\n  ".join(missing)
        )
    print(f"All {len(names) * seed_count} {model} seed jobs present.\n")

    before = _summary(args.result_root / "results_mean.csv", model)

    for name in names:
        print(f"[{name}]", flush=True)
        _run([
            sys.executable, str(SCRIPTS / "aggregate_semi_synthetic_model_seeds.py"),
            "--config", str(args.config), "--dataset", name,
            "--model", model, "--result-root", str(args.result_root),
        ], args.dry_run)
        _run([
            sys.executable, str(SCRIPTS / "aggregate_semi_synthetic_dataset_models.py"),
            "--config", str(args.config), "--dataset", name,
            "--result-root", str(args.result_root),
        ], args.dry_run)
    print("\n[cross-dataset]", flush=True)
    _run([
        sys.executable, str(SCRIPTS / "aggregate_semi_synthetic_results.py"),
        "--config", str(args.config), "--result-root", str(args.result_root),
    ], args.dry_run)

    if args.dry_run:
        print("\nDry run: nothing was written.")
        return

    for filename in RESULT_FILES:
        path = args.result_root / filename
        if not path.is_file() or not path.stat().st_size:
            raise SystemExit(f"Aggregation did not produce {path}")

    after = _summary(args.result_root / "results_mean.csv", model)
    print(f"\nMean oracle C-index for {model}:")
    if before is None or after is None:
        print("  (no comparable before/after summary available)")
    else:
        width = max(len(name) for name in after.index)
        for name in after.index:
            old = before.get(name)
            if old is None:
                print(f"  {name:<{width}}  {after[name]:.4f}  (new)")
            else:
                print(f"  {name:<{width}}  {old:.4f} -> {after[name]:.4f}  ({after[name] - old:+.4f})")
        print(f"  {'overall':<{width}}  {before.mean():.4f} -> {after.mean():.4f}"
              f"  ({after.mean() - before.mean():+.4f})")


if __name__ == "__main__":
    main()
