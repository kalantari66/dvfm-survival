"""Print the winning hyperparameters from a semi-synthetic tuning run.

The tuning artifacts are written as JSON for downstream tooling; this renders
them for a job log, together with a block that can be pasted straight into
``models.tuned_by_dataset`` in configs/semi_synthetic.yaml.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _format_value(value) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    if isinstance(value, float):
        # YAML-safe: no exponent notation, which the config avoids throughout,
        # and a trailing zero so floats stay visibly float (0.0, not 0).
        text = f"{value:.10f}".rstrip("0")
        return text + "0" if text.endswith(".") else text
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--model", help="Optional model name; defaults to every model present")
    args = parser.parse_args()

    selections = json.loads(
        (args.result_root / "selected_hyperparameters.json").read_text(encoding="utf-8")
    )
    if args.model:
        selections = [item for item in selections if item["model"] == args.model]
    if not selections:
        raise SystemExit(f"No tuning selections found under {args.result_root}")

    selections.sort(key=lambda item: (item["model"], item["dataset"]))
    width = max(len(item["dataset"]) for item in selections)

    print("=" * 78)
    print(f"Best hyperparameters by validation oracle IBS  ({args.result_root})")
    print("=" * 78)
    for item in selections:
        parameters = ", ".join(
            f"{key}: {_format_value(value)}"
            for key, value in sorted(item["parameters"].items())
        )
        print(
            f"{item['dataset']:<{width}}  oracle_ibs={item['validation_oracle_ibs']:.5f}"
            f"  (trial {item['trial']:>2})  {parameters}"
        )

    print()
    print("-" * 78)
    print("Paste into models.tuned_by_dataset in configs/semi_synthetic.yaml:")
    print("-" * 78)
    for item in selections:
        parameters = ", ".join(
            f"{key}: {_format_value(value)}"
            for key, value in sorted(item["parameters"].items())
        )
        print(f"    {item['dataset']}:")
        print(f"      {item['model']}: {{{parameters}}}")


if __name__ == "__main__":
    main()
