"""Run one configured semi-synthetic dataset/model into a result directory."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from experiments.config import load_config, validate_config
from experiments.runner import run, validate_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    datasets = config["data"].get("datasets", [])
    selected = [item for item in datasets if str(item["name"]).lower() == args.dataset.lower()]
    if len(selected) != 1:
        raise ValueError(f"Expected one configured dataset named {args.dataset!r}")

    config = deepcopy(config)
    config["data"]["datasets"] = selected
    if args.model is not None:
        enabled = [str(model) for model in config["models"]["enabled"]]
        selected_models = [
            model for model in enabled if model.lower() == args.model.lower()
        ]
        if len(selected_models) != 1:
            raise ValueError(
                f"Expected one enabled model named {args.model!r}; "
                f"available models are {enabled}"
            )
        config["models"]["enabled"] = selected_models
    config["study"]["output_dir"] = str(args.output_dir)
    validate_config(config)
    validate_inputs(config)
    if not args.validate_only:
        run(config)


if __name__ == "__main__":
    main()
