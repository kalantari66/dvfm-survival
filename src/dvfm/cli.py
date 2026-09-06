"""Command-line interface."""

import argparse
from pathlib import Path

from .config import expand_scenarios, load_config
from .runner import run, validate_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DVFM survival experiments with reference parameters")
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration and input files without fitting any model",
    )
    parser.add_argument(
        "--scenario-index", type=int,
        help="Run one zero-based expanded data-grid scenario (used by GWF sharding)",
    )
    parser.add_argument(
        "--output-dir", help="Override study.output_dir for a sharded run"
    )
    parser.add_argument(
        "--repeat-index", type=int,
        help="Run one zero-based paired seed repeat (used by GWF sharding)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.scenario_index is not None:
        scenarios = expand_scenarios(cfg["data"])
        if not 0 <= args.scenario_index < len(scenarios):
            parser.error(
                f"--scenario-index must be between 0 and {len(scenarios) - 1}"
            )
        cfg["data"].pop("grid", None)
        cfg["data"]["scenarios"] = [scenarios[args.scenario_index]]
    if args.output_dir:
        cfg["study"]["output_dir"] = str(Path(args.output_dir))
    if args.repeat_index is not None:
        repeat_count = len(cfg["seeds"]["sampling"])
        if not 0 <= args.repeat_index < repeat_count:
            parser.error(f"--repeat-index must be between 0 and {repeat_count - 1}")
        for seed_name in ("sampling", "split", "model"):
            cfg["seeds"][seed_name] = [cfg["seeds"][seed_name][args.repeat_index]]
        cfg["_repeat_index"] = args.repeat_index
    if args.validate_only:
        validate_inputs(cfg)
        print("Configuration and inputs are valid. No model was trained.")
        return
    run(cfg)


if __name__ == "__main__":
    main()
