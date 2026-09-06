"""Command-line interface."""

import argparse

from .config import load_config
from .runner import run, validate_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DVFM survival experiments with reference parameters")
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration and input files without fitting any model",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.validate_only:
        validate_inputs(cfg)
        print("Configuration and inputs are valid. No model was trained.")
        return
    run(cfg)


if __name__ == "__main__":
    main()
