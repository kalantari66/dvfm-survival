"""Command-line interface."""

import argparse

from .config import apply_quick_mode, load_config
from .runner import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DVFM survival experiments")
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file")
    parser.add_argument("--quick", action="store_true", help="Use a short smoke-test configuration")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.quick:
        cfg = apply_quick_mode(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
