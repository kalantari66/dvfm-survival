"""Backward-compatible CLI import for existing editable installations."""

from experiments.cli import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
