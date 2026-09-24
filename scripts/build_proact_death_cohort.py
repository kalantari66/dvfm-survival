"""Build the PRO-ACT death-versus-censoring cohort from the raw form exports.

PRO-ACT is access-controlled: the raw forms and the built cohort stay under
the git-ignored ``data/`` directory and are never committed. The outcome and
covariate definitions live in :mod:`utility.proact`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utility.proact import build_proact_death_cohort


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", required=True, type=Path,
                        help="Directory holding the PROACT_*.csv form exports")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "data" / "proact_death.csv")
    args = parser.parse_args()
    cohort, flow = build_proact_death_cohort(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output, index=False)
    flow_path = args.output.with_name(f"{args.output.stem}_cohort_flow.json")
    flow_path.write_text(json.dumps(flow, indent=2), encoding="utf-8")
    print(json.dumps(flow, indent=2))
    print(f"Wrote {len(cohort)} subjects to {args.output}")


if __name__ == "__main__":
    main()
