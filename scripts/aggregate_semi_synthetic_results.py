"""Create root-level, cross-dataset semi-synthetic result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


FILES = ("results_raw.csv", "results_mean.csv", "results_std.csv", "dgp_diagnostics.csv")
ACCURACY_REPORT_FILE = "accuracy_primary_and_sensitivity.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = args.result_root
    root.mkdir(parents=True, exist_ok=True)
    names = [str(dataset["name"]) for dataset in config["data"].get("datasets", [])]
    for filename in FILES:
        frames = [pd.read_csv(root / name / filename) for name in names]
        pd.concat(frames, ignore_index=True).to_csv(root / filename, index=False)
    means = pd.read_csv(root / "results_mean.csv")
    report_columns = [
        column for column in (
            "Dataset", "Scenario", "Copula", "Dependence", "Target Kendall Tau",
            "Target Censoring Rate", "Model", "latent_dim",
            "IBS Oracle", "IBS IPCW",
        ) if column in means.columns
    ]
    accuracy = means.loc[:, report_columns].rename(columns={
        "IBS Oracle": "Oracle IBS (primary; DGP ground truth)",
        "IBS IPCW": (
            "IPCW IBS (secondary sensitivity; assumes conditionally "
            "independent censoring)"
        ),
    })
    accuracy.to_csv(root / ACCURACY_REPORT_FILE, index=False)
    diagnostics = pd.read_csv(root / "dgp_diagnostics.csv")
    characteristics = (
        diagnostics.sort_values(["Dataset", "Repeat"])
        .drop_duplicates("Dataset")
        .loc[:, ["Dataset", "Source Samples", "Raw Features", "Encoded Features", "Source Event Rate"]]
        .rename(columns={
            "Source Samples": "$N$",
            "Raw Features": "Raw features",
            "Encoded Features": "Encoded features",
            "Source Event Rate": "Original event rate",
        })
    )
    characteristics["Original censoring rate"] = 1.0 - characteristics["Original event rate"]
    characteristics["Split"] = "70/10/20"
    characteristics.to_csv(root / "dataset_characteristics.csv", index=False)
    (root / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
