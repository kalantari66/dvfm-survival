"""Dataset-independent subject-level exports for frailty recovery figures."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from utility.data import SurvivalDataset
from utility.metrics import pearson_correlation, spearman_correlation


def export_latent_recovery(model, validation, test, validation_indices, test_indices,
                           output_dir, context, batch_size, device):
    """Export one fitted model; calibration sees validation truth only.

    Inputs are SurvivalData instances from any generator exposing true_z.
    Missing truth (including independence) and multidimensional latents are
    explicitly recorded as unavailable rather than reporting fictitious recovery.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(context)
    if validation.true_z is None or test.true_z is None:
        metadata["status"] = "unavailable_no_true_frailty"
    elif model.encoder is None:
        metadata["status"] = "unavailable_requires_scalar_latent"
    else:
        frames = []
        model.eval()
        for data, indices in ((validation, validation_indices), (test, test_indices)):
            means, stds = [], []
            loader = DataLoader(SurvivalDataset(data.X, data.time, data.event),
                                batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                for x, time, event in loader:
                    mu, logvar = model.encoder(x.to(device), time.to(device), event.to(device))
                    means.append(mu.cpu().numpy())
                    stds.append(torch.exp(0.5 * logvar).cpu().numpy())
            mu, std = np.concatenate(means), np.concatenate(stds)
            if mu.shape[1] != 1:
                metadata["status"] = "unavailable_requires_scalar_latent"
                break
            frames.append(pd.DataFrame({
                "row_index": indices, "event": data.event, "true_z": data.true_z,
                "learned_mu_raw": mu[:, 0], "learned_std": std[:, 0],
            }))
        else:
            metadata["status"] = "available"
            valid, heldout = frames
            correlation = pearson_correlation(valid.true_z, valid.learned_mu_raw)
            sign = -1.0 if correlation < 0 else 1.0
            calibrator = LinearRegression().fit(
                (sign * valid.learned_mu_raw.to_numpy()).reshape(-1, 1), valid.true_z
            )
            alpha, beta = float(calibrator.intercept_), float(calibrator.coef_[0])
            metadata.update(alignment_sign=sign, calibration_intercept=alpha,
                            calibration_slope=beta)
            rows = []
            for split, frame in (("validation", valid), ("test", heldout)):
                frame["learned_mu_aligned"] = sign * frame.learned_mu_raw
                frame["learned_z_calibrated"] = alpha + beta * frame.learned_mu_aligned
                frame["learned_z_calibrated_std"] = abs(beta) * frame.learned_std
                radius = 1.96 * frame.learned_z_calibrated_std
                frame["calibrated_lower_95"] = frame.learned_z_calibrated - radius
                frame["calibrated_upper_95"] = frame.learned_z_calibrated + radius
                frame["covered_95"] = frame.true_z.between(
                    frame.calibrated_lower_95, frame.calibrated_upper_95)
                frame["interval_width"] = 2 * radius
                frame.to_csv(output_dir / f"latent_recovery_{split}.csv", index=False)
                for subgroup, subset in (("All", frame),
                                         ("Event observed", frame[frame.event == 1]),
                                         ("Censored", frame[frame.event == 0])):
                    for representation, column in (
                        ("Aligned, uncalibrated", "learned_mu_aligned"),
                        ("Validation-calibrated", "learned_z_calibrated"),
                    ):
                        truth, prediction = subset.true_z, subset[column]
                        rows.append({
                            **context, "split": split, "subgroup": subgroup,
                            "representation": representation, "n": len(subset),
                            "pearson": pearson_correlation(truth, prediction),
                            "spearman": spearman_correlation(truth, prediction),
                            "r2": float(r2_score(truth, prediction)) if len(subset) > 1 else np.nan,
                            "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))) if len(subset) else np.nan,
                            "coverage_95": float(subset.covered_95.mean()) if representation == "Validation-calibrated" else np.nan,
                            "mean_interval_width": float(subset.interval_width.mean()) if representation == "Validation-calibrated" else np.nan,
                        })
            pd.DataFrame(rows).to_csv(output_dir / "latent_calibration_metrics.csv", index=False)
    with (output_dir / "latent_recovery_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
