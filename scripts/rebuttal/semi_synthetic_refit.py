"""Tasks 1 and 5: refit DVFM variants on the semi-synthetic benchmark cells.

One dataset and seed per call, every scenario of ``configs/semi_synthetic.yaml``
(independence and the four copulas at tau = 0.5), with the dataset's tuned
DVFM settings unchanged. ``--variant`` selects:

* ``dvfm``: the paper's DVFM (d_z = 1); Task 1 and the d_z = 1 reference of
  Task 5;
* ``two_trunk_independent``: d_z = 2, z1 feeds only the event trunk and z2
  only the censoring trunk (Task 5 control);
* ``two_trunk_shared``: d_z = 2, both latents feed both trunks (Task 5
  capacity-matched reference).

The benchmark's prediction path is replayed first, in the same order and with
the same RNG consumption as ``experiments.runner`` (aggregate-posterior curves,
learned tau and joint ISE under the prior), so ``dvfm`` refits are comparable
with the paper rows. Only afterwards are the additions computed: prior-sampled
curves, joint ISE under the aggregate posterior, per-subject posterior means
and the checkpoint.
"""

from __future__ import annotations

import argparse
import time as timer
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dvfm.likelihood import encode_posterior
from dvfm.prediction import get_median_survival_time, predict_survival_curves, predict_survival_from_prior
from experiments.config import load_config
from experiments.runner import (
    _models_for_dataset, evaluation_time_grid, fit_dvfm, scale_observed_time, training_time_scale,
)
from utility.metrics import (
    JointSurvivalEvaluation, collect_metrics, learned_conditional_kendall_tau,
    oracle_joint_survival_ise, predict_dvfm_joint_survival,
)
from utility.runtime import resolve_device, seed_everything

from rebuttal_common import PROJECT_ROOT, environment_record, semisynthetic_cells, write_json

VARIANTS = {
    "dvfm": {"display": "DVFM", "overrides": {}},
    "two_trunk_independent": {
        "display": "DVFM-2T-independent",
        "overrides": {"latent_dim": 2, "decoder_architecture": "two_trunk_independent"},
    },
    "two_trunk_shared": {
        "display": "DVFM-2T-shared",
        "overrides": {"latent_dim": 2, "decoder_architecture": "two_trunk_shared"},
    },
}


def refit_cell(cell, cfg, settings, display, device, output_dir):
    evaluation = cfg["evaluation"]
    train, validation, test = cell["train"], cell["validation"], cell["test"]
    time_points = evaluation_time_grid(train, evaluation)
    time_scale = training_time_scale(train)
    model_time_points = time_points / time_scale
    model_train, model_validation, model_test = (
        scale_observed_time(part, time_scale) for part in (train, validation, test)
    )
    joint = cell["joint_evaluation"]
    model_joint = None if joint is None else JointSurvivalEvaluation(
        X=joint.X, event_grid=np.asarray(joint.event_grid) / time_scale,
        censor_grid=np.asarray(joint.censor_grid) / time_scale, truth=joint.truth,
    )

    started = timer.perf_counter()
    model, artifacts, train_loader, checkpoint = fit_dvfm(model_train, model_validation, settings, device)
    fit_seconds = timer.perf_counter() - started

    # --- The benchmark's path, in the benchmark's order. -------------------
    aggregate = predict_survival_curves(
        model=model, X=model_test.X, time_points=model_time_points, train_loader=train_loader,
        n_samples=int(settings["mc_samples"]), device=device,
    )
    learned_tau = learned_conditional_kendall_tau(
        model, np.mean(train.X, axis=0), int(evaluation.get("dependence_samples", 2000)), device,
    )
    joint_kwargs = dict(
        mc_samples=int(evaluation.get("joint_model_samples", settings["mc_samples"])),
        batch_size=int(evaluation.get("joint_model_batch_size", settings["batch_size"])),
        seed=0, device=device,
    )
    joint_prior = None if model_joint is None else oracle_joint_survival_ise(
        predict_dvfm_joint_survival(model, model_joint, **joint_kwargs), joint
    )

    # --- Additions. ---------------------------------------------------------
    mixture_mu, mixture_std = encode_posterior(
        model, model_train.X, model_train.time, model_train.event,
        batch_size=int(settings["batch_size"]), device=device,
    )
    joint_aggregate = None if model_joint is None else oracle_joint_survival_ise(
        predict_dvfm_joint_survival(model, model_joint, mixture=(mixture_mu, mixture_std), **joint_kwargs), joint
    )
    seed_everything(int(cell["seeds"]["model"]) + 60_000)
    prior = predict_survival_from_prior(
        model, model_test.X, model_time_points, int(settings["mc_samples"]), device
    )

    subject_frames = []
    for split, part, indices in (("train", model_train, cell["train_idx"]),
                                 ("validation", model_validation, cell["validation_idx"]),
                                 ("test", model_test, cell["test_idx"])):
        mu, std = encode_posterior(model, part.X, part.time, part.event,
                                   batch_size=int(settings["batch_size"]), device=device)
        frame = pd.DataFrame({"split": split, "row_index": indices, "event": part.event,
                              "true_z": part.true_z if part.true_z is not None else np.nan})
        for dim in range(mu.shape[1]):
            frame[f"mu_{dim}"], frame[f"std_{dim}"] = mu[:, dim], std[:, dim]
        subject_frames.append(frame)
    stem = f"{cell['scenario']}_repeat_{cell['repeat']}"
    (output_dir / "subjects").mkdir(parents=True, exist_ok=True)
    pd.concat(subject_frames, ignore_index=True).to_csv(
        output_dir / "subjects" / f"{stem}.csv.gz", index=False, compression="gzip"
    )
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "settings": settings, "display": display,
        "input_dim": int(model_train.X.shape[1]), "time_scale": time_scale,
        "checkpoint": checkpoint, "best_validation_elbo_epoch": artifacts["best_validation_elbo_epoch"],
        "scenario": cell["scenario"], "repeat": cell["repeat"], "seeds": cell["seeds"],
    }, output_dir / "checkpoints" / f"{stem}.pt")

    base = {
        "Dataset": cell["dataset"], "Scenario": cell["scenario"], "Copula": cell["copula"],
        "Target Kendall Tau": cell["kendall_tau"], "Repeat": cell["repeat"],
        "Model Seed": int(cell["seeds"]["model"]), "Model": display,
        "latent_dim": int(settings["latent_dim"]),
        "decoder_architecture": str(settings.get("decoder_architecture", "single")),
        "checkpoint": checkpoint, "best_validation_elbo_epoch": artifacts["best_validation_elbo_epoch"],
        "fit_seconds": fit_seconds, "Training Time Scale": time_scale,
        "Censoring Rate Train": float(1.0 - np.mean(train.event)),
        "learned_conditional_kendall_tau_prior": learned_tau,
        "latent_loading_l1_magnitude": float(model.decoder.latent_loading_l1().detach().cpu()),
        "numerical_failure": False,
    }
    rows = []
    for mode, survival, joint_ise in (("aggregate_posterior", aggregate, joint_aggregate),
                                      ("prior", prior, joint_prior)):
        median = get_median_survival_time(survival, model_time_points) * time_scale
        metrics = collect_metrics(display, median, survival, test.time, test.event, test.true_event_time,
                                  time_points, t_train=train.time, e_train=train.event, oracle_only=True)
        clean = {key.removeprefix(f"{display} "): value for key, value in metrics.items()}
        rows.append({
            **base, "prediction_mode": mode,
            "oracle_ibs": clean["IBS Oracle"], "oracle_ci": clean["CI Oracle"], "oracle_mae": clean["MAE Oracle"],
            "oracle_mae_censored": clean["MAE Oracle Censored"],
            "oracle_median_clipped_fraction": clean["Median Clipped Fraction"],
            "oracle_joint_survival_ise": joint_ise,
            # The benchmark's own joint ISE (prior) on every row, for the reproduction check.
            "benchmark_oracle_joint_survival_ise_prior": joint_prior,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "semi_synthetic.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="dvfm")
    parser.add_argument("--kendall-tau", type=float, nargs="*",
                        help="Restrict to these target taus (default: all configured)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device or str(cfg["compute"].get("device", "auto")))
    torch.set_num_threads(max(1, int(cfg["compute"].get("torch_num_threads", 1))))
    variant = VARIANTS[args.variant]
    settings = {**_models_for_dataset(cfg, args.dataset)["dvfm"], **variant["overrides"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "_SUCCESS").unlink(missing_ok=True)
    write_json(args.output_dir / "environment.json", environment_record(device))
    write_json(args.output_dir / "resolved_settings.json", {
        "variant": args.variant, "dataset": args.dataset, "seed_index": args.seed_index, "dvfm": settings,
    })
    rows = []
    for cell in semisynthetic_cells(cfg, args.dataset, args.seed_index):
        if args.kendall_tau and not any(np.isclose(cell["kendall_tau"], tau) for tau in args.kendall_tau):
            continue
        try:
            cell_rows = refit_cell(cell, cfg, settings, variant["display"], device, args.output_dir)
        except Exception as error:  # recorded, as the benchmark does, so pairing stays auditable
            cell_rows = [{"Dataset": cell["dataset"], "Scenario": cell["scenario"], "Repeat": cell["repeat"],
                          "Model": variant["display"], "numerical_failure": True, "error": repr(error)}]
        rows.extend(cell_rows)
        print(f"{cell['scenario']} repeat {cell['repeat']}: "
              + ", ".join(f"{r.get('prediction_mode', 'failed')} IBS={r.get('oracle_ibs', np.nan):.4f}" for r in cell_rows),
              flush=True)
        pd.DataFrame(rows).to_csv(args.output_dir / "results_raw.csv", index=False)
    (args.output_dir / "_SUCCESS").write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
