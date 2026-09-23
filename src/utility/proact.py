"""Build a death-versus-censoring PRO-ACT cohort from the raw form exports.

Covariates follow the MENSA single-event PRO-ACT pipeline
(``mensa/src/data/preprocess_proact.py`` and ``PROACTSingleDataLoader``): the
features that loader keeps after its drops are Age, Sex, Site_of_Onset,
Onset_Delta, baseline ALSFRS_R_Total, DiseaseProgressionRate,
Subject_used_Riluzole and FVC_Mean. Strength tests, race, El Escorial,
height, weight and BMI are dropped there and are not built here.

The outcome replaces MENSA's ALSFRS-item events with two:

* E, death: ``Death_Days`` from the mortality form, observed when
  ``Subject_Died == "Yes"``.
* C, censoring: the last day the subject appears on a clinical assessment
  form (not a lab draw), i.e. last seen alive.

Deliberate deviations from MENSA, each of which would otherwise leak outcome
information, duplicate subjects, or invent outcomes:

* Only subjects on the mortality form are kept. MENSA treats every absent
  subject as alive, but absence means vital status was never recorded (in the
  raw export no absent subject has a death), so neither E nor C is observed.
* MENSA sets an alive subject's death time to the maximum of its ALSFRS-item
  event times. Here it is the measured last-seen day.
* Per-subject tables are deduplicated before merging; MENSA's riluzole and
  history merges duplicate some subjects.
* FVC is the earliest measurement, not the first row in file order, which can
  lie months after baseline.
* An onset fewer than ``MIN_ONSET_DAYS`` days before baseline is set to
  missing: it is not clinically plausible for trial enrollment, and MENSA's
  progression rate divides by it, producing extreme outliers. The subject is
  kept; both covariates are imputed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Forms recording a dated clinical assessment of the subject. Adverse-event
# and concomitant-medication dates are excluded: they contain implausible
# offsets (up to ~10^6 days) and stop dates that are not contacts. Lab draws
# are excluded too: an isolated lab record can postdate every clinical
# assessment by years, which would set an implausible censoring time.
CONTACT_FORMS: dict[str, str] = {
    "ALSFRS": "ALSFRS_Delta",
    "FVC": "Forced_Vital_Capacity_Delta",
    "SVC": "Slow_vital_Capacity_Delta",
    "VITALSIGNS": "Vital_Signs_Delta",
    "HANDGRIPSTRENGTH": "MS_Delta",
    "MUSCLESTRENGTH": "MS_Delta",
}

NUMERIC_FEATURES = [
    "Age", "Onset_Delta", "ALSFRS_R_Total", "DiseaseProgressionRate", "FVC_Mean",
]
CATEGORICAL_FEATURES = ["Sex", "Site_of_Onset", "Subject_used_Riluzole"]
DAYS_PER_MONTH = 30.44
MIN_ONSET_DAYS = 30


def _read(raw_dir: Path, form: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = raw_dir / f"PROACT_{form}.csv"
    if not path.exists():
        raise FileNotFoundError(f"PRO-ACT form not found: {path}")
    return pd.read_csv(path, usecols=columns, low_memory=False)


def _first_non_null(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.dropna(subset=[column]).groupby("subject_id")[column].first()


def last_seen_day(raw_dir: Path, forms: dict[str, str] = CONTACT_FORMS) -> pd.Series:
    """Latest assessment day per subject across ``forms``."""
    per_form = []
    for form, column in forms.items():
        table = _read(raw_dir, form, ["subject_id", column])
        day = pd.to_numeric(table[column], errors="coerce")
        per_form.append(table.assign(day=day).groupby("subject_id")["day"].max())
    return pd.concat(per_form, axis=1).max(axis=1).rename("last_seen_day")


def alsfrs_r_decline(alsfrs: pd.DataFrame) -> pd.DataFrame:
    """Per-subject OLS decline in ALSFRS-R total, in points per month.

    Positive values mean faster functional loss. Uses every visit with a
    recorded ALSFRS-R total; it is an external validation signal only and is
    never a model input.
    """
    visits = alsfrs.dropna(subset=["ALSFRS_R_Total", "ALSFRS_Delta"])
    rows = []
    for subject, group in visits.groupby("subject_id"):
        months = group["ALSFRS_Delta"].to_numpy(float) / DAYS_PER_MONTH
        score = group["ALSFRS_R_Total"].to_numpy(float)
        span = float(np.ptp(months) * DAYS_PER_MONTH) if len(months) else 0.0
        slope = np.polyfit(months, score, 1)[0] if len(np.unique(months)) >= 2 else np.nan
        rows.append({
            "subject_id": subject, "alsfrs_r_decline_per_month": -slope,
            "alsfrs_r_n_visits": len(group), "alsfrs_r_span_days": span,
        })
    return pd.DataFrame(rows, columns=[
        "subject_id", "alsfrs_r_decline_per_month", "alsfrs_r_n_visits",
        "alsfrs_r_span_days",
    ])


def build_proact_death_cohort(
    raw_dir: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Return the analysis cohort and a cohort-flow summary."""
    raw_dir = Path(raw_dir)
    alsfrs = _read(raw_dir, "ALSFRS").sort_values(["subject_id", "ALSFRS_Delta"])
    history = _read(raw_dir, "ALSHISTORY", ["subject_id", "Onset_Delta", "Site_of_Onset"])
    flow: dict[str, int] = {"alsfrs_subjects": int(alsfrs["subject_id"].nunique())}

    df = pd.DataFrame({"subject_id": alsfrs["subject_id"].unique()})
    onset = _first_non_null(history, "Onset_Delta").abs()
    df = df.merge(onset.rename("Onset_Delta"), on="subject_id", how="left")
    df = df.dropna(subset=["Onset_Delta"])
    flow["with_onset"] = len(df)
    implausible_onset = df["Onset_Delta"] < MIN_ONSET_DAYS
    df.loc[implausible_onset, "Onset_Delta"] = np.nan

    baseline = alsfrs.drop_duplicates("subject_id")[["subject_id", "ALSFRS_R_Total"]]
    df = df.merge(baseline, on="subject_id", how="left")

    demographics = _read(raw_dir, "DEMOGRAPHICS", ["subject_id", "Age", "Sex"])
    demographics["Sex"] = demographics["Sex"].map(
        {"Male": "Male", "M": "Male", "Female": "Female", "F": "Female"}
    )
    df = df.merge(
        demographics.drop_duplicates("subject_id"), on="subject_id", how="left"
    )

    site = _first_non_null(history, "Site_of_Onset")
    site = site.str.replace("Onset: ", "", regex=False).str.replace(
        "Limb and Bulbar", "LimbAndBulbar", regex=False
    )
    df = df.merge(site.rename("Site_of_Onset"), on="subject_id", how="left")

    rate = (48 - df["ALSFRS_R_Total"]) / (df["Onset_Delta"].abs() / 30)
    df["DiseaseProgressionRate"] = rate.replace([np.inf, -np.inf], np.nan)

    riluzole = _read(raw_dir, "RILUZOLE", ["subject_id", "Subject_used_Riluzole"])
    df = df.merge(
        _first_non_null(riluzole, "Subject_used_Riluzole"), on="subject_id", how="left"
    )

    fvc = _read(raw_dir, "FVC")
    trials = [f"Subject_Liters_Trial_{i}" for i in range(1, 4)]
    fvc["FVC_Mean"] = fvc[trials].mean(axis=1)
    fvc = fvc.dropna(subset=["FVC_Mean"]).sort_values(
        ["subject_id", "Forced_Vital_Capacity_Delta"], na_position="last"
    )
    df = df.merge(
        fvc.drop_duplicates("subject_id")[["subject_id", "FVC_Mean"]],
        on="subject_id", how="left",
    )

    death = _read(raw_dir, "DEATHDATA").drop_duplicates("subject_id")
    df = df.merge(death, on="subject_id", how="left")
    df = df.merge(last_seen_day(raw_dir), on="subject_id", how="left")
    flow["dropped_no_death_record"] = int(df["Subject_Died"].isna().sum())
    df = df.loc[df["Subject_Died"].notna()]
    died = df["Subject_Died"].eq("Yes")
    flow["died"] = int(died.sum())
    unknown_death_day = died & df["Death_Days"].isna()
    flow["dropped_died_without_death_day"] = int(unknown_death_day.sum())
    df, died = df.loc[~unknown_death_day], died.loc[~unknown_death_day]

    df["time"] = np.where(died, df["Death_Days"], df["last_seen_day"])
    df["event"] = died.astype(int).to_numpy()
    flow["died_with_assessment_after_death"] = int(
        (died & (df["last_seen_day"] > df["Death_Days"])).sum()
    )
    positive = df["time"].notna() & (df["time"] > 0)
    flow["dropped_nonpositive_time"] = int((~positive).sum())
    df = df.loc[positive]

    df = df.merge(alsfrs_r_decline(alsfrs), on="subject_id", how="left")
    df["alsfrs_r_n_visits"] = df["alsfrs_r_n_visits"].fillna(0).astype(int)
    columns = [
        "subject_id", "time", "event", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES,
        "last_seen_day", "alsfrs_r_decline_per_month",
        "alsfrs_r_n_visits", "alsfrs_r_span_days",
    ]
    df = df[columns].sort_values("subject_id").reset_index(drop=True)
    if df["subject_id"].duplicated().any():
        raise RuntimeError("PRO-ACT cohort contains duplicated subjects")
    flow.update({
        "min_onset_days": MIN_ONSET_DAYS,
        "onset_set_missing_below_min": int(df["Onset_Delta"].isna().sum()),
        "final_subjects": len(df), "final_events": int(df["event"].sum()),
        "final_event_rate": float(df["event"].mean()),
        "with_alsfrs_r_decline": int(df["alsfrs_r_decline_per_month"].notna().sum()),
    })
    return df, flow


__all__ = [
    "CATEGORICAL_FEATURES", "CONTACT_FORMS", "MIN_ONSET_DAYS", "NUMERIC_FEATURES",
    "alsfrs_r_decline", "build_proact_death_cohort", "last_seen_day",
]
