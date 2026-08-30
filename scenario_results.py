"""Build JSON-safe, frontend-facing datasets from SDR model results."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


MMR_MULTIPLIER = 100_000
MMR_LEVELS = (
    ("l23", "L2/3", 1),
    ("l4", "L4", 2),
    ("l5", "L5", 3),
)
DEATH_CAUSES = (
    ("pph", "Postpartum hemorrhage"),
    ("sepsis", "Sepsis"),
    ("eclampsia", "Eclampsia"),
    ("ol", "Obstructed labor"),
    ("aph", "Antepartum hemorrhage"),
    ("other", "Other"),
)
SUPPORTED_CI_METHODS = {
    "mean_per_run_poisson_bounds",
    "empirical_run_quantiles",
}


def _json_value(value: Any) -> Any:
    """Convert NumPy/Pandas scalar values to strict JSON-compatible values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def frame_dataset(dataset_id: str, label: str, frame: pd.DataFrame, **metadata: Any) -> dict[str, Any]:
    """Encode a DataFrame using the named-column-record convention."""
    rows = [
        {column: _json_value(value) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]
    dataset = {
        "datasetId": dataset_id,
        "label": label,
        "rowCount": len(rows),
        "columns": list(frame.columns),
        "rows": rows,
    }
    if metadata:
        dataset["metadata"] = metadata
    return dataset


def _location_values(frame: pd.DataFrame, column: str, n_months: int) -> pd.DataFrame:
    """Expand an SDR four-location array column while retaining L2/3, L4, and L5."""
    if column not in frame:
        raise KeyError(f"Missing required model-result column: {column}")
    if n_months < 1:
        raise ValueError("n_months must be at least 1")

    arrays = [np.asarray(value) for value in frame[column].values]
    if not arrays:
        return pd.DataFrame(columns=["run", "month", "levelId", "value"])
    values = np.concatenate(arrays).reshape(-1, 4)
    if len(values) % n_months:
        raise ValueError(
            f"Column {column!r} expands to {len(values)} monthly rows, "
            f"which is not divisible by n_months={n_months}"
        )

    run_count = len(values) // n_months
    base = pd.DataFrame(
        {
            "run": np.repeat(np.arange(1, run_count + 1), n_months),
            "month": np.tile(np.arange(1, n_months + 1), run_count),
        }
    )
    parts = []
    for level_id, _label, location_index in MMR_LEVELS:
        part = base.copy()
        part["levelId"] = level_id
        part["value"] = values[:, location_index]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _mmr_by_run(monthly_outcomes: pd.DataFrame, n_months: int) -> pd.DataFrame:
    deaths = _location_values(monthly_outcomes, "Deaths", n_months).rename(
        columns={"value": "deaths"}
    )
    births = _location_values(monthly_outcomes, "Live Births Final", n_months).rename(
        columns={"value": "liveBirths"}
    )
    values = deaths.merge(births, on=["run", "month", "levelId"], validate="one_to_one")
    valid = values["liveBirths"] > 0
    values["mmr"] = np.where(
        valid,
        values["deaths"] / values["liveBirths"] * MMR_MULTIPLIER,
        np.nan,
    )
    return values


def _add_poisson_bounds(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    alpha = 0.05
    lower_counts = np.where(
        result["deaths"] == 0,
        0,
        stats.chi2.ppf(alpha / 2, 2 * result["deaths"]) / 2,
    )
    upper_counts = stats.chi2.ppf(
        1 - alpha / 2,
        2 * (result["deaths"] + 1),
    ) / 2
    valid = result["liveBirths"] > 0
    result["lower95"] = np.where(
        valid, lower_counts / result["liveBirths"] * MMR_MULTIPLIER, np.nan
    )
    result["upper95"] = np.where(
        valid, upper_counts / result["liveBirths"] * MMR_MULTIPLIER, np.nan
    )
    return result


def _summarize_run_rates(frame: pd.DataFrame, group_columns: list[str], ci_method: str) -> pd.DataFrame:
    if ci_method not in SUPPORTED_CI_METHODS:
        raise ValueError(f"Unsupported confidence interval method: {ci_method}")

    if ci_method == "mean_per_run_poisson_bounds":
        bounded = _add_poisson_bounds(frame)
        summary = bounded.groupby(group_columns, as_index=False, sort=False).agg(
            mmr=("mmr", "mean"),
            lower95=("lower95", "mean"),
            upper95=("upper95", "mean"),
        )
    else:
        grouped = frame.groupby(group_columns, as_index=False, sort=False)["mmr"]
        summary = grouped.mean().rename(columns={"mmr": "mmr"})
        lower = grouped.quantile(0.025).rename(columns={"mmr": "lower95"})
        upper = grouped.quantile(0.975).rename(columns={"mmr": "upper95"})
        summary = summary.merge(lower, on=group_columns).merge(upper, on=group_columns)

    numeric = ["mmr", "lower95", "upper95"]
    summary[numeric] = summary[numeric].round(2)
    return summary


def monthly_mmr_by_level(
    monthly_outcomes: pd.DataFrame,
    n_months: int,
    ci_method: str = "mean_per_run_poisson_bounds",
) -> dict[str, Any]:
    """Return mean monthly MMR and interval bounds for L2/3, L4, and L5."""
    per_run = _mmr_by_run(monthly_outcomes, n_months)
    summary = _summarize_run_rates(per_run, ["month", "levelId"], ci_method)
    return frame_dataset(
        "monthlyMmrByLevel",
        "Monthly MMR by delivery level",
        summary,
        confidenceIntervalMethod=ci_method,
        scenarioDisplayMode="single",
        zeroDenominatorValue=None,
    )


def period_mmr_by_level(
    monthly_outcomes: pd.DataFrame,
    n_months: int,
    ci_method: str = "mean_per_run_poisson_bounds",
) -> dict[str, Any]:
    """Return the mean of run-level MMR values over the full simulation period."""
    monthly = _mmr_by_run(monthly_outcomes, n_months)
    per_run = monthly.groupby(["run", "levelId"], as_index=False, sort=False).agg(
        deaths=("deaths", "sum"),
        liveBirths=("liveBirths", "sum"),
    )
    valid = per_run["liveBirths"] > 0
    per_run["mmr"] = np.where(
        valid,
        per_run["deaths"] / per_run["liveBirths"] * MMR_MULTIPLIER,
        np.nan,
    )
    summary = _summarize_run_rates(per_run, ["levelId"], ci_method)
    return frame_dataset(
        "periodMmrByLevel",
        "Average Ratio Over Full Simulation Period",
        summary,
        aggregationPeriod="full_simulation",
        acrossRuns="arithmetic_mean_of_run_level_ratios",
        confidenceIntervalMethod=ci_method,
        zeroDenominatorValue=None,
    )


def maternal_deaths_by_cause(
    individual_outcomes: pd.DataFrame,
    number_of_runs: int,
) -> dict[str, Any]:
    """Return pooled cause proportions and mean death counts per run."""
    required = {"death_cause"}
    missing = required - set(individual_outcomes.columns)
    if missing:
        raise KeyError(f"Missing required individual-result columns: {sorted(missing)}")
    if number_of_runs < 1:
        raise ValueError("number_of_runs must be at least 1")

    attributed = individual_outcomes.loc[
        individual_outcomes["death_cause"].isin([cause_id for cause_id, _ in DEATH_CAUSES]),
        "death_cause",
    ]
    counts = attributed.value_counts().to_dict()
    total = int(len(attributed))
    rows = []
    for cause_id, cause_label in DEATH_CAUSES:
        pooled_count = int(counts.get(cause_id, 0))
        rows.append(
            {
                "deathCauseId": cause_id,
                "deathCauseLabel": cause_label,
                "pooledCount": pooled_count,
                "meanCountPerRun": round(pooled_count / number_of_runs, 2),
                "proportionOfAttributedDeaths": (
                    round(pooled_count / total, 6) if total else None
                ),
            }
        )
    return frame_dataset(
        "maternalDeathsByCause",
        "Maternal deaths by cause",
        pd.DataFrame(rows),
        aggregationMethod="pooled_across_runs",
        excludedCauseValues=["none"],
    )


def build_mmr_datasets(
    monthly_outcomes: pd.DataFrame,
    individual_outcomes: pd.DataFrame,
    n_months: int,
    number_of_runs: int,
    ci_method: str = "mean_per_run_poisson_bounds",
) -> list[dict[str, Any]]:
    """Build every dataset currently required by the MMR plot catalog."""
    return [
        monthly_mmr_by_level(monthly_outcomes, n_months, ci_method),
        period_mmr_by_level(monthly_outcomes, n_months, ci_method),
        maternal_deaths_by_cause(individual_outcomes, number_of_runs),
    ]
