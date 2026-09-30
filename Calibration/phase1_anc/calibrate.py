"""Run the shared Phase 1 ANC calibration for one county.

The file follows the calibration sequence directly:
1. Load parameters and targets.
2. Simulate ANC and initial delivery location.
3. Define the objective.
4. Run the optimizer.
5. Save reproducible results.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import optuna
import pandas as pd

import parameter_loader
from LB_effect import f_ANC_LB_effect_vectorized
from global_func import reset_E, reset_flags, reset_HSS, reset_S

from .config import (
    COUNTY_CONFIG,
    FIXED_TO_KAKAMEGA,
    KAKAMEGA_REFERRAL_RELATIONSHIP,
    PARAMETERS_TO_FIT,
    SHARED_SETTINGS,
)


# ---------------------------------------------------------------------------
# 1. Parameters and targets
# ---------------------------------------------------------------------------

def load_calibration_inputs(
    county: str,
    workbook_path: Path,
) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    """Load model parameters and configured targets for one county."""
    county = county.strip().lower()
    if county not in COUNTY_CONFIG:
        choices = ", ".join(sorted(COUNTY_CONFIG))
        raise ValueError(f"Unsupported county {county!r}. Choose from: {choices}")

    workbook_path = workbook_path.expanduser().resolve()
    if not workbook_path.exists():
        raise FileNotFoundError(f"Parameter workbook not found: {workbook_path}")

    parameter_loader.WORKBOOK_PATH = workbook_path
    county_param = parameter_loader.get_parameters(
        county=county,
        seed=SHARED_SETTINGS["simulation_seed"],
    )
    kakamega_param = parameter_loader.get_parameters(
        county="kakamega",
        seed=SHARED_SETTINGS["simulation_seed"],
    )
    for name in FIXED_TO_KAKAMEGA:
        county_param[name] = kakamega_param[name]

    # Match the runtime initialization used by the dashboard and scenario
    # runner before calling the production ANC pathway.
    county_param = parameter_loader.calculate_derived_parameters(county_param)
    slider_params = parameter_loader.get_slider_params(county=county)
    county_param.update(
        {
            "E": reset_E(),
            "S": reset_S(slider_params),
            "HSS": reset_HSS(slider_params),
        }
    )

    config = COUNTY_CONFIG[county]
    targets = _load_target_map(
        county_param,
        config["objective_targets"],
        required=True,
    )
    diagnostics = _load_target_map(
        county_param,
        config.get("diagnostic_targets", {}),
    )
    return county_param, targets, diagnostics


def _load_target_map(
    param: Mapping[str, Any],
    target_map: Mapping[str, str],
    *,
    required: bool = False,
) -> dict[str, float]:
    missing = [
        name
        for name in target_map.values()
        if name not in param or pd.isna(param[name])
    ]
    if required and missing:
        raise KeyError(
            f"Missing required target(s): {', '.join(missing)}. "
            "Add these rows to the calibration_targets sheet before running."
        )
    return {
        metric: float(param[name])
        for metric, name in target_map.items()
        if name not in missing
    }


# ---------------------------------------------------------------------------
# 2. Phase 1 ANC simulation
# ---------------------------------------------------------------------------

def evaluate_candidate(
    param: Mapping[str, Any],
    fitted: Mapping[str, float],
    seeds: np.ndarray,
) -> dict[str, float]:
    """Run the production ANC pathway and average its Phase 1 metrics."""
    calibration_param = copy.deepcopy(param)
    calibration_param.update(fitted)
    calibration_param["L23_highrisk"] = calculate_l23_highrisk(
        fitted["close_to_L23"]
    )

    runs = [
        run_phase1_once(calibration_param, int(seed)) for seed in seeds
    ]
    return {
        name: float(np.mean([run[name] for run in runs]))
        for name in runs[0]
    }


def run_phase1_once(
    calibration_param: Mapping[str, Any],
    seed: int,
) -> dict[str, float]:
    """Call the model's baseline ANC and initial-location implementation."""
    rng = np.random.default_rng(seed)
    model_param = dict(calibration_param)
    track = parameter_loader.reset_inputs(model_param, n_months=1)
    flags = reset_flags()
    individual, *_ = f_ANC_LB_effect_vectorized(
        track=track,
        LB_base=track["LB_Track"][0],
        param=model_param,
        flags=flags,
        i=0,
        int_period=2,
        rng=rng,
    )
    return _calculate_metrics(
        locations=individual["i_loc"].to_numpy(),
        anc=individual["i_ANC"].to_numpy(),
    )


def calculate_l23_highrisk(close_to_l23: float) -> float:
    """Preserve Kakamega's high-risk versus routine higher-level referral OR."""
    kakamega_l45_routine = (
        1.0 - KAKAMEGA_REFERRAL_RELATIONSHIP["close_to_L23_routine"]
    )
    kakamega_l45_highrisk = (
        1.0 - KAKAMEGA_REFERRAL_RELATIONSHIP["L23_highrisk"]
    )
    referral_or = (
        kakamega_l45_highrisk / (1.0 - kakamega_l45_highrisk)
    ) / (
        kakamega_l45_routine / (1.0 - kakamega_l45_routine)
    )
    p_l45_routine = 1.0 - close_to_l23
    odds_l45_highrisk = referral_or * p_l45_routine / (1.0 - p_l45_routine)
    return 1.0 - odds_l45_highrisk / (1.0 + odds_l45_highrisk)


def _calculate_metrics(locations: np.ndarray, anc: np.ndarray) -> dict[str, float]:
    all_shares = np.bincount(locations, minlength=4) / len(locations)

    def anc_share(location_mask: np.ndarray) -> float:
        return float(np.mean(anc[location_mask])) if np.any(location_mask) else np.nan

    return {
        "home_all": float(all_shares[0]),
        "l23_all": float(all_shares[1]),
        "l4_all": float(all_shares[2]),
        "l5_all": float(all_shares[3]),
        "l45_all": float(all_shares[2] + all_shares[3]),
        "anc_among_home": anc_share(locations == 0),
        "anc_among_l23": anc_share(locations == 1),
        "anc_among_l4": anc_share(locations == 2),
        "anc_among_l5": anc_share(locations == 3),
        "anc_among_l45": anc_share(locations >= 2),
    }


# ---------------------------------------------------------------------------
# 3. Objective
# ---------------------------------------------------------------------------

def normalized_rmse(
    simulated: Mapping[str, float],
    targets: Mapping[str, float],
    relative_tolerance: float,
) -> float:
    errors = []
    for name, target in targets.items():
        accepted_error = max(abs(target) * relative_tolerance, 1e-6)
        errors.append(((simulated[name] - target) / accepted_error) ** 2)
    return float(np.sqrt(np.mean(errors)))


def build_objective(
    param: Mapping[str, Any],
    targets: Mapping[str, float],
    seeds: np.ndarray,
):
    def objective(trial: optuna.Trial) -> float:
        fitted = {
            name: trial.suggest_float(
                name, float(spec["lower"]), float(spec["upper"])
            )
            for name, spec in PARAMETERS_TO_FIT.items()
        }
        simulated = evaluate_candidate(param, fitted, seeds)
        for name, value in simulated.items():
            trial.set_user_attr(name, value)
        trial.set_user_attr(
            "L23_highrisk", calculate_l23_highrisk(fitted["close_to_L23"])
        )
        return normalized_rmse(
            simulated, targets, float(SHARED_SETTINGS["relative_tolerance"])
        )

    return objective


# ---------------------------------------------------------------------------
# 4. Optimizer
# ---------------------------------------------------------------------------

def stop_when_target_reached(
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
) -> None:
    """Stop once a completed trial reaches the configured RMSE target."""
    threshold = float(SHARED_SETTINGS["early_stopping_rmse"])
    if trial.state == optuna.trial.TrialState.COMPLETE and trial.value < threshold:
        study.stop()

def run_calibration(
    county: str,
    workbook_path: Path,
    output_root: Path,
    *,
    trials: int,
    repetitions: int,
) -> Path:
    param, targets, diagnostics = load_calibration_inputs(
        county, workbook_path
    )
    seeds = np.random.default_rng(
        SHARED_SETTINGS["simulation_seed"]
    ).integers(0, 1_000_000, size=repetitions)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=SHARED_SETTINGS["optimizer_seed"], multivariate=True
        ),
    )
    study.enqueue_trial({name: float(param[name]) for name in PARAMETERS_TO_FIT})
    study.optimize(
        build_objective(param, targets, seeds),
        n_trials=trials,
        callbacks=[stop_when_target_reached],
    )

    fitted = {name: float(value) for name, value in study.best_params.items()}
    simulated = evaluate_candidate(param, fitted, seeds)
    return save_results(
        study=study,
        county=county,
        output_root=output_root,
        targets=targets,
        diagnostic_targets=diagnostics,
        simulated=simulated,
        fitted=fitted,
    )


# ---------------------------------------------------------------------------
# 5. Results
# ---------------------------------------------------------------------------

def save_results(
    *,
    study: optuna.Study,
    county: str,
    output_root: Path,
    targets: Mapping[str, float],
    diagnostic_targets: Mapping[str, float],
    simulated: Mapping[str, float],
    fitted: Mapping[str, float],
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    result_dir = output_root / county / timestamp
    result_dir.mkdir(parents=True, exist_ok=False)

    parameters = [
        {
            "parameter": name,
            "value": value,
            "role": "fitted",
            "lower_bound": PARAMETERS_TO_FIT[name]["lower"],
            "upper_bound": PARAMETERS_TO_FIT[name]["upper"],
        }
        for name, value in fitted.items()
    ]
    parameters.append(
        {
            "parameter": "L23_highrisk",
            "value": calculate_l23_highrisk(fitted["close_to_L23"]),
            "role": "derived",
            "lower_bound": np.nan,
            "upper_bound": np.nan,
        }
    )
    pd.DataFrame(parameters).to_csv(
        result_dir / "calibrated_parameters.csv", index=False
    )

    target_rows = [
        (name, "objective", value) for name, value in targets.items()
    ] + [
        (name, "diagnostic", value)
        for name, value in diagnostic_targets.items()
    ]
    comparison = pd.DataFrame(target_rows, columns=["metric", "role", "target"])
    comparison["simulated"] = comparison["metric"].map(simulated)
    comparison["absolute_error"] = comparison["simulated"] - comparison["target"]
    comparison["relative_error"] = (
        comparison["absolute_error"] / comparison["target"]
    )
    comparison.to_csv(result_dir / "target_comparison.csv", index=False)
    study.trials_dataframe().to_csv(result_dir / "trials.csv", index=False)
    return result_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--county", required=True, choices=sorted(COUNTY_CONFIG))
    parser.add_argument(
        "--workbook", type=Path, default=Path("data/SDR Parameters.xlsx")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Calibration/phase1_anc/results"),
    )
    parser.add_argument(
        "--trials", type=int, default=SHARED_SETTINGS["optimizer_trials"]
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=SHARED_SETTINGS["simulation_repetitions"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 1 or args.repetitions < 1:
        raise ValueError("--trials and --repetitions must both be positive")
    result_dir = run_calibration(
        county=args.county,
        workbook_path=args.workbook,
        output_root=args.output_dir,
        trials=args.trials,
        repetitions=args.repetitions,
    )
    print(f"Calibration results: {result_dir.resolve()}")


if __name__ == "__main__":
    main()
