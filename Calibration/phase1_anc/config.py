"""Transparent configuration for the shared Phase 1 ANC calibration."""

from __future__ import annotations


# These are the only parameters adjusted by the optimizer.
PARAMETERS_TO_FIT = {
    "home_lowrisk": {
        "lower": 0.0,
        "upper": 0.4,
        "description": "P(home | ANC and routine/low-risk pathway)",
    },
    "close_to_L23": {
        "lower": 0.05,
        "upper": 0.98,
        "description": "P(L2/3 | facility delivery on routine ANC pathway)",
    },
}


# Traditional risk-recognition performance remains anchored to Kakamega until
# county-specific validation data are available.
FIXED_TO_KAKAMEGA = (
    "sen_risk_trad",
    "spec_risk_trad",
)


SHARED_SETTINGS = {
    "relative_tolerance": 0.05,
    "early_stopping_rmse": 0.5,
    "simulation_repetitions": 10,
    "simulation_seed": 2025,
    "optimizer_seed": 42,
    "optimizer_trials": 500,
}


KAKAMEGA_REFERRAL_RELATIONSHIP = {
    "close_to_L23_routine": 0.8889440069942813,
    "L23_highrisk": 0.3255,
}


# Keys are simulated metric names. Values are target rows loaded from the
# workbook's calibration_targets sheet by parameter_loader.py.
COMBINED_L45_TARGETS = {
    "home_all": "home_all_target",
    "l23_all": "l23_all_target",
    "l45_all": "l45_all_target",
}

COMBINED_L45_DIAGNOSTICS = {
    "anc_among_home": "home_anc_target",
    "anc_among_l23": "l23_anc_target",
    "anc_among_l45": "l45_anc_target",
}


COUNTY_CONFIG = {
    "kakamega": {
        "objective_targets": COMBINED_L45_TARGETS,
        "diagnostic_targets": COMBINED_L45_DIAGNOSTICS,
        "notes": (
            "Reference county. Retain the combined L4/5 objective until the "
            "shared runner reproduces the historical calibration."
        ),
    },
    "kisii": {
        "objective_targets": COMBINED_L45_TARGETS,
        "diagnostic_targets": COMBINED_L45_DIAGNOSTICS,
        "notes": "Review external L5 referral-in treatment before final calibration.",
    },
    "makueni": {
        "objective_targets": COMBINED_L45_TARGETS,
        "diagnostic_targets": COMBINED_L45_DIAGNOSTICS,
        "notes": "Review transfer-out treatment before final calibration.",
    },
    "mombasa": {
        "objective_targets": {
            "home_all": "home_all_target",
            "l23_all": "l23_all_target",
            "l4_all": "l4_all_target",
        },
        "diagnostic_targets": {
            "l5_all": "l5_all_target",
            "anc_among_home": "home_anc_target",
            "anc_among_l23": "l23_anc_target",
            "anc_among_l4": "l4_anc_target",
            "anc_among_l5": "l5_anc_target",
        },
        "notes": (
            "Observed L5 deliveries include external referrals. L4 anchors the "
            "internal higher-level pathway; L5 remains diagnostic. Prepare the "
            "split targets before calibration with data/phase1_anc_targets.ipynb."
        ),
    },
}
