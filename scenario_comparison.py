from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import parameter_loader
from cost import calculate_sdr_costs, get_cost_parameters, summarize_by_t_ci
from global_func import get_P_l45, reset_E, reset_HSS, reset_S, reset_flags
from model_run import run_model_dash
from parameter_loader import calculate_derived_parameters, get_parameters, get_slider_params


WORKBOOK_PATH = Path(
    "/Users/poppy/Library/CloudStorage/OneDrive-SharedLibraries-JohnsHopkins"
    "/Meibin Chen - MOMISH interventions/SDR Parameters.xlsx"
)
COUNTIES = ["kakamega", "mombasa", "makueni", "kisii"]
N_MONTHS = 36
SKIP_INITIAL_RUNS = 200
ANALYSIS_RUNS = 50
FIRST_ANALYSIS_RUN = SKIP_INITIAL_RUNS + 1
N_RUNS = SKIP_INITIAL_RUNS + ANALYSIS_RUNS
BASE_SEED = 4200
COST_EFFECTIVENESS_N_MONTHS = get_cost_parameters()["time_horizon_years"] * 12
ENABLED_SINGLE_INTERVENTION_SUPPLY = 0.95
PROMPTS_HIGH_SENS_TRAD = 0.95
BASELINE_COST_OUTPUT_ROOT = Path("DALY_SDR_Sep_23")


def clip01(value):
    return float(np.clip(value, 0.0, 1.0))


CURRENT_IMPLEMENTATION_INDEX = {
    "prompts": {
        "kakamega": 0.0,
        "kisii": 0.151,
        "mombasa": 0.187,
        "makueni": 0.116,
    },
    "mentors": {
        "kakamega": 0.0,
        "kisii": 0.0713,
        "mombasa": 0.0583,
        "makueni": 0.1099,
    },
    "referral": {
        "kakamega": 0.0,
        "kisii": 0.0,
        "mombasa": 0.18,
        "makueni": 0.0,
    },
    "pulse": {
        "kakamega": 0.0,
        "kisii": 0.0455,
        "mombasa": 0.0126,
        "makueni": 0.003,
    },
    "fqa": {
        "kakamega": 0.0,
        "kisii": 0.0165,
        "mombasa": 0.0067,
        "makueni": 0.0296,
    },
}


IMPLEMENTATION_INDEX_BY_LEVEL = {
    "moderate": 0.50,
    "high": 0.95,
}


PROMPTS_RR_BY_LEVEL = {
    "current": 1.02,
    "moderate": 1.18,
    "high": 1.35,
}


BLOOD_ADOPTION_BY_LEVEL = {
    "current": 0.15,
    "moderate": 0.30,
    "high": 0.50,
}


SCENARIOS_ALL = [
    {"scenario": "mentors_base", "intervention": "MENTORS", "level": "base", "description": "No HSS; MENTORS off"},
    {"scenario": "mentors_current", "intervention": "MENTORS", "level": "current", "description": "County-specific current implementation index; all single interventions on"},
    {"scenario": "mentors_moderate", "intervention": "MENTORS", "level": "moderate", "description": "50% implementation index; all single interventions on"},
    {"scenario": "mentors_high", "intervention": "MENTORS", "level": "high", "description": "95% implementation index; all single interventions on"},
    {"scenario": "prompts_base", "intervention": "PROMPTS", "level": "base", "description": "All off"},
    {"scenario": "prompts_current", "intervention": "PROMPTS", "level": "current", "description": "County-specific current implementation index; 1.02 RR effectiveness"},
    {"scenario": "prompts_moderate", "intervention": "PROMPTS", "level": "moderate", "description": "50% implementation index; 1.18 RR effectiveness"},
    {"scenario": "prompts_high", "intervention": "PROMPTS", "level": "high", "description": "95% implementation index; 1.35 RR effectiveness"},
    {"scenario": "prompts_high_shift", "intervention": "PROMPTS", "level": "high_shift", "description": "PROMPTS high plus conservative delivery shift and matched facility expansion"},
    {"scenario": "blood_base", "intervention": "Blood tracking", "level": "base", "description": "All off"},
    {"scenario": "blood_current", "intervention": "Blood tracking", "level": "current", "description": "Blood tracking on; 15% adoption"},
    {"scenario": "blood_moderate", "intervention": "Blood tracking", "level": "moderate", "description": "Blood tracking on; 30% adoption"},
    {"scenario": "blood_high", "intervention": "Blood tracking", "level": "high", "description": "Blood tracking on; 50% adoption"},
    {"scenario": "referral_base", "intervention": "Transfer & EMT", "level": "base", "description": "All off"},
    {"scenario": "referral_current", "intervention": "Transfer & EMT", "level": "current", "description": "County-specific referral_implementation_index; 1 hour delay shift"},
    {"scenario": "referral_moderate", "intervention": "Transfer & EMT", "level": "moderate", "description": "50% referral_implementation_index; 1 hour delay shift"},
    {"scenario": "referral_high", "intervention": "Transfer & EMT", "level": "high", "description": "95% referral_implementation_index; 1 hour delay shift"},
    {"scenario": "pulse_base", "intervention": "PULSE", "level": "base", "description": "All off"},
    {"scenario": "pulse_current", "intervention": "PULSE", "level": "current", "description": "County-specific current implementation index"},
    {"scenario": "pulse_moderate", "intervention": "PULSE", "level": "moderate", "description": "50% implementation index"},
    {"scenario": "pulse_high", "intervention": "PULSE", "level": "high", "description": "95% implementation index"},
    {"scenario": "fqa_base", "intervention": "FQA", "level": "base", "description": "PULSE current; FQA off; all single interventions on"},
    {"scenario": "fqa_current", "intervention": "FQA", "level": "current", "description": "County-specific current FQA index; PULSE current; all single interventions on; 10% amplification"},
    {"scenario": "fqa_moderate", "intervention": "FQA", "level": "moderate", "description": "50% FQA implementation index; PULSE current; all single interventions on; 10% amplification"},
    {"scenario": "fqa_high", "intervention": "FQA", "level": "high", "description": "95% FQA implementation index; PULSE current; all single interventions on; 30% amplification"},
    {"scenario": "combined_base", "intervention": "All", "level": "base", "description": "No HSS; all MOMISH interventions off, all single interventions off"},
    {"scenario": "combined_current", "intervention": "All", "level": "current", "description": "County-specific current implementation index for all interventions; all single interventions on; Blood tracking 15% adoption; Transfer & EMT 1 hour delay shift; PROMPTS 1.02 RR effectiveness; FQA 10% amplification; pulse_implementation_boost=0.25"},
    {"scenario": "combined_moderate", "intervention": "All", "level": "moderate", "description": "50% implementation index for all interventions; all single interventions on; Blood tracking 30% adoption; Transfer & EMT 1 hour delay shift; PROMPTS 1.18 RR effectiveness; FQA 10% amplification; pulse_implementation_boost=0.5"},
    {"scenario": "combined_high", "intervention": "All", "level": "high", "description": "95% implementation index for all interventions; all single interventions on; Blood tracking 50% adoption; Transfer & EMT 1 hour delay shift; PROMPTS 1.35 RR effectiveness; FQA 30% amplification; pulse_implementation_boost=0.75"},
]

# Keep the full Aug 19 scenario set above for reference/reuse. The active list
# below reproduces the compact direct-diagnosis/POCUS comparison outputs.
SCENARIOS = [
    {
        "scenario": "diagnosis_base",
        "intervention": "Baseline",
        "level": "base",
        "description": "No diagnosis intervention flags: flag_us=0, flag_intrasensor=0, flag_sensor_ai=0",
    },
    {
        "scenario": "pocus",
        "intervention": "POCUS",
        "level": "direct_flags",
        "description": "flag_us=1, flag_intrasensor=0, flag_sensor_ai=0",
    },
    {
        "scenario": "intrapartum_sensors",
        "intervention": "Intrapartum sensors",
        "level": "direct_flags",
        "description": "flag_us=0, flag_intrasensor=1, flag_sensor_ai=0",
    },
    {
        "scenario": "ai_intrapartum_sensors",
        "intervention": "AI Intrapartum sensors",
        "level": "direct_flags",
        "description": "flag_us=0, flag_intrasensor=1, flag_sensor_ai=1",
    },
]

DALY_COUNTIES = COUNTIES
LABOR_COUNTIES = ["kakamega"]
COST_EFFECTIVENESS_COUNTIES = []
COST_DALY_COUNTIES = COUNTIES
POCUS_QUANTITY_METHOD = "num_L2/3 + num_L4 + num_L5 from get_parameters(county, seed=BASE_SEED)"

COST_CATEGORY_BY_TYPE = {
    "Program Management": "pm_overhead_cost_usd",
    "Overhead": "pm_overhead_cost_usd",
    "Referral Recurrent": "referral_transfer_cost_usd",
    "Transfer Recurrent": "referral_transfer_cost_usd",
    "Referral (Capital)": "referral_transfer_cost_usd",
    "Transfer (Capital)": "referral_transfer_cost_usd",
    "ANC": "service_delivery_cost_usd",
    "Facility Delivery": "service_delivery_cost_usd",
    "CS Delivery": "service_delivery_cost_usd",
    "Labor (Surgical)": "labor_cost_usd",
    "Labor (Nurse)": "labor_cost_usd",
    "Labor (Anesthetist)": "labor_cost_usd",
    "Infrastructure": "equipment_cost_usd",
    "Equipment (General)": "equipment_cost_usd",
    "Equipment (Doppler)": "equipment_cost_usd",
    "Equipment (CTG)": "equipment_cost_usd",
    "Equipment (POCUS)": "equipment_cost_usd",
}

MAJOR_COST_COLUMNS = [
    "pm_overhead_cost_usd",
    "referral_transfer_cost_usd",
    "service_delivery_cost_usd",
    "labor_cost_usd",
    "equipment_cost_usd",
]

REQUIRED_COST_MONTHLY_COLUMNS = [
    "ANC",
    "Fac non-CS",
    "CS",
    "Free_referrals",
    "Emergency transfers",
    "Surgical_actual",
    "Nurse_actual",
    "Anesthetist_actual",
    "Facility_capacity_actual",
    "Doppler_Actual",
    "CTG_Actual",
    "M_DALYs",
    "N_DALYs",
    "DALYs",
]

# To regenerate the full Aug 19 scenario set instead, set:
# SCENARIOS = SCENARIOS_ALL

# PULSE implementation-index boost, by combined-scenario level (mirrors the
# SDR_Dash.py PULSE_IMPLEMENTATION_BOOST_OPTIONS Current/Moderate/High values).
COMBINED_PULSE_BOOST_BY_LEVEL = {
    "current": 0.25,
    "moderate": 0.50,
    "high": 0.75,
}


def enable_single_interventions(param, flags):
    treatment_flags = {
        "flag_pph_bundle": "pph_bundle",
        "flag_iv_iron": "iv_iron",
        "flag_MgSO4": "MgSO4",
        "flag_antibiotics": "antibiotics",
        "flag_oxytocin": "oxytocin",
    }
    for flag, supply in treatment_flags.items():
        flags[flag] = 1
        param["S"][supply] = ENABLED_SINGLE_INTERVENTION_SUPPLY


def apply_prompts_sensitivity(param, implementation_index):
    baseline = float(param["sen_risk_trad"])
    scale = clip01(implementation_index / IMPLEMENTATION_INDEX_BY_LEVEL["high"])
    effective_sensitivity = baseline + (PROMPTS_HIGH_SENS_TRAD - baseline) * scale
    param["sen_risk_trad"] = clip01(effective_sensitivity)
    param["sen_risk_trad_target"] = param["sen_risk_trad"]


def apply_pulse_implementation_boost(param, flags):
    """Mirror SDR_Dash.py's sync_param_momish_from_hss PULSE boost: lend PULSE's
    implementation index to PROMPTS, MENTORS, FQA, and REFERRAL (not Blood, not
    itself). run_model_dash() never calls the dashboard's sync step, so this
    has to be applied explicitly here for scenarios that set flag_pulse.
    """
    if not flags.get("flag_pulse", 0):
        return
    pulse_implementation_index = float(param.get("pulse_implementation_index", 0.0))
    pulse_implementation_boost = float(param.get("pulse_implementation_boost", 0.0))
    fqa_pulse_modifier = float(param.get("fqa_pulse_modifier", 0.0))
    flag_fqa = float(flags.get("flag_fqa", 0))
    param["fqa_pulse_amplifier_index"] = float(param.get("fqa_implementation_index", 0.0))

    pulse_implementation_boost_effective = clip01(
        pulse_implementation_index * pulse_implementation_boost * (1 + flag_fqa * fqa_pulse_modifier)
    )
    for key in (
        "prompts_implementation_index",
        "mentors_implementation_index",
        "fqa_implementation_index",
        "referral_implementation_index",
    ):
        param[key] = clip01(param.get(key, 0.0) + pulse_implementation_boost_effective)


def current_implementation(intervention, county):
    return CURRENT_IMPLEMENTATION_INDEX[intervention].get(county, 0.0)


def implementation_for_level(intervention, level, county):
    if level == "current":
        return current_implementation(intervention, county)
    return IMPLEMENTATION_INDEX_BY_LEVEL[level]


def configure_conservative_shift(param, flags, slider_params):
    flags["flag_CHV"] = 1
    flags["flag_ANC"] = 1
    flags["flag_LB"] = 1
    flags["flag_capacity"] = 1

    param["HSS"]["P_ANC"] = 0.70
    p_l45_exp = get_P_l45(param["HSS"]["P_ANC"], slider_params)
    min_l45 = p_l45_exp if p_l45_exp is not None else 0.0
    param["HSS"]["P_L45"] = max(min_l45, 0.53)
    param["HSS"]["CHV_memory"] = "Logistic Decay"
    param["HSS"]["tau_decay"] = 6
    param["HSS"]["capacity_added"] = 0.25


def configure_scenario(name, param, flags, slider_params=None):
    hss = param["HSS"]
    county = param.get("county", "kakamega")

    if name.endswith("_base") and not name.startswith("fqa_"):
        if name == "fqa_base":
            flags["flag_pulse"] = 1
            pulse_index = current_implementation("pulse", county)
            hss["pulse_implementation_index"] = pulse_index
            hss["fqa_implementation_index"] = 0.0
            param["pulse_implementation_index"] = pulse_index
            param["fqa_implementation_index"] = 0.0
            enable_single_interventions(param, flags)
        return

    if name == "diagnosis_base":
        return

    if name == "pocus":
        flags["flag_us"] = 1
        param["S"]["US"] = 1
        param["E"]["sens_us"] = 0.95
        param["E"]["spec_us"] = 0.95
        return

    if name == "intrapartum_sensors":
        flags["flag_intrasensor"] = 1
        flags["flag_sensor_ai"] = 0
        return

    if name == "ai_intrapartum_sensors":
        flags["flag_intrasensor"] = 1
        flags["flag_sensor_ai"] = 1
        param["E"]["sens_sensor"] = 0.95
        param["E"]["spec_sensor"] = 0.95
        return

    if name.startswith("mentors_"):
        level = name.removeprefix("mentors_")
        implementation_index = implementation_for_level("mentors", level, county)
        flags["flag_MENTOR"] = 1
        hss.update({
            "mentor_implementation_index": implementation_index,
        })
        param["mentors_implementation_index"] = implementation_index
        enable_single_interventions(param, flags)
        return

    if name.startswith("prompts_"):
        level = "high" if name == "prompts_high_shift" else name.removeprefix("prompts_")
        implementation_index = implementation_for_level("prompts", level, county)
        prompts_rr_anc4p = PROMPTS_RR_BY_LEVEL[level]
        flags["flag_PROMPTS"] = 1
        hss.update({
            "prompts_implementation_index": implementation_index,
            "prompts_rr_anc4p": prompts_rr_anc4p,
        })
        param["prompts_implementation_index"] = implementation_index
        param["prompts_rr_anc4p"] = prompts_rr_anc4p
        apply_prompts_sensitivity(param, implementation_index)
        if name == "prompts_high_shift":
            if slider_params is None:
                raise ValueError("slider_params is required for prompts_high_shift")
            configure_conservative_shift(param, flags, slider_params)
        return

    if name.startswith("blood_"):
        level = name.removeprefix("blood_")
        adoption = BLOOD_ADOPTION_BY_LEVEL[level]
        flags["flag_blood"] = 1
        flags["flag_blood_tracking"] = 1
        hss["blood_adoption"] = adoption
        return

    if name.startswith("referral_"):
        level = name.removeprefix("referral_")
        implementation_index = implementation_for_level("referral", level, county)
        flags["flag_transfer_delay"] = 1
        hss["referral_implementation_index"] = implementation_index
        param["referral_implementation_index"] = implementation_index
        return

    if name.startswith("pulse_"):
        level = name.removeprefix("pulse_")
        implementation_index = implementation_for_level("pulse", level, county)
        flags["flag_pulse"] = 1
        hss["pulse_implementation_index"] = implementation_index
        hss["fqa_implementation_index"] = 0.0
        param["pulse_implementation_index"] = implementation_index
        param["fqa_implementation_index"] = 0.0
        return

    if name.startswith("fqa_"):
        level = name.removeprefix("fqa_")
        pulse_implementation_index = current_implementation("pulse", county)
        flags["flag_pulse"] = 1
        hss["pulse_implementation_index"] = pulse_implementation_index
        param["pulse_implementation_index"] = pulse_implementation_index
        enable_single_interventions(param, flags)
        if level != "base":
            flags["flag_fqa"] = 1
            fqa_implementation_index = implementation_for_level("fqa", level, county)
            fqa_pulse_modifier = 0.30 if level == "high" else 0.10
            hss["fqa_pulse_modifier_level"] = "High" if level == "high" else "Low"
            hss["fqa_pulse_modifier"] = fqa_pulse_modifier
            hss["fqa_implementation_index"] = fqa_implementation_index
            param["fqa_pulse_modifier"] = fqa_pulse_modifier
            param["fqa_implementation_index"] = fqa_implementation_index
        else:
            hss["fqa_implementation_index"] = 0.0
            param["fqa_implementation_index"] = 0.0
        return

    if name.startswith("combined_"):
        # combined_base is handled by the generic "_base" branch above (all off).
        level = name.removeprefix("combined_")

        mentors_index = implementation_for_level("mentors", level, county)
        flags["flag_MENTOR"] = 1
        hss["mentor_implementation_index"] = mentors_index
        param["mentors_implementation_index"] = mentors_index

        prompts_index = implementation_for_level("prompts", level, county)
        prompts_rr_anc4p = PROMPTS_RR_BY_LEVEL[level]
        flags["flag_PROMPTS"] = 1
        hss.update({
            "prompts_implementation_index": prompts_index,
            "prompts_rr_anc4p": prompts_rr_anc4p,
        })
        param["prompts_implementation_index"] = prompts_index
        param["prompts_rr_anc4p"] = prompts_rr_anc4p
        apply_prompts_sensitivity(param, prompts_index)

        adoption = BLOOD_ADOPTION_BY_LEVEL[level]
        flags["flag_blood"] = 1
        flags["flag_blood_tracking"] = 1
        hss["blood_adoption"] = adoption

        referral_index = implementation_for_level("referral", level, county)
        flags["flag_transfer_delay"] = 1
        hss["referral_implementation_index"] = referral_index
        param["referral_implementation_index"] = referral_index

        pulse_index = implementation_for_level("pulse", level, county)
        pulse_boost = COMBINED_PULSE_BOOST_BY_LEVEL[level]
        flags["flag_pulse"] = 1
        hss["pulse_implementation_index"] = pulse_index
        hss["pulse_implementation_boost"] = pulse_boost
        param["pulse_implementation_index"] = pulse_index
        param["pulse_implementation_boost"] = pulse_boost

        fqa_index = implementation_for_level("fqa", level, county)
        fqa_pulse_modifier = 0.30 if level == "high" else 0.10
        flags["flag_fqa"] = 1
        hss["fqa_pulse_modifier_level"] = "High" if level == "high" else "Low"
        hss["fqa_pulse_modifier"] = fqa_pulse_modifier
        hss["fqa_implementation_index"] = fqa_index
        param["fqa_pulse_modifier"] = fqa_pulse_modifier
        param["fqa_implementation_index"] = fqa_index

        enable_single_interventions(param, flags)
        apply_pulse_implementation_boost(param, flags)
        return

    raise ValueError(f"Unknown scenario: {name}")


def safe_rate(df, column, mask=None):
    if column not in df:
        return np.nan
    values = df[column] if mask is None else df.loc[mask, column]
    return float(values.mean()) if len(values) else np.nan


def summarize(individuals):
    transfer = individuals["i_transfer"] == 1
    facility = individuals["i_loc_new_v2"] > 0
    anemia_with_anc = (individuals["i_anemia"] == 1) & (individuals["i_ANC"] == 1)
    pph_case = individuals["i_pph"] == 1
    eclampsia_case = individuals["i_eclampsia"] == 1
    sepsis_case = individuals["i_mat_sepsis"] == 1
    ol_case = individuals["i_OL"] == 1
    return {
        "anc": safe_rate(individuals, "i_ANC"),
        "initial_l45": float((individuals["i_loc"] >= 2).mean()),
        "final_l45": float((individuals["i_loc_new_v2"] >= 2).mean()),
        "free_referral": safe_rate(individuals, "i_free_referral"),
        "self_referral": safe_rate(individuals, "i_self_referral"),
        "emergency_transfer": safe_rate(individuals, "i_transfer"),
        "delay_lt1h_given_transfer": float((individuals.loc[transfer, "travel_time_transfer"] == 0).mean()),
        "delay_1_2h_given_transfer": float((individuals.loc[transfer, "travel_time_transfer"] == 1).mean()),
        "delay_2plus_given_transfer": float((individuals.loc[transfer, "travel_time_transfer"] == 2).mean()),
        "iv_iron": safe_rate(individuals, "i_iv_iron"),
        "pph_bundle": safe_rate(individuals, "i_pph_bundle"),
        "mgso4": safe_rate(individuals, "i_MgSO4"),
        "antibiotics": safe_rate(individuals, "i_antibiotics"),
        "oxytocin": safe_rate(individuals, "i_oxytocin"),
        "iv_iron_given_anemia_anc": safe_rate(individuals, "i_iv_iron", anemia_with_anc),
        "pph_bundle_given_pph": safe_rate(individuals, "i_pph_bundle", pph_case),
        "mgso4_given_eclampsia": safe_rate(individuals, "i_MgSO4", eclampsia_case),
        "antibiotics_given_sepsis": safe_rate(individuals, "i_antibiotics", sepsis_case),
        "oxytocin_given_ol": safe_rate(individuals, "i_oxytocin", ol_case),
        "pph": safe_rate(individuals, "i_pph_new"),
        "maternal_sepsis": safe_rate(individuals, "i_mat_sepsis_new"),
        "eclampsia": safe_rate(individuals, "i_eclampsia_new"),
        "obstructed_labor": safe_rate(individuals, "i_OL_final"),
        "aph": safe_rate(individuals, "i_aph"),
        "uterine_rupture": safe_rate(individuals, "i_ruptured_uterus"),
        "severe_complication": safe_rate(individuals, "i_severe_new"),
        "maternal_deaths_per_100k": safe_rate(individuals, "i_mat_death") * 100000,
        "facility_maternal_deaths_per_100k": safe_rate(individuals, "i_mat_death", facility) * 100000,
        "neonatal_deaths_per_1000": safe_rate(individuals, "i_neo_death") * 1000,
    }


LOCATION_LABELS = {
    0: "Home",
    1: "L2/3",
    2: "L4",
    3: "L5",
}


def summarize_by_location(individuals):
    """Summarize outcomes by final delivery location for one model run."""
    rows = []
    total_births = len(individuals)
    for location_code, location in LOCATION_LABELS.items():
        at_location = individuals["i_loc_new_v2"] == location_code
        location_df = individuals.loc[at_location]
        births = len(location_df)
        anemia_with_anc = (
            (location_df["i_anemia"] == 1) & (location_df["i_ANC"] == 1)
        )
        rows.append({
            "location": location,
            "location_code": location_code,
            "births": births,
            "delivery_share": births / total_births if total_births else np.nan,
            "anc": safe_rate(location_df, "i_ANC"),
            "pph": safe_rate(location_df, "i_pph_new"),
            "maternal_sepsis": safe_rate(location_df, "i_mat_sepsis_new"),
            "eclampsia": safe_rate(location_df, "i_eclampsia_new"),
            "obstructed_labor": safe_rate(location_df, "i_OL_final"),
            "aph": safe_rate(location_df, "i_aph"),
            "uterine_rupture": safe_rate(location_df, "i_ruptured_uterus"),
            "severe_complication": safe_rate(location_df, "i_severe_new"),
            "maternal_deaths": float(location_df["i_mat_death"].sum()),
            "maternal_deaths_per_100k": (
                safe_rate(location_df, "i_mat_death") * 100000
            ),
            "neonatal_deaths": float(location_df["i_neo_death"].sum()),
            "neonatal_deaths_per_1000": (
                safe_rate(location_df, "i_neo_death") * 1000
            ),
            "iv_iron_given_anemia_anc": safe_rate(
                location_df, "i_iv_iron", anemia_with_anc
            ),
            "pph_bundle_given_pph": safe_rate(
                location_df, "i_pph_bundle", location_df["i_pph"] == 1
            ),
            "mgso4_given_eclampsia": safe_rate(
                location_df, "i_MgSO4", location_df["i_eclampsia"] == 1
            ),
            "antibiotics_given_sepsis": safe_rate(
                location_df, "i_antibiotics", location_df["i_mat_sepsis"] == 1
            ),
            "oxytocin_given_ol": safe_rate(
                location_df, "i_oxytocin", location_df["i_OL"] == 1
            ),
        })
    return rows


def _sum_array(value):
    return float(np.nansum(np.asarray(value, dtype=float)))


def _sum_array_column(df, column):
    if column not in df:
        return np.nan
    return float(df[column].apply(lambda value: np.asarray(value, dtype=float).sum()).sum())


def _mean_array_column(df, column):
    if column not in df:
        return np.nan
    values = np.concatenate([np.asarray(value, dtype=float).ravel() for value in df[column]])
    return float(np.nanmean(values)) if values.size else np.nan


def summarize_daly(individuals):
    maternal = float(individuals["M_DALY"].sum()) if "M_DALY" in individuals else np.nan
    neonatal = float(individuals["N_DALY"].sum()) if "N_DALY" in individuals else np.nan
    total = float(individuals["DALY"].sum()) if "DALY" in individuals else maternal + neonatal
    births = len(individuals)
    return {
        "births": births,
        "maternal_dalys": maternal,
        "neonatal_dalys": neonatal,
        "total_dalys": total,
        "maternal_dalys_per_100k": maternal / births * 100000 if births else np.nan,
        "neonatal_dalys_per_100k": neonatal / births * 100000 if births else np.nan,
        "total_dalys_per_100k": total / births * 100000 if births else np.nan,
    }


def summarize_labor(monthly):
    surgical_actual = _sum_array_column(monthly, "Surgical_actual")
    nurse_actual = _sum_array_column(monthly, "Nurse_actual")
    anesthetist_actual = _sum_array_column(monthly, "Anesthetist_actual")
    surgical_needed = _sum_array_column(monthly, "Surgical_needed")
    nurse_needed = _sum_array_column(monthly, "Nurse_needed")
    anesthetist_needed = _sum_array_column(monthly, "Anesthetist_needed")
    return {
        "surgical_actual_total": surgical_actual,
        "nurse_actual_total": nurse_actual,
        "anesthetist_actual_total": anesthetist_actual,
        "surgical_needed_total": surgical_needed,
        "nurse_needed_total": nurse_needed,
        "anesthetist_needed_total": anesthetist_needed,
        "surgical_gap_total": surgical_needed - surgical_actual,
        "nurse_gap_total": nurse_needed - nurse_actual,
        "anesthetist_gap_total": anesthetist_needed - anesthetist_actual,
        "surgical_ratio_mean": _mean_array_column(monthly, "Surgical_ratio"),
        "nurse_ratio_mean": _mean_array_column(monthly, "Nurse_ratio"),
        "anesthetist_ratio_mean": _mean_array_column(monthly, "Anesthetist_ratio"),
    }


def monthly_with_run(monthly, run):
    out = monthly.copy()
    out["Run"] = int(run)
    return out


def _yearly_daly_sums(monthly, columns):
    rows = monthly_with_year(monthly)
    data = rows[["Run", "Month", "year"]].copy()
    for column in columns:
        data[column] = rows[column].apply(_sum_array)
    return (
        data.groupby(["Run", "year"], as_index=False)[columns]
        .sum()
        .sort_values(["Run", "year"])
    )


def monthly_with_year(monthly):
    out = monthly.copy()
    out["Run"] = out["Run"].astype(int)
    out["Month"] = out["Month"].astype(int)
    out["year"] = (out["Month"] // 12) + 1
    return out


def calculate_total_dalys_averted_by_year(baseline_monthly, intervention_monthly):
    columns = ["M_DALYs", "N_DALYs", "DALYs"]
    baseline = _yearly_daly_sums(baseline_monthly, columns)
    intervention = _yearly_daly_sums(intervention_monthly, columns)
    merged = intervention.merge(
        baseline,
        on=["Run", "year"],
        suffixes=("_intervention", "_baseline"),
    )
    out = merged[["Run", "year"]].copy()
    out["maternal_dalys_averted"] = (
        merged["M_DALYs_baseline"] - merged["M_DALYs_intervention"]
    )
    out["neonatal_dalys_averted"] = (
        merged["N_DALYs_baseline"] - merged["N_DALYs_intervention"]
    )
    out["total_dalys_averted"] = (
        out["maternal_dalys_averted"] + out["neonatal_dalys_averted"]
    )
    out["cumulative_maternal_dalys_averted"] = out.groupby("Run")[
        "maternal_dalys_averted"
    ].cumsum()
    out["cumulative_neonatal_dalys_averted"] = out.groupby("Run")[
        "neonatal_dalys_averted"
    ].cumsum()
    out["cumulative_total_dalys_averted"] = out.groupby("Run")[
        "total_dalys_averted"
    ].cumsum()
    return out


def build_cost_effectiveness_outputs(monthly_by_scenario, scenario_definitions, num_pocus=0):
    if "diagnosis_base" in monthly_by_scenario:
        baseline_name = "diagnosis_base"
    elif "mentors_base" in monthly_by_scenario:
        baseline_name = "mentors_base"
    else:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    baseline_monthly = monthly_by_scenario[baseline_name]
    cost_parameters = get_cost_parameters()
    scenario_lookup = {
        item["scenario"]: item
        for item in scenario_definitions
    }
    run_rows = []
    component_rows = []
    for scenario, intervention_monthly in monthly_by_scenario.items():
        if scenario == baseline_name:
            continue
        scenario_def = scenario_lookup.get(scenario, {})
        cost_results = calculate_sdr_costs(
            baseline_monthly,
            intervention_monthly,
            cost_parameters,
            scenario_name=scenario,
            include_pocus=scenario == "pocus",
            num_pocus=num_pocus,
        )

        total_dalys = calculate_total_dalys_averted_by_year(
            baseline_monthly,
            intervention_monthly,
        )
        icer = cost_results["icer_yearly"].replace([np.inf, -np.inf], np.nan)
        icer = icer.merge(total_dalys, on=["Run", "year"], how="left")
        icer = icer.rename(
            columns={
                "Scenario": "scenario",
                "cost_per_daly_averted": "cost_per_maternal_daly_averted_usd",
                "dalys_averted": "maternal_dalys_averted_existing",
                "cumulative_dalys_averted": "cumulative_maternal_dalys_averted_existing",
            }
        )
        icer["scenario"] = scenario
        icer["intervention"] = scenario_def.get("intervention", "")
        icer["level"] = scenario_def.get("level", "")
        icer["dalys_averted"] = icer["maternal_dalys_averted"]
        icer["cumulative_dalys_averted"] = icer[
            "cumulative_maternal_dalys_averted"
        ]
        icer["cost_per_total_daly_averted_usd"] = (
            icer["cumulative_discounted_cost"] / icer["cumulative_total_dalys_averted"]
        )
        icer.loc[
            icer["cumulative_total_dalys_averted"] <= 0,
            "cost_per_total_daly_averted_usd",
        ] = np.nan
        icer["cost_per_daly_averted"] = icer["cost_per_maternal_daly_averted_usd"]
        run_rows.append(icer)

        components = cost_results["component_yearly"].copy()
        components = components.rename(columns={"Scenario": "scenario"})
        components["scenario"] = scenario
        components["intervention"] = scenario_def.get("intervention", "")
        components["level"] = scenario_def.get("level", "")
        component_rows.append(components)

    run_results = pd.concat(run_rows, ignore_index=True) if run_rows else pd.DataFrame()
    component_results = (
        pd.concat(component_rows, ignore_index=True) if component_rows else pd.DataFrame()
    )
    summary = summarize_cost_effectiveness(run_results)
    return run_results, component_results, summary


def validate_cost_monthly_outputs(monthly_by_scenario):
    for scenario, monthly in monthly_by_scenario.items():
        missing = [
            column for column in REQUIRED_COST_MONTHLY_COLUMNS
            if column not in monthly.columns
        ]
        if missing:
            raise ValueError(
                f"Missing required cost-effectiveness monthly outputs for {scenario}: {missing}"
            )


def build_combined_cost_daly_outputs(monthly_by_scenario, scenario_definitions, num_pocus=0):
    validate_cost_monthly_outputs(monthly_by_scenario)
    if "diagnosis_base" in monthly_by_scenario:
        baseline_name = "diagnosis_base"
    elif "mentors_base" in monthly_by_scenario:
        baseline_name = "mentors_base"
    else:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    baseline_monthly = monthly_by_scenario[baseline_name]
    cost_parameters = get_cost_parameters()
    scenario_lookup = {item["scenario"]: item for item in scenario_definitions}
    run_rows = []
    component_rows = []

    for scenario, intervention_monthly in monthly_by_scenario.items():
        if scenario == baseline_name:
            continue
        scenario_def = scenario_lookup.get(scenario, {})
        cost_results = calculate_sdr_costs(
            baseline_monthly,
            intervention_monthly,
            cost_parameters,
            scenario_name=scenario,
            include_pocus=scenario == "pocus",
            num_pocus=num_pocus if scenario == "pocus" else 0,
        )

        components = cost_results["component_yearly"].copy()
        components = components.rename(columns={"Scenario": "scenario"})
        components["scenario"] = scenario
        components["intervention"] = scenario_def.get("intervention", "")
        components["level"] = scenario_def.get("level", "")
        components["cost_category"] = components["cost_type"].map(COST_CATEGORY_BY_TYPE)
        if components["cost_category"].isna().any():
            unknown = sorted(components.loc[components["cost_category"].isna(), "cost_type"].unique())
            raise ValueError(f"Unmapped cost types for {scenario}: {unknown}")
        component_rows.append(components)

        category_costs = (
            components.groupby(["scenario", "intervention", "level", "Run", "cost_category"], as_index=False)
            .agg(cost_yearly=("cost_yearly", "sum"))
            .pivot_table(
                index=["scenario", "intervention", "level", "Run"],
                columns="cost_category",
                values="cost_yearly",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reset_index()
        )
        category_costs.columns.name = None
        for column in MAJOR_COST_COLUMNS:
            if column not in category_costs.columns:
                category_costs[column] = 0.0
        category_costs["total_cost_usd"] = category_costs[MAJOR_COST_COLUMNS].sum(axis=1)

        discounted = (
            components.groupby(["scenario", "intervention", "level", "Run"], as_index=False)
            .agg(total_discounted_cost_usd=("cost_discounted_yearly", "sum"))
        )
        dalys = calculate_total_dalys_averted_by_year(
            baseline_monthly,
            intervention_monthly,
        )
        final_year = dalys.groupby("Run")["year"].transform("max")
        final_dalys = dalys.loc[
            dalys["year"] == final_year,
            [
                "Run",
                "year",
                "cumulative_maternal_dalys_averted",
                "cumulative_neonatal_dalys_averted",
                "cumulative_total_dalys_averted",
            ],
        ].copy()
        final_dalys = final_dalys.rename(
            columns={
                "year": "final_year",
                "cumulative_maternal_dalys_averted": "maternal_dalys_averted",
                "cumulative_neonatal_dalys_averted": "neonatal_dalys_averted",
                "cumulative_total_dalys_averted": "total_dalys_averted",
            }
        )

        run_result = category_costs.merge(
            discounted,
            on=["scenario", "intervention", "level", "Run"],
            how="left",
        ).merge(final_dalys, on="Run", how="left")
        run_result["num_pocus"] = num_pocus if scenario == "pocus" else 0
        run_result["pocus_quantity_method"] = POCUS_QUANTITY_METHOD if scenario == "pocus" else "Not applicable"
        run_result["cost_category_sum_check_usd"] = (
            run_result[MAJOR_COST_COLUMNS].sum(axis=1) - run_result["total_cost_usd"]
        )
        run_result["total_daly_sum_check"] = (
            run_result["maternal_dalys_averted"]
            + run_result["neonatal_dalys_averted"]
            - run_result["total_dalys_averted"]
        )
        if not np.allclose(run_result["cost_category_sum_check_usd"], 0.0, atol=1e-9):
            raise ValueError(f"Cost category totals do not sum to total_cost_usd for {scenario}")
        if not np.allclose(run_result["total_daly_sum_check"], 0.0, atol=1e-9):
            raise ValueError(f"Maternal + neonatal DALYs do not sum to total DALYs for {scenario}")
        run_result["cost_per_total_daly_averted_usd"] = (
            run_result["total_discounted_cost_usd"] / run_result["total_dalys_averted"]
        )
        run_result.loc[
            run_result["total_dalys_averted"] <= 0,
            "cost_per_total_daly_averted_usd",
        ] = np.nan
        run_rows.append(run_result)

    run_results = pd.concat(run_rows, ignore_index=True) if run_rows else pd.DataFrame()
    component_results = (
        pd.concat(component_rows, ignore_index=True) if component_rows else pd.DataFrame()
    )
    summary = summarize_combined_cost_daly_run_results(run_results)
    return run_results, component_results, summary


def summarize_combined_cost_daly_run_results(run_results):
    if run_results.empty:
        return pd.DataFrame()
    mean_columns = [
        "total_cost_usd",
        *MAJOR_COST_COLUMNS,
        "total_discounted_cost_usd",
        "maternal_dalys_averted",
        "neonatal_dalys_averted",
        "total_dalys_averted",
        "num_pocus",
        "cost_category_sum_check_usd",
        "total_daly_sum_check",
        "final_year",
    ]
    summary = (
        run_results.groupby(["scenario", "intervention", "level"], as_index=False)[mean_columns]
        .mean()
    )
    summary["cost_per_maternal_daly_averted_usd"] = (
        summary["total_discounted_cost_usd"] / summary["maternal_dalys_averted"]
    )
    summary.loc[
        summary["maternal_dalys_averted"] <= 0,
        "cost_per_maternal_daly_averted_usd",
    ] = np.nan
    summary["cost_per_total_daly_averted_usd"] = (
        summary["total_discounted_cost_usd"] / summary["total_dalys_averted"]
    )
    summary.loc[
        summary["total_dalys_averted"] <= 0,
        "cost_per_total_daly_averted_usd",
    ] = np.nan
    ordered_columns = [
        "scenario",
        "intervention",
        "level",
        "total_cost_usd",
        "pm_overhead_cost_usd",
        "referral_transfer_cost_usd",
        "service_delivery_cost_usd",
        "labor_cost_usd",
        "equipment_cost_usd",
        "total_discounted_cost_usd",
        "maternal_dalys_averted",
        "neonatal_dalys_averted",
        "total_dalys_averted",
        "cost_per_maternal_daly_averted_usd",
        "cost_per_total_daly_averted_usd",
        "num_pocus",
        "cost_category_sum_check_usd",
        "total_daly_sum_check",
        "final_year",
    ]
    return summary[ordered_columns]


def cost_daly_calculation_methods():
    rows = [
        ("Total cost (USD)", "= PM/overhead + Referral/transfer + Service delivery + Labor + Equipment"),
        ("PM/overhead cost (USD)", "= Program Management + Overhead"),
        ("Referral/transfer cost (USD)", "= Referral Recurrent + Transfer Recurrent + Referral Capital + Transfer Capital"),
        ("Service delivery cost (USD)", "= ANC + Facility Delivery + CS Delivery"),
        ("Labor cost (USD)", "= Surgical Labor + Nurse Labor + Anesthetist Labor"),
        ("Equipment cost (USD)", "= Infrastructure + General Equipment + Doppler + CTG + POCUS"),
        ("Discounted yearly cost", "= Yearly Cost / (1 + 0.03)^(Year - 1)"),
        ("Total discounted cost (USD)", "= SUM(Discounted Yearly Cost, Years 1-10)"),
        ("Maternal DALYs averted", "= Baseline Maternal DALYs - Intervention Maternal DALYs"),
        ("Neonatal DALYs averted", "= Baseline Neonatal DALYs - Intervention Neonatal DALYs"),
        ("Total DALYs averted", "= Maternal DALYs averted + Neonatal DALYs averted"),
        ("Cost per maternal DALY averted (USD)", "= Mean Total Discounted Cost / Mean Maternal DALYs Averted"),
        ("Cost per total DALY averted (USD)", "= Mean Total Discounted Cost / Mean Total DALYs Averted"),
        ("POCUS equipment cost", "= Number of POCUS Units x $5,000 x Annualization Factor"),
        ("POCUS units", "= num_L2/3 + num_L4 + num_L5 from county parameters"),
        ("Capital equipment annualization", "= Capital cost x annuity factor using useful life and 3% discount rate"),
        ("Doppler/CTG/infrastructure capital", "= Added units or capacity x unit cost x annuity factor using useful life and 3% discount rate"),
    ]
    return pd.DataFrame(rows, columns=["Variable", "Calculation"])


def summarize_cost_effectiveness(run_results):
    if run_results.empty:
        return pd.DataFrame()

    final_year = run_results.groupby(["scenario", "Run"])["year"].transform("max")
    final = run_results.loc[run_results["year"] == final_year].copy()
    final = final.rename(
        columns={
            "cumulative_discounted_cost": "total_discounted_cost_usd",
            "cumulative_maternal_dalys_averted": "maternal_dalys_averted",
            "cumulative_neonatal_dalys_averted": "neonatal_dalys_averted",
            "cumulative_total_dalys_averted": "total_dalys_averted",
        }
    )
    metric_columns = [
        "total_discounted_cost_usd",
        "maternal_dalys_averted",
        "neonatal_dalys_averted",
        "total_dalys_averted",
        "cost_per_maternal_daly_averted_usd",
        "cost_per_total_daly_averted_usd",
    ]
    summaries = []
    for metric in metric_columns:
        metric_frame = final.dropna(subset=[metric])
        if metric_frame.empty:
            continue
        summary = summarize_by_t_ci(
            metric_frame,
            metric,
            ["scenario", "intervention", "level"],
        )
        summary = summary.rename(
            columns={
                "mean": f"{metric}_mean",
                "lower_CI": f"{metric}_lower_CI",
                "upper_CI": f"{metric}_upper_CI",
            }
        )
        summaries.append(summary)
    if not summaries:
        return pd.DataFrame()

    out = summaries[0]
    for summary in summaries[1:]:
        out = out.merge(summary, on=["scenario", "intervention", "level"], how="outer")
    out.insert(
        3,
        "icer_summary_method",
        "Mean of final run-specific cumulative ICERs; not mean(cost)/mean(DALYs).",
    )
    return out


def _find_baseline_name(means, baseline_candidates):
    return next(
        (name for name in baseline_candidates if (means["scenario"] == name).any()),
        None,
    )


def difference_from_baseline(means, metric_columns, baseline_candidates):
    baseline_name = _find_baseline_name(means, baseline_candidates)
    if baseline_name is None:
        return pd.DataFrame(columns=[*means.columns[:3], "comparison_baseline", *metric_columns])
    differences = means.copy()
    baseline = means.loc[means["scenario"] == baseline_name, metric_columns].iloc[0]
    differences[metric_columns] = means[metric_columns].to_numpy() - baseline.to_numpy()
    differences.insert(3, "comparison_baseline", baseline_name)
    return differences


def mean_vs_baseline(means, metric_columns, baseline_candidates):
    baseline_name = _find_baseline_name(means, baseline_candidates)
    if baseline_name is None:
        return pd.DataFrame()
    baseline = means.loc[means["scenario"] == baseline_name, metric_columns].iloc[0]
    comparison = means[["scenario", "intervention", "level"]].copy()
    comparison.insert(3, "comparison_baseline", baseline_name)
    for column in metric_columns:
        comparison[f"{column}_mean"] = means[column].to_numpy()
        comparison[f"{column}_baseline_mean"] = baseline[column]
        comparison[f"{column}_difference_from_baseline"] = (
            means[column].to_numpy() - baseline[column]
        )
    return comparison


def _read_existing_csv(path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def load_existing_run_results(county):
    """Load previously saved per-run results for `county`, if any, so a rerun
    can skip scenarios that were already generated instead of redoing them."""
    results_dir = Path(f"scenario_comparison_results_{county}")
    run_results_path = results_dir / "scenario_run_results.csv"
    location_run_results_path = results_dir / "scenario_location_run_results.csv"
    daly_run_results_path = results_dir / "scenario_daly_run_results.csv"
    labor_run_results_path = results_dir / "scenario_labor_run_results.csv"
    cost_effectiveness_run_results_path = (
        results_dir / "scenario_cost_effectiveness_run_results.csv"
    )
    cost_components_run_results_path = (
        results_dir / "scenario_cost_components_run_results.csv"
    )
    if not run_results_path.exists():
        return None, None, None, None, None, None
    existing_run_results = _read_existing_csv(run_results_path)
    existing_location_run_results = _read_existing_csv(location_run_results_path)
    existing_daly_run_results = _read_existing_csv(daly_run_results_path)
    existing_labor_run_results = _read_existing_csv(labor_run_results_path)
    existing_cost_effectiveness_run_results = _read_existing_csv(
        cost_effectiveness_run_results_path
    )
    existing_cost_components_run_results = _read_existing_csv(
        cost_components_run_results_path
    )
    return (
        existing_run_results,
        existing_location_run_results,
        existing_daly_run_results,
        existing_labor_run_results,
        existing_cost_effectiveness_run_results,
        existing_cost_components_run_results,
    )


def _combine_frames(existing, new):
    frames = [df for df in (existing, new) if df is not None and not df.empty]
    if not frames:
        return new if new is not None else pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_scenarios(county):
    parameter_loader.WORKBOOK_PATH = WORKBOOK_PATH
    slider_params = get_slider_params(county=county)
    county_param_sample = get_parameters(county=county, seed=BASE_SEED)
    num_pocus = (
        county_param_sample["num_L2/3"]
        + county_param_sample["num_L4"]
        + county_param_sample["num_L5"]
    )

    (
        existing_run_results,
        existing_location_run_results,
        existing_daly_run_results,
        existing_labor_run_results,
        existing_cost_effectiveness_run_results,
        existing_cost_components_run_results,
    ) = load_existing_run_results(county)
    active_scenarios = {scenario["scenario"] for scenario in SCENARIOS}

    def filter_existing(frame, keep_scenarios=None):
        if frame is None:
            return None
        allowed_scenarios = active_scenarios if keep_scenarios is None else keep_scenarios
        frame = frame.loc[frame["scenario"].isin(allowed_scenarios)].copy()
        return None if frame.empty else frame

    existing_run_results = filter_existing(existing_run_results)
    existing_location_run_results = filter_existing(existing_location_run_results)
    existing_daly_run_results = filter_existing(existing_daly_run_results)
    existing_labor_run_results = filter_existing(existing_labor_run_results)
    existing_cost_effectiveness_run_results = filter_existing(
        existing_cost_effectiveness_run_results
    )
    existing_cost_components_run_results = filter_existing(
        existing_cost_components_run_results
    )

    complete_scenarios = (
        set(existing_run_results["scenario"].unique())
        if existing_run_results is not None
        else set()
    )
    if county in DALY_COUNTIES:
        complete_scenarios &= (
            set(existing_daly_run_results["scenario"].unique())
            if existing_daly_run_results is not None
            else set()
        )
    if county in LABOR_COUNTIES:
        complete_scenarios &= (
            set(existing_labor_run_results["scenario"].unique())
            if existing_labor_run_results is not None
            else set()
        )
    if county in COST_EFFECTIVENESS_COUNTIES:
        active_cost_scenarios = {
            scenario["scenario"]
            for scenario in SCENARIOS
            if scenario["scenario"] not in {"diagnosis_base", "mentors_base"}
        }
        cost_complete = (
            set(existing_cost_effectiveness_run_results["scenario"].unique())
            if existing_cost_effectiveness_run_results is not None
            else set()
        )
        component_complete = (
            set(existing_cost_components_run_results["scenario"].unique())
            if existing_cost_components_run_results is not None
            else set()
        )
        cost_complete &= component_complete
        if active_cost_scenarios and active_cost_scenarios.issubset(cost_complete):
            cost_complete.add("diagnosis_base")
            cost_complete.add("mentors_base")
        complete_scenarios &= cost_complete
    existing_run_results = filter_existing(existing_run_results, complete_scenarios)
    existing_location_run_results = filter_existing(
        existing_location_run_results,
        complete_scenarios,
    )
    existing_daly_run_results = filter_existing(existing_daly_run_results, complete_scenarios)
    existing_labor_run_results = filter_existing(existing_labor_run_results, complete_scenarios)
    cost_keep_scenarios = complete_scenarios - {"diagnosis_base", "mentors_base"}
    existing_cost_effectiveness_run_results = filter_existing(
        existing_cost_effectiveness_run_results,
        cost_keep_scenarios,
    )
    existing_cost_components_run_results = filter_existing(
        existing_cost_components_run_results,
        cost_keep_scenarios,
    )
    already_done = complete_scenarios
    scenarios_to_run = [s for s in SCENARIOS if s["scenario"] not in already_done]
    skipped_scenarios = [s["scenario"] for s in SCENARIOS if s["scenario"] in already_done]
    if skipped_scenarios:
        print(
            f"{county.title()}: skipping already-generated scenarios: {', '.join(skipped_scenarios)}",
            flush=True,
        )
    if not scenarios_to_run:
        print(f"{county.title()}: all scenarios already generated, nothing to run", flush=True)

    rows = []
    location_rows = []
    daly_rows = []
    labor_rows = []
    monthly_rows_by_scenario = {}

    run_indices = range(FIRST_ANALYSIS_RUN - 1, N_RUNS) if scenarios_to_run else []

    for completed_runs, run in enumerate(run_indices, start=1):
        parameter_seed = BASE_SEED + run
        month_seeds = np.arange(parameter_seed, parameter_seed + N_MONTHS, dtype=int)
        overall_baseline_summary = None
        overall_baseline_location_summary = None
        overall_baseline_monthly = None
        for scenario_def in scenarios_to_run:
            scenario = scenario_def["scenario"]
            is_standard_base = (
                scenario_def["level"] == "base"
                and scenario_def["intervention"] != "FQA"
            )
            if is_standard_base and overall_baseline_summary is not None:
                scenario_summary = overall_baseline_summary.copy()
                location_summary = [
                    row.copy() for row in overall_baseline_location_summary
                ]
                monthly = overall_baseline_monthly.copy()
            else:
                param = get_parameters(county=county, seed=parameter_seed)
                param = calculate_derived_parameters(param)
                param.update({
                    "E": reset_E(),
                    "S": reset_S(slider_params),
                    "HSS": reset_HSS(slider_params),
                })
                flags = reset_flags()
                configure_scenario(scenario, param, flags, slider_params)
                monthly, individuals, _ = run_model_dash(
                    param,
                    flags,
                    n_months=N_MONTHS,
                    int_period=N_MONTHS,
                    base_seed=month_seeds,
                )
                scenario_summary = summarize(individuals)
                location_summary = summarize_by_location(individuals)
                if county in DALY_COUNTIES:
                    daly_rows.append({
                        "scenario": scenario,
                        "intervention": scenario_def["intervention"],
                        "level": scenario_def["level"],
                        "run": run + 1,
                        **summarize_daly(individuals),
                    })
                if county in LABOR_COUNTIES:
                    labor_rows.append({
                        "scenario": scenario,
                        "intervention": scenario_def["intervention"],
                        "level": scenario_def["level"],
                        "run": run + 1,
                        **summarize_labor(monthly),
                    })
                if is_standard_base:
                    overall_baseline_summary = scenario_summary.copy()
                    overall_baseline_location_summary = [
                        row.copy() for row in location_summary
                    ]
                    overall_baseline_monthly = monthly.copy()
            if county in COST_EFFECTIVENESS_COUNTIES:
                monthly_rows_by_scenario.setdefault(scenario, []).append(
                    monthly_with_run(monthly, run + 1)
                )
            rows.append({
                "scenario": scenario,
                "intervention": scenario_def["intervention"],
                "level": scenario_def["level"],
                "run": run + 1,
                **scenario_summary,
            })
            for location_result in location_summary:
                location_rows.append({
                    "scenario": scenario,
                    "intervention": scenario_def["intervention"],
                    "level": scenario_def["level"],
                    "run": run + 1,
                    **location_result,
                })
        if completed_runs % 10 == 0 or completed_runs == 1:
            print(
                f"{county.title()}: completed {completed_runs}/{ANALYSIS_RUNS} "
                f"matched runs using run {run + 1}/{N_RUNS}",
                flush=True,
            )
    new_run_results = pd.DataFrame(rows)
    new_location_run_results = pd.DataFrame(location_rows)
    run_results = _combine_frames(existing_run_results, new_run_results)
    location_run_results = _combine_frames(existing_location_run_results, new_location_run_results)
    metric_columns = [
        column for column in run_results.columns
        if column not in {"scenario", "intervention", "level", "run"}
    ]
    means = run_results.groupby(
        ["scenario", "intervention", "level"], sort=False
    )[metric_columns].mean().reset_index()

    differences = difference_from_baseline(
        means,
        metric_columns,
        baseline_candidates=("diagnosis_base", "mentors_base"),
    )
    mean_baseline_comparison = mean_vs_baseline(
        means,
        metric_columns,
        baseline_candidates=("diagnosis_base", "mentors_base"),
    )

    location_keys = [
        "scenario", "intervention", "level", "location", "location_code"
    ]
    location_metric_columns = [
        column for column in location_run_results.columns
        if column not in {*location_keys, "run"}
    ]
    location_means = location_run_results.groupby(
        location_keys, sort=False
    )[location_metric_columns].mean().reset_index()
    location_baseline_name = next(
        (
            name for name in ("diagnosis_base", "mentors_base")
            if (location_means["scenario"] == name).any()
        ),
        None,
    )
    location_differences = location_means.copy()
    if location_baseline_name is None:
        location_differences.insert(3, "comparison_baseline", "")
    else:
        baseline_by_location = location_means.loc[
            location_means["scenario"] == location_baseline_name,
            ["location_code", *location_metric_columns],
        ].set_index("location_code")
        for index, row in location_differences.iterrows():
            baseline = baseline_by_location.loc[row["location_code"]]
            location_differences.loc[index, location_metric_columns] = (
                row[location_metric_columns].to_numpy(dtype=float)
                - baseline.to_numpy(dtype=float)
            )
        location_differences.insert(3, "comparison_baseline", location_baseline_name)

    daly_run_results = _combine_frames(existing_daly_run_results, pd.DataFrame(daly_rows))
    labor_run_results = _combine_frames(existing_labor_run_results, pd.DataFrame(labor_rows))

    daly_metric_columns = [
        column for column in daly_run_results.columns
        if column not in {"scenario", "intervention", "level", "run"}
    ]
    daly_means = (
        daly_run_results.groupby(["scenario", "intervention", "level"], sort=False)[daly_metric_columns].mean().reset_index()
        if not daly_run_results.empty
        else pd.DataFrame()
    )
    daly_differences = (
        difference_from_baseline(daly_means, daly_metric_columns, ("diagnosis_base", "mentors_base"))
        if not daly_means.empty
        else pd.DataFrame()
    )
    daly_mean_baseline_comparison = (
        mean_vs_baseline(daly_means, daly_metric_columns, ("diagnosis_base", "mentors_base"))
        if not daly_means.empty
        else pd.DataFrame()
    )

    labor_metric_columns = [
        column for column in labor_run_results.columns
        if column not in {"scenario", "intervention", "level", "run"}
    ]
    labor_means = (
        labor_run_results.groupby(["scenario", "intervention", "level"], sort=False)[labor_metric_columns].mean().reset_index()
        if not labor_run_results.empty
        else pd.DataFrame()
    )
    labor_differences = (
        difference_from_baseline(labor_means, labor_metric_columns, ("diagnosis_base", "mentors_base"))
        if not labor_means.empty
        else pd.DataFrame()
    )
    labor_mean_baseline_comparison = (
        mean_vs_baseline(labor_means, labor_metric_columns, ("diagnosis_base", "mentors_base"))
        if not labor_means.empty
        else pd.DataFrame()
    )

    if county in COST_EFFECTIVENESS_COUNTIES and monthly_rows_by_scenario:
        monthly_by_scenario = {
            scenario: pd.concat(frames, ignore_index=True)
            for scenario, frames in monthly_rows_by_scenario.items()
        }
        (
            new_cost_effectiveness_run_results,
            new_cost_components_run_results,
            _,
        ) = build_cost_effectiveness_outputs(
            monthly_by_scenario,
            SCENARIOS,
            num_pocus=num_pocus,
        )
    else:
        new_cost_effectiveness_run_results = pd.DataFrame()
        new_cost_components_run_results = pd.DataFrame()

    cost_effectiveness_run_results = _combine_frames(
        existing_cost_effectiveness_run_results,
        new_cost_effectiveness_run_results,
    )
    cost_components_run_results = _combine_frames(
        existing_cost_components_run_results,
        new_cost_components_run_results,
    )
    cost_effectiveness_mean_baseline_comparison = summarize_cost_effectiveness(
        cost_effectiveness_run_results
    )

    return (
        pd.DataFrame(SCENARIOS),
        run_results,
        means,
        differences,
        location_run_results,
        location_means,
        location_differences,
        mean_baseline_comparison,
        daly_run_results,
        daly_means,
        daly_differences,
        daly_mean_baseline_comparison,
        labor_run_results,
        labor_means,
        labor_differences,
        labor_mean_baseline_comparison,
        cost_effectiveness_run_results,
        cost_components_run_results,
        cost_effectiveness_mean_baseline_comparison,
    )


def build_ce_monthly_outputs(county):
    parameter_loader.WORKBOOK_PATH = WORKBOOK_PATH
    slider_params = get_slider_params(county=county)
    monthly_rows_by_scenario = {}

    for completed_runs, run in enumerate(range(FIRST_ANALYSIS_RUN - 1, N_RUNS), start=1):
        parameter_seed = BASE_SEED + run
        month_seeds = np.arange(
            parameter_seed,
            parameter_seed + COST_EFFECTIVENESS_N_MONTHS,
            dtype=int,
        )
        for scenario_def in SCENARIOS:
            scenario = scenario_def["scenario"]
            param = get_parameters(county=county, seed=parameter_seed)
            param = calculate_derived_parameters(param)
            param.update({
                "E": reset_E(),
                "S": reset_S(slider_params),
                "HSS": reset_HSS(slider_params),
            })
            flags = reset_flags()
            configure_scenario(scenario, param, flags, slider_params)
            monthly, _, _ = run_model_dash(
                param,
                flags,
                n_months=COST_EFFECTIVENESS_N_MONTHS,
                int_period=COST_EFFECTIVENESS_N_MONTHS,
                base_seed=month_seeds,
            )
            monthly_rows_by_scenario.setdefault(scenario, []).append(
                monthly_with_run(monthly, run + 1)
            )
        if completed_runs % 10 == 0 or completed_runs == 1:
            print(
                f"{county.title()} CE: completed {completed_runs}/{ANALYSIS_RUNS} "
                f"matched runs using run {run + 1}/{N_RUNS}",
                flush=True,
            )

    return {
        scenario: pd.concat(frames, ignore_index=True)
        for scenario, frames in monthly_rows_by_scenario.items()
    }


def infer_num_pocus(county):
    parameter_loader.WORKBOOK_PATH = WORKBOOK_PATH
    county_param_sample = get_parameters(county=county, seed=BASE_SEED)
    return int(
        county_param_sample["num_L2/3"]
        + county_param_sample["num_L4"]
        + county_param_sample["num_L5"]
    )


def _resource_array(value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.array([float(arr)])
    return arr.ravel()


def _resource_sum(value):
    return float(np.nansum(_resource_array(value)))


def _resource_level(value, index):
    arr = _resource_array(value)
    return float(arr[index]) if index < arr.size else 0.0


def _cost_unit_costs_usd(cost_parameters):
    exchange_rate = cost_parameters["USD_to_Ksh"]
    return {
        key: value / exchange_rate
        for key, value in cost_parameters["cost_dict"].items()
    }


def _discount_costs(df, discount_rate):
    out = df.copy()
    out["cost_discounted_yearly"] = (
        out["cost_yearly"] / ((1 + discount_rate) ** (out["year"] - 1))
    )
    return out


def _annuity_factor(discount_rate, useful_life):
    if discount_rate == 0:
        return 1 / useful_life
    return (
        discount_rate * (1 + discount_rate) ** useful_life
    ) / ((1 + discount_rate) ** useful_life - 1)


def _annualized_stock_cost(run_stock, unit_cost, useful_life, discount_rate, max_year, cost_type):
    factor = _annuity_factor(discount_rate, useful_life)
    rows = []
    for _, stock in run_stock.iterrows():
        annual_cost = float(stock["stock"] * unit_cost * factor)
        for year in range(1, max_year + 1):
            rows.append({
                "Run": int(stock["Run"]),
                "year": year,
                "cost_yearly": annual_cost,
                "resource_count": float(stock["stock"]),
                "cost_type": cost_type,
            })
    if not rows:
        return pd.DataFrame(columns=["Run", "year", "cost_yearly", "resource_count", "cost_type", "cost_discounted_yearly"])
    return _discount_costs(pd.DataFrame(rows), discount_rate)


def baseline_resource_use_monthly(monthly):
    monthly = monthly_with_year(monthly)
    out = pd.DataFrame({
        "Run": monthly["Run"].astype(int),
        "Month": monthly["Month"].astype(int),
        "year": monthly["year"].astype(int),
        "ANC": monthly["ANC"].apply(_resource_sum),
        "Fac non-CS": monthly["Fac non-CS"].apply(_resource_sum),
        "CS": monthly["CS"].apply(_resource_sum),
        "Free_referrals": monthly["Free_referrals"].apply(_resource_sum),
        "Emergency transfers": monthly["Emergency transfers"].apply(_resource_sum),
        "Live Births Final": monthly["Live Births Final"].apply(_resource_sum),
        "M_DALYs": monthly["M_DALYs"].apply(_resource_sum),
        "N_DALYs": monthly["N_DALYs"].apply(_resource_sum),
        "DALYs": monthly["DALYs"].apply(_resource_sum),
        "Facility_capacity_actual": monthly["Facility_capacity_actual"].astype(float),
    })
    staff_columns = {
        "Surgical_actual": ["L4", "L5"],
        "Nurse_actual": ["L4", "L5"],
        "Anesthetist_actual": ["L4", "L5"],
    }
    for column, levels in staff_columns.items():
        for index, level in enumerate(levels):
            out[f"{column}_{level}"] = monthly[column].apply(lambda value, i=index: _resource_level(value, i))
    equipment_columns = {
        "Doppler_Actual": ["L23", "L4", "L5"],
        "CTG_Actual": ["L23", "L4", "L5"],
    }
    for column, levels in equipment_columns.items():
        for index, level in enumerate(levels):
            out[f"{column}_{level}"] = monthly[column].apply(lambda value, i=index: _resource_level(value, i))
    return out


def calculate_absolute_baseline_costs(resource_monthly):
    cost_parameters = get_cost_parameters()
    unit_costs = _cost_unit_costs_usd(cost_parameters)
    discount_rate = cost_parameters["cost_discount_rate"]
    max_year = int(cost_parameters["time_horizon_years"])
    dispatches_per_vehicle = cost_parameters["dispatches_per_vehicle"]
    components = []

    service_specs = [
        ("ANC", "ANC", "ANC"),
        ("Fac non-CS", "Fac_Delivery", "Facility Delivery"),
        ("CS", "CS_Delivery", "CS Delivery"),
    ]
    for column, unit_key, cost_type in service_specs:
        yearly = (
            resource_monthly.assign(cost_monthly=resource_monthly[column] * unit_costs[unit_key])
            .groupby(["Run", "year"], as_index=False)
            .agg(cost_yearly=("cost_monthly", "sum"), resource_count=(column, "sum"))
        )
        yearly["cost_type"] = cost_type
        components.append(_discount_costs(yearly, discount_rate))

    referral_specs = [
        ("Free_referrals", "Taxi_Dispatch", "Taxi_Monthly", "n_taxi_used", "Referral Recurrent", "Referral (Capital)", "Taxi_Setup"),
        ("Emergency transfers", "Ambulance_Dispatch", "Ambulance_Monthly", "n_ambulance_used", "Transfer Recurrent", "Transfer (Capital)", "Ambulance_Setup"),
    ]
    vehicle_stock_frames = []
    for column, dispatch_key, monthly_key, vehicle_col, recurrent_type, capital_type, setup_key in referral_specs:
        monthly = resource_monthly[["Run", "Month", "year", column]].copy()
        monthly["n_dispatches"] = monthly[column]
        monthly[vehicle_col] = np.ceil(monthly["n_dispatches"] / dispatches_per_vehicle)
        monthly["cost_monthly"] = (
            monthly["n_dispatches"] * unit_costs[dispatch_key]
            + monthly[vehicle_col] * unit_costs[monthly_key]
        )
        yearly = (
            monthly.groupby(["Run", "year"], as_index=False)
            .agg(
                cost_yearly=("cost_monthly", "sum"),
                resource_count=("n_dispatches", "sum"),
                vehicle_stock=(vehicle_col, "max"),
            )
        )
        yearly["cost_type"] = recurrent_type
        components.append(_discount_costs(yearly, discount_rate))
        vehicle_stock = (
            monthly.groupby("Run", as_index=False)
            .agg(stock=(vehicle_col, "max"))
        )
        vehicle_stock_frames.append((vehicle_stock, setup_key, capital_type))
    for vehicle_stock, setup_key, capital_type in vehicle_stock_frames:
        components.append(_annualized_stock_cost(
            vehicle_stock,
            unit_costs[setup_key],
            cost_parameters["useful_life_dict"][setup_key],
            discount_rate,
            max_year,
            capital_type,
        ))

    labor_specs = [
        (["Surgical_actual_L4", "Surgical_actual_L5"], "surgical_staff", "Labor (Surgical)"),
        (["Nurse_actual_L4", "Nurse_actual_L5"], "nurse_staff", "Labor (Nurse)"),
        (["Anesthetist_actual_L4", "Anesthetist_actual_L5"], "anesthetist", "Labor (Anesthetist)"),
    ]
    for columns, unit_key, cost_type in labor_specs:
        monthly = resource_monthly[["Run", "year", *columns]].copy()
        monthly["staff_count"] = monthly[columns].sum(axis=1)
        monthly["cost_monthly"] = monthly["staff_count"] * unit_costs[unit_key]
        yearly = (
            monthly.groupby(["Run", "year"], as_index=False)
            .agg(cost_yearly=("cost_monthly", "sum"), resource_count=("staff_count", "sum"))
        )
        yearly["cost_type"] = cost_type
        components.append(_discount_costs(yearly, discount_rate))

    beds_per_month = 83 / 12
    infrastructure_stock = (
        resource_monthly.groupby("Run", as_index=False)
        .agg(stock=("Facility_capacity_actual", lambda values: math.ceil(float(np.nanmax(values)) / beds_per_month)))
    )
    components.append(_annualized_stock_cost(
        infrastructure_stock,
        unit_costs["Infra"],
        cost_parameters["useful_life_dict"]["Infra"],
        discount_rate,
        max_year,
        "Infrastructure",
    ))

    equipment_specs = [
        (["Doppler_Actual_L23", "Doppler_Actual_L4", "Doppler_Actual_L5"], "Doppler", "Equipment (Doppler)"),
        (["CTG_Actual_L23", "CTG_Actual_L4", "CTG_Actual_L5"], "CTG", "Equipment (CTG)"),
    ]
    for columns, unit_key, cost_type in equipment_specs:
        stock = resource_monthly[["Run", *columns]].copy()
        stock["stock"] = stock[columns].sum(axis=1)
        run_stock = stock.groupby("Run", as_index=False).agg(stock=("stock", "max"))
        components.append(_annualized_stock_cost(
            run_stock,
            unit_costs[unit_key],
            cost_parameters["useful_life_dict"][unit_key],
            discount_rate,
            max_year,
            cost_type,
        ))

    component_results = pd.concat(components, ignore_index=True)
    component_results["scenario"] = "diagnosis_base"
    component_results["intervention"] = "Baseline"
    component_results["level"] = "base"
    component_results["cost_category"] = component_results["cost_type"].map(COST_CATEGORY_BY_TYPE)
    if component_results["cost_category"].isna().any():
        missing = sorted(component_results.loc[component_results["cost_category"].isna(), "cost_type"].unique())
        raise ValueError(f"Unmapped baseline cost types: {missing}")
    return component_results


def summarize_absolute_baseline_costs(resource_monthly, component_results):
    category_costs = (
        component_results.groupby(["scenario", "intervention", "level", "Run", "cost_category"], as_index=False)
        .agg(cost_yearly=("cost_yearly", "sum"))
        .pivot_table(
            index=["scenario", "intervention", "level", "Run"],
            columns="cost_category",
            values="cost_yearly",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )
    category_costs.columns.name = None
    for column in MAJOR_COST_COLUMNS:
        if column not in category_costs.columns:
            category_costs[column] = 0.0
    category_costs["total_cost_usd"] = category_costs[MAJOR_COST_COLUMNS].sum(axis=1)
    discounted = (
        component_results.groupby(["scenario", "intervention", "level", "Run"], as_index=False)
        .agg(total_discounted_cost_usd=("cost_discounted_yearly", "sum"))
    )
    dalys = (
        resource_monthly.groupby("Run", as_index=False)
        .agg(
            maternal_dalys=("M_DALYs", "sum"),
            neonatal_dalys=("N_DALYs", "sum"),
            total_dalys=("DALYs", "sum"),
        )
    )
    run_results = (
        category_costs.merge(discounted, on=["scenario", "intervention", "level", "Run"], how="left")
        .merge(dalys, on="Run", how="left")
    )
    run_results["final_year"] = int(get_cost_parameters()["time_horizon_years"])
    run_results["cost_category_sum_check_usd"] = run_results[MAJOR_COST_COLUMNS].sum(axis=1) - run_results["total_cost_usd"]
    run_results["total_daly_sum_check"] = run_results["maternal_dalys"] + run_results["neonatal_dalys"] - run_results["total_dalys"]
    summary_columns = [
        "total_cost_usd",
        *MAJOR_COST_COLUMNS,
        "total_discounted_cost_usd",
        "maternal_dalys",
        "neonatal_dalys",
        "total_dalys",
        "cost_category_sum_check_usd",
        "total_daly_sum_check",
        "final_year",
    ]
    summary = run_results.groupby(["scenario", "intervention", "level"], as_index=False)[summary_columns].mean()
    return run_results, summary


def build_baseline_monthly_outputs(county):
    parameter_loader.WORKBOOK_PATH = WORKBOOK_PATH
    slider_params = get_slider_params(county=county)
    monthly_rows = []
    for completed_runs, run in enumerate(range(FIRST_ANALYSIS_RUN - 1, N_RUNS), start=1):
        parameter_seed = BASE_SEED + run
        month_seeds = np.arange(
            parameter_seed,
            parameter_seed + COST_EFFECTIVENESS_N_MONTHS,
            dtype=int,
        )
        param = get_parameters(county=county, seed=parameter_seed)
        param = calculate_derived_parameters(param)
        param.update({
            "E": reset_E(),
            "S": reset_S(slider_params),
            "HSS": reset_HSS(slider_params),
        })
        flags = reset_flags()
        configure_scenario("diagnosis_base", param, flags, slider_params)
        monthly, _, _ = run_model_dash(
            param,
            flags,
            n_months=COST_EFFECTIVENESS_N_MONTHS,
            int_period=COST_EFFECTIVENESS_N_MONTHS,
            base_seed=month_seeds,
        )
        monthly_rows.append(monthly_with_run(monthly, run + 1))
        if completed_runs % 10 == 0 or completed_runs == 1:
            print(
                f"{county.title()} baseline cost: completed {completed_runs}/{ANALYSIS_RUNS} "
                f"matched runs using run {run + 1}/{N_RUNS}",
                flush=True,
            )
    return pd.concat(monthly_rows, ignore_index=True)


def write_absolute_baseline_cost_outputs(counties=None):
    counties = list(COST_DALY_COUNTIES if counties is None else counties)
    outputs = {}
    for county in counties:
        monthly = build_baseline_monthly_outputs(county)
        resource_monthly = baseline_resource_use_monthly(monthly)
        component_results = calculate_absolute_baseline_costs(resource_monthly)
        run_results, summary = summarize_absolute_baseline_costs(resource_monthly, component_results)
        results_dir = BASELINE_COST_OUTPUT_ROOT / f"scenario_comparison_results_{county}"
        results_dir.mkdir(parents=True, exist_ok=True)
        resource_monthly.to_csv(results_dir / "scenario_baseline_resource_use_monthly.csv", index=False)
        component_results.to_csv(results_dir / "scenario_baseline_cost_component_run_results.csv", index=False)
        run_results.to_csv(results_dir / "scenario_baseline_cost_run_results.csv", index=False)
        summary.to_csv(results_dir / "scenario_baseline_cost_mean.csv", index=False)
        print(f"{county.title()} absolute baseline cost outputs written to {results_dir.resolve()}", flush=True)
        outputs[county] = (resource_monthly, component_results, run_results, summary)
    return outputs


def write_cost_daly_method_file(results_dir):
    cost_daly_calculation_methods().to_csv(
        results_dir / "scenario_cost_daly_calculation_methods.csv",
        index=False,
    )


def find_existing_cost_daly_results_dir(county):
    candidates = [
        Path(f"scenario_comparison_results_{county}"),
        Path("untitled folder") / f"scenario_comparison_results_{county}",
    ]
    for results_dir in candidates:
        if (results_dir / "scenario_cost_daly_run_results.csv").exists():
            return results_dir
    return candidates[0]


def write_cost_daly_summary_from_existing(counties=None):
    counties = list(COST_DALY_COUNTIES if counties is None else counties)
    summaries = {}
    for county in counties:
        results_dir = find_existing_cost_daly_results_dir(county)
        run_results_path = results_dir / "scenario_cost_daly_run_results.csv"
        if not run_results_path.exists():
            raise FileNotFoundError(
                f"Missing {run_results_path}; run --cost-daly-only first."
            )
        run_results = pd.read_csv(run_results_path)
        summary = summarize_combined_cost_daly_run_results(run_results)
        results_dir.mkdir(exist_ok=True)
        summary.to_csv(results_dir / "scenario_cost_daly_mean_vs_baseline.csv", index=False)
        write_cost_daly_method_file(results_dir)
        print(
            f"{county.title()} cost/DALY summary recalculated from existing run-level results",
            flush=True,
        )
        summaries[county] = summary
    return summaries


def write_cost_daly_outputs(counties=None):
    counties = list(COST_DALY_COUNTIES if counties is None else counties)
    outputs = {}
    for county in counties:
        monthly_by_scenario = build_ce_monthly_outputs(county)
        num_pocus = infer_num_pocus(county)
        run_results, component_results, summary = build_combined_cost_daly_outputs(
            monthly_by_scenario,
            SCENARIOS,
            num_pocus=num_pocus,
        )

        results_dir = Path(f"scenario_comparison_results_{county}")
        results_dir.mkdir(exist_ok=True)
        run_results.to_csv(results_dir / "scenario_cost_daly_run_results.csv", index=False)
        summary.to_csv(results_dir / "scenario_cost_daly_mean_vs_baseline.csv", index=False)
        component_results.to_csv(
            results_dir / "scenario_cost_daly_component_run_results.csv",
            index=False,
        )
        write_cost_daly_method_file(results_dir)
        print(
            f"{county.title()} 10-year cost/DALY outputs written to {results_dir.resolve()}",
            flush=True,
        )
        outputs[county] = (run_results, component_results, summary)
    return outputs


def write_kakamega_cost_daly_outputs():
    return write_cost_daly_outputs(["kakamega"])["kakamega"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SDR scenario comparison exports.")
    parser.add_argument(
        "--cost-daly-only",
        action="store_true",
        help="Run only the 10-year combined cost/DALY export for COST_DALY_COUNTIES.",
    )
    parser.add_argument(
        "--cost-daly-summary-only",
        action="store_true",
        help="Recalculate cost/DALY summary and methods CSVs from existing run-level outputs.",
    )
    parser.add_argument(
        "--baseline-cost-only",
        action="store_true",
        help="Run only diagnosis_base for 120 months and export absolute baseline cost outputs.",
    )
    args = parser.parse_args()
    if args.cost_daly_only:
        write_cost_daly_outputs()
        raise SystemExit
    if args.cost_daly_summary_only:
        write_cost_daly_summary_from_existing()
        raise SystemExit
    if args.baseline_cost_only:
        write_absolute_baseline_cost_outputs()
        raise SystemExit

    process_columns = [
        "anc", "initial_l45", "final_l45", "free_referral",
        "emergency_transfer", "delay_lt1h_given_transfer",
        "delay_1_2h_given_transfer", "delay_2plus_given_transfer",
        "iv_iron_given_anemia_anc", "pph_bundle_given_pph",
        "mgso4_given_eclampsia", "antibiotics_given_sepsis",
        "oxytocin_given_ol",
    ]
    outcome_columns = [
        "pph", "maternal_sepsis", "eclampsia", "obstructed_labor",
        "severe_complication", "maternal_deaths_per_100k",
        "neonatal_deaths_per_1000",
    ]
    for county in COUNTIES:
        (
            definitions,
            run_results,
            means,
            differences,
            location_run_results,
            location_means,
            location_differences,
            mean_baseline_comparison,
            daly_run_results,
            daly_means,
            daly_differences,
            daly_mean_baseline_comparison,
            labor_run_results,
            labor_means,
            labor_differences,
            labor_mean_baseline_comparison,
            cost_effectiveness_run_results,
            cost_components_run_results,
            cost_effectiveness_mean_baseline_comparison,
        ) = run_scenarios(county)
        process = means[["scenario", "intervention", "level", *process_columns]].copy()
        process[process_columns] *= 100
        outcomes = means[["scenario", "intervention", "level", *outcome_columns]].copy()
        outcomes[[c for c in outcome_columns if "deaths_per" not in c]] *= 100
        outcome_differences = differences[
            ["scenario", "intervention", "level", "comparison_baseline", *outcome_columns]
        ].copy()
        outcome_differences[[c for c in outcome_columns if "deaths_per" not in c]] *= 100

        results_json = Path(f"scenario_comparison_results_{county}.json")
        results_dir = Path(f"scenario_comparison_results_{county}")
        payload = {
            "metadata": {
                "county": county,
                "n_months": N_MONTHS,
                "n_runs": N_RUNS,
                "first_analysis_run": FIRST_ANALYSIS_RUN,
                "analysis_runs": ANALYSIS_RUNS,
                "base_seed": BASE_SEED,
            },
            "definitions": definitions.to_dict(orient="records"),
            "run_results": json.loads(run_results.to_json(orient="records")),
            "means": json.loads(means.to_json(orient="records")),
            "differences": json.loads(differences.to_json(orient="records")),
            "process": json.loads(process.to_json(orient="records")),
            "outcomes": json.loads(outcomes.to_json(orient="records")),
            "outcome_differences": json.loads(outcome_differences.to_json(orient="records")),
            "location_run_results": json.loads(
                location_run_results.to_json(orient="records")
            ),
            "location_means": json.loads(location_means.to_json(orient="records")),
            "location_differences": json.loads(
                location_differences.to_json(orient="records")
            ),
            "mean_baseline_comparison": json.loads(mean_baseline_comparison.to_json(orient="records")),
            "daly_run_results": json.loads(daly_run_results.to_json(orient="records")),
            "daly_means": json.loads(daly_means.to_json(orient="records")),
            "daly_differences": json.loads(daly_differences.to_json(orient="records")),
            "daly_mean_baseline_comparison": json.loads(daly_mean_baseline_comparison.to_json(orient="records")),
            "labor_run_results": json.loads(labor_run_results.to_json(orient="records")),
            "labor_means": json.loads(labor_means.to_json(orient="records")),
            "labor_differences": json.loads(labor_differences.to_json(orient="records")),
            "labor_mean_baseline_comparison": json.loads(labor_mean_baseline_comparison.to_json(orient="records")),
            "cost_effectiveness_run_results": json.loads(cost_effectiveness_run_results.to_json(orient="records")),
            "cost_components_run_results": json.loads(cost_components_run_results.to_json(orient="records")),
            "cost_effectiveness_mean_baseline_comparison": json.loads(cost_effectiveness_mean_baseline_comparison.to_json(orient="records")),
        }
        results_json.write_text(json.dumps(payload, indent=2))
        results_dir.mkdir(exist_ok=True)
        output_tables = {
            "scenario_definitions.csv": definitions,
            "scenario_run_results.csv": run_results,
            "scenario_mean_results.csv": means,
            "scenario_mean_vs_baseline.csv": mean_baseline_comparison,
            "scenario_daly_run_results.csv": daly_run_results,
            "scenario_daly_mean_vs_baseline.csv": daly_mean_baseline_comparison,
        }
        if county in LABOR_COUNTIES:
            output_tables.update({
                "scenario_labor_run_results.csv": labor_run_results,
                "scenario_labor_mean_vs_baseline.csv": labor_mean_baseline_comparison,
            })
        if county in COST_EFFECTIVENESS_COUNTIES:
            output_tables.update({
                "scenario_cost_effectiveness_run_results.csv": cost_effectiveness_run_results,
                "scenario_cost_components_run_results.csv": cost_components_run_results,
                "scenario_cost_effectiveness_mean_vs_baseline.csv": cost_effectiveness_mean_baseline_comparison,
            })
        for filename, table in output_tables.items():
            table.to_csv(results_dir / filename, index=False)
        print(f"{county.title()} results written to {results_dir.resolve()}", flush=True)
