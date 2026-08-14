import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import math
import scipy.stats as stats
import time
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from parameter_loader import (
    get_parameters,
    calculate_derived_parameters,
    get_available_counties,
    get_slider_params,
    get_fqa_pulse_modifier_options,
    DEFAULT_COUNTY,
)
from model_run import run_model_dash
from global_func import reset_flags, reset_E, reset_HSS, reset_S, get_P_l45
from debug_report import build_parameter_debug_report

# Set to False to disable the parameter debug report feature entirely.
ENABLE_PARAM_DEBUG_REPORT = True
DASHBOARD_RANDOM_MASTER_SEED = 2025
DASHBOARD_RANDOM_RUN_OFFSET = 20

st.set_page_config(layout="wide")
selected_plot = None

FQA_PULSE_MODIFIER_OPTIONS = get_fqa_pulse_modifier_options()

county_options = get_available_counties()
if DEFAULT_COUNTY not in county_options:
    county_options = [DEFAULT_COUNTY] + county_options
selected_county = st.selectbox(
    "County",
    county_options,
    index=county_options.index(DEFAULT_COUNTY),
)

MODEL = {
    "imple_time": 3,
    "main_time": 0,
    "int_period": 0,
    "n_months": 36,
    "multiple_run": False,
    "n_runs": 1,
}

#initalize intervention parameters
#b_param = get_parameters(seed = 1)
slider_params = get_slider_params()
i_flags, i_E, i_HSS, i_S = reset_flags(), reset_E(), reset_HSS(slider_params), reset_S(slider_params)


def make_shifted_run_seeds(n_runs, n_months):
    master_rng = np.random.default_rng(DASHBOARD_RANDOM_MASTER_SEED)
    run_seeds = master_rng.integers(
        low=0,
        high=1e9,
        size=(n_runs + DASHBOARD_RANDOM_RUN_OFFSET, n_months),
    )
    return run_seeds[DASHBOARD_RANDOM_RUN_OFFSET:]


def go_back_to_main():
    st.session_state.intervention_selection = None
    st.session_state.hss_mode = None
    st.session_state.scenario_selected = None
    st.session_state.model_finished = False

def go_back_to_hss():
    st.session_state.hss_mode = None
    st.session_state.scenario_selected = None
    st.session_state.model_finished = False


def clear_model_runtime_state(clear_comparison=True, clear_table=False):
    """Clear cached model outputs without changing currently selected interventions."""
    runtime_keys = [
        "b_df",
        "i_df",
        "b_df_multiple",
        "b_ind_outcomes",
        "i_ind_outcomes",
        "b_param",
        "i_param",
        "n_months",
        "int_period",
        "n_runs",
        "baseline_config",
        "param_debug_report",
        "scenario_name",
    ]
    comparison_keys = [
        "dual_first_config",
        "ab_base_df",
        "ab_base_ind_outcomes",
    ]

    for key in runtime_keys:
        st.session_state.pop(key, None)
    for key in ("b_df", "i_df", "n_months", "int_period", "i_param", "n_runs"):
        st.session_state[key] = None

    if clear_comparison:
        for key in comparison_keys:
            st.session_state.pop(key, None)
        st.session_state.ab_base_df = None
        st.session_state.ab_base_ind_outcomes = None

    if clear_table:
        st.session_state.dl_table = None

    st.session_state.model_finished = False
    st.session_state.selected_outcomes = []


HSS_DEMAND_PRESETS = {
    "Conservative": {"P_ANC": 70, "P_L45": 53},
    "Moderate": {"P_ANC": 80, "P_L45": 68},
    "Aggressive": {"P_ANC": 90, "P_L45": 90},
}
HSS_CAPACITY_MATCH = {
    "Conservative": 25.0,
    "Moderate": 50.0,
    "Aggressive": 85.0,
}
HSS_CAPACITY_MISMATCH = {
    "Conservative": 12.5,
    "Moderate": 25.0,
    "Aggressive": 42.5,
}
# Fixed implementation-index values for the non-"Current" fidelity levels.
# These replace the county's Excel-read index outright; they are not multipliers.
MENTORS_FIDELITY_OVERRIDE = {"Moderate": 0.5, "High": 0.95}
PROMPTS_IMPLEMENTATION_OVERRIDE = {"Moderate": 0.5, "High": 0.95}
# RR on 4+ ANC applied to the share of mothers reached by PROMPTS (see
# get_prompts_implementation_index / render_prompts). Fixed at all three
# levels, including "Current" -- only the implementation index (population
# coverage) varies with the workbook there.
PROMPTS_RR_OVERRIDE = {"Current": 1.02, "Moderate": 1.18, "High": 1.35}
BLOOD_TRACKING_FIDELITY_OVERRIDE = {"Current": 0.25, "Moderate": 0.5, "High": 0.95}
PULSE_FIDELITY_OVERRIDE = {"Moderate": 0.5, "High": 0.95}
FQA_FIDELITY_OVERRIDE = {"Moderate": 0.5, "High": 0.95}
# Coverage for the transfer-delay shift (assign_transfer_delay_categories /
# apply_transfer_delay_intervention in mortality.py). "Current" reads
# referral_implementation_index for the selected county.
TRANSFER_FIDELITY_OVERRIDE = {"Moderate": 0.50, "High": 0.95}
FQA_PULSE_MODIFIER_LEVELS = list(FQA_PULSE_MODIFIER_OPTIONS.keys())
# Maximum implementation-index boost a fully-implemented PULSE system can lend
# to the other MOMISH interventions (see sync_param_momish_from_hss).
# Names mirror scenario_comparison.py combined_current/moderate/high.
PULSE_IMPLEMENTATION_BOOST_OPTIONS = {"Current": 0.25, "Moderate": 0.5, "High": 0.75}
PULSE_IMPLEMENTATION_BOOST_LEVELS = list(PULSE_IMPLEMENTATION_BOOST_OPTIONS.keys())


def fqa_pulse_modifier_default_index():
    level = st.session_state.get("fqa_pulse_modifier_level", "Medium")
    if level not in FQA_PULSE_MODIFIER_LEVELS:
        level = "Medium"
    return FQA_PULSE_MODIFIER_LEVELS.index(level)


def pulse_implementation_boost_default_index():
    legacy_level_names = {"Low": "Current", "Medium": "Moderate"}
    saved_level = st.session_state.get("pulse_implementation_boost_level", "Moderate")
    level = legacy_level_names.get(saved_level, saved_level)
    if level not in PULSE_IMPLEMENTATION_BOOST_LEVELS:
        level = "Moderate"
    return PULSE_IMPLEMENTATION_BOOST_LEVELS.index(level)


def get_mentor_implementation_index(fidelity_choice):
    if fidelity_choice in MENTORS_FIDELITY_OVERRIDE:
        return MENTORS_FIDELITY_OVERRIDE[fidelity_choice]
    try:
        param_sample = get_parameters(rng=np.random.default_rng(0), county=selected_county)
        param_sample = calculate_derived_parameters(param_sample)
        return float(np.clip(param_sample.get("mentors_implementation_index", 0.0), 0.0, 1.0))
    except Exception:
        return 0.0

def get_prompts_implementation_index(fidelity_choice):
    if fidelity_choice in PROMPTS_IMPLEMENTATION_OVERRIDE:
        return PROMPTS_IMPLEMENTATION_OVERRIDE[fidelity_choice]
    try:
        param_sample = get_parameters(rng=np.random.default_rng(0), county=selected_county)
        param_sample = calculate_derived_parameters(param_sample)
        return float(np.clip(param_sample.get("prompts_implementation_index", 0.0), 0.0, 1.0))
    except Exception:
        return 0.0


def get_blood_tracking_implementation_index(fidelity_choice):
    return BLOOD_TRACKING_FIDELITY_OVERRIDE.get(fidelity_choice, 0.0)


def get_transfer_implementation_index(fidelity_choice):
    if fidelity_choice in TRANSFER_FIDELITY_OVERRIDE:
        return TRANSFER_FIDELITY_OVERRIDE[fidelity_choice]
    try:
        param_sample = get_parameters(rng=np.random.default_rng(0), county=selected_county)
        param_sample = calculate_derived_parameters(param_sample)
        return float(np.clip(param_sample.get("referral_implementation_index", 0.0), 0.0, 1.0))
    except Exception:
        return 0.0


def get_fqa_implementation_index(fidelity_choice):
    if fidelity_choice in FQA_FIDELITY_OVERRIDE:
        return FQA_FIDELITY_OVERRIDE[fidelity_choice]
    try:
        param_sample = get_parameters(rng=np.random.default_rng(0), county=selected_county)
        param_sample = calculate_derived_parameters(param_sample)
        return float(np.clip(param_sample.get("fqa_implementation_index", 0.0), 0.0, 1.0))
    except Exception:
        return 0.0

def get_pulse_implementation_index(fidelity_choice):
    if fidelity_choice in PULSE_FIDELITY_OVERRIDE:
        return PULSE_FIDELITY_OVERRIDE[fidelity_choice]
    try:
        param_sample = get_parameters(rng=np.random.default_rng(0), county=selected_county)
        param_sample = calculate_derived_parameters(param_sample)
        return float(np.clip(param_sample.get("pulse_implementation_index", 0.0), 0.0, 1.0))
    except Exception:
        return 0.0


def render_hss_preset():
    """All (Preset): component toggles + intensity; presets apply only to enabled blocks."""
    st.subheader(":chart_with_upwards_trend: HSS interventions (Preset)")
    st.caption(
        "Choose which components are on, then set intensity. "
        "Conservative / Moderate / Aggressive values apply only to enabled components."
    )

    col_d, col_s = st.columns(2)
    with col_d:
        st.markdown("**Demand**")
        enable_chv = st.toggle(
            "Employ CHVs",
            value=False,
            key="preset_enable_chv",
            help="CHVs refer to Community Healthcare Workers",
        )
        intensity = st.radio(
            "Intervention intensity",
            list(HSS_DEMAND_PRESETS.keys()),
            horizontal=True,
            key="preset_demand_intensity",
        )

    with col_s:
        st.markdown("**Supply**")
        enable_facility = st.toggle(
            "Upgrade L4/5 facilities",
            value=False,
            key="preset_enable_facility",
        )
        enable_network = st.toggle(
            "Upgrade Rescue network",
            value=False,
            key="preset_enable_network",
            help="Referral capacity and emergency transfer",
        )
        supply_scenario = st.radio(
            "Supply relative to demand",
            ["Match Demand", "Cannot Meet Demand"],
            horizontal=True,
            key="preset_supply_scenario",
        )

    d = HSS_DEMAND_PRESETS[intensity]
    match_supply = supply_scenario == "Match Demand"

    i_flags["flag_CHV"] = 0
    i_flags["flag_ANC"] = 0
    i_flags["flag_LB"] = 0
    i_HSS["P_ANC"] = slider_params["p_ANC_base_slider"]
    i_HSS["P_L45"] = slider_params["base_p_45_slider"]
    i_HSS["tau_decay"] = 6
    i_HSS["CHV_memory"] = "Always Forget"

    if enable_chv:
        i_flags["flag_CHV"] = 1
        i_flags["flag_ANC"] = 1
        i_flags["flag_LB"] = 1
        i_HSS["P_ANC"] = d["P_ANC"] / 100.0
        p_l45_exp = get_P_l45(i_HSS["P_ANC"], slider_params)
        min_l45 = round(p_l45_exp * 100) if p_l45_exp is not None else 0
        i_HSS["P_L45"] = max(min_l45, d["P_L45"]) / 100.0
        i_HSS["tau_decay"] = 6
        i_HSS["CHV_memory"] = "Logistic Decay"
        st.session_state["P_ANC"] = i_HSS["P_ANC"]
        st.session_state["P_L45"] = i_HSS["P_L45"]

    i_flags["flag_performance"] = 0
    i_flags["flag_capacity"] = 0
    i_flags["flag_labor"] = 0
    i_flags["flag_equipment"] = 0
    i_HSS["knowledge"] = slider_params["base_knowledge_L45_slider"]
    i_HSS["capacity_added"] = 0
    i_HSS["labor_ratio"] = 0
    i_HSS["sensor_ratio"] = 0

    if enable_facility:
        if match_supply:
            performance_pct = 100
            capacity_pct = HSS_CAPACITY_MATCH[intensity]
            labor_pct = 100
            equipment_pct = 100
        else:
            performance_pct = 75
            capacity_pct = HSS_CAPACITY_MISMATCH[intensity]
            labor_pct = 50
            equipment_pct = 50
        i_flags["flag_performance"] = 1
        i_flags["flag_capacity"] = 1
        i_flags["flag_labor"] = 1
        i_flags["flag_equipment"] = 1
        i_HSS["knowledge"] = performance_pct / 100.0
        i_HSS["capacity_added"] = capacity_pct / 100.0
        i_HSS["labor_ratio"] = labor_pct / 100.0
        i_HSS["sensor_ratio"] = equipment_pct / 100.0

    i_flags["flag_refer_voucher"] = 0
    i_flags["flag_transfer_capacity"] = 0
    i_HSS["P_refer"] = 0
    i_HSS["transfer_capacity_target"] = 0.0

    if enable_network:
        if match_supply:
            refer_pct = 100
        else:
            refer_pct = 50
        i_flags["flag_refer_voucher"] = 1
        i_HSS["P_refer"] = refer_pct / 100.0
        i_flags["flag_transfer_capacity"] = 1
        i_HSS["transfer_capacity_target"] = float(refer_pct)

    i_flags["flag_SDR"] = 1 if (enable_chv or enable_facility or enable_network) else 0


def render_single_preset():
    """All (Preset): treatment/diagnosis as enable toggles; parameters use analyst UI defaults."""
    st.subheader(":pill: Treatment interventions (Preset)")
    mat_specs = [
        ("PPH bundle", "flag_pph_bundle", "pph_bundle"),
        ("IV iron infusion", "flag_iv_iron", "iv_iron"),
        ("Magnesium sulfate (MgSO4)", "flag_MgSO4", "MgSO4"),
        ("Antibiotics for maternal sepsis", "flag_antibiotics", "antibiotics"),
        ("Oxytocin for prolonged labor", "flag_oxytocin", "oxytocin"),
    ]
    for label, flag_key, s_key in mat_specs:
        on = st.toggle(label, value=False, key=f"preset_single_{flag_key}")
        if on:
            i_flags[flag_key] = 1
            i_S[s_key] = 1.0
        else:
            i_flags[flag_key] = 0
            i_S[s_key] = 0

    st.markdown("---")
    st.subheader(":stethoscope: Diagnosis interventions (Preset)")
    diag_col1, diag_col2, diag_col3 = st.columns(3)
    with diag_col1:
        us_on = st.toggle("AI portable ultrasound (AI-US)", value=False, key="preset_single_us")
    with diag_col2:
        sensor_on = st.toggle("Intrapartum sensors", value=False, key="preset_single_intrasensor")
    with diag_col3:
        ai_sensor_on = st.toggle(
            "AI algorithms (intrapartum)",
            value=False,
            key="preset_single_sensor_ai",
            disabled=not sensor_on,
        )

    if us_on:
        i_flags["flag_us"] = 1
        i_E["sens_us"] = 0.95
        i_E["spec_us"] = 0.95
        i_S["US"] = 1
    else:
        i_flags["flag_us"] = 0
        i_S["US"] = 0

    if sensor_on:
        i_flags["flag_intrasensor"] = 1
        if ai_sensor_on:
            i_flags["flag_sensor_ai"] = 1
            i_E["sens_sensor"] = 0.95
            i_E["spec_sensor"] = 0.95
        else:
            i_flags["flag_sensor_ai"] = 0
    else:
        i_flags["flag_intrasensor"] = 0
        i_flags["flag_sensor_ai"] = 0


def render_prompts_preset():
    """All (Preset): MOMISH as checkboxes; low/high fidelity presets for PROMPTS and MENTORS only."""
    st.subheader(":bulb: MOMISH interventions (Preset)")

    st.markdown("**PROMPTS**")
    p_col1, p_col2 = st.columns([1, 2])
    with p_col1:
        prompts_on = st.checkbox("Enable PROMPTS", value=False, key="preset_prompts_on")
    with p_col2:
        prompts_fidelity = st.radio(
            "PROMPTS fidelity",
            ["Moderate Fidelity", "High Fidelity"],
            horizontal=True,
            key="preset_prompts_fidelity",
            disabled=not prompts_on,
        )

    i_flags["flag_PROMPTS"] = 1 if prompts_on else 0
    st.session_state["flag_PROMPTS"] = int(prompts_on)
    if prompts_on:
        fid_key = "Moderate" if prompts_fidelity.startswith("Moderate") else "High"
        prompts_index = get_prompts_implementation_index(fid_key)
        i_HSS["prompts_implementation_index"] = prompts_index
        i_HSS["prompts_rr_anc4p"] = PROMPTS_RR_OVERRIDE[fid_key]
        st.session_state["prompts_implementation_index"] = prompts_index
    else:
        i_HSS["prompts_implementation_index"] = 0.0
        i_HSS["prompts_rr_anc4p"] = 1.0
        st.session_state["prompts_implementation_index"] = 0.0

    st.markdown("**MENTORS**")
    m_col1, m_col2 = st.columns([1, 2])
    with m_col1:
        mentor_on = st.checkbox("Enable MENTORS", value=False, key="preset_mentor_on")
    with m_col2:
        mentor_fidelity = st.radio(
            "MENTORS fidelity",
            ["Moderate Fidelity", "High Fidelity"],
            horizontal=True,
            key="preset_mentor_fidelity",
            disabled=not mentor_on,
        )

    i_flags["flag_MENTOR"] = 1 if mentor_on else 0
    st.session_state["flag_MENTOR"] = int(mentor_on)
    if mentor_on:
        fid_key = "Moderate" if mentor_fidelity.startswith("Moderate") else "High"
        mentor_index = get_mentor_implementation_index(fid_key)
        i_HSS["mentor_implementation_index"] = mentor_index
        st.session_state["mentor_implementation_index"] = mentor_index
    else:
        i_HSS["mentor_implementation_index"] = 0.0
        st.session_state["mentor_implementation_index"] = 0.0

    st.markdown("**Other MOMISH programs**")

    p_col1, p_col2 = st.columns([1, 2])
    with p_col1:
        pulse_on = st.checkbox("Enable PULSE", value=False, key="preset_pulse_on")
    with p_col2:
        pulse_fidelity = st.radio(
            "PULSE fidelity",
            ["Moderate Fidelity", "High Fidelity"],
            horizontal=True,
            key="preset_pulse_fidelity",
            disabled=not pulse_on,
        )

    i_flags["flag_pulse"] = 1 if pulse_on else 0
    st.session_state["flag_pulse"] = int(pulse_on)
    if pulse_on:
        fid_key = "Moderate" if pulse_fidelity.startswith("Moderate") else "High"
        pulse_index = get_pulse_implementation_index(fid_key)
        i_HSS["pulse_implementation_index"] = pulse_index
        st.session_state["pulse_implementation_index"] = pulse_index
    else:
        i_HSS["pulse_implementation_index"] = 0.0
        st.session_state["pulse_implementation_index"] = 0.0

    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        fqa_on = st.checkbox("Enable FQA", value=False, key="preset_fqa_on")
    with f_col2:
        fqa_fidelity = st.radio(
            "FQA fidelity",
            ["Moderate Fidelity", "High Fidelity"],
            horizontal=True,
            key="preset_fqa_fidelity",
            disabled=not fqa_on,
        )

    i_flags["flag_fqa"] = 1 if fqa_on else 0
    st.session_state["flag_fqa"] = int(fqa_on)
    if fqa_on:
        fid_key = "Moderate" if fqa_fidelity.startswith("Moderate") else "High"
        fqa_index = get_fqa_implementation_index(fid_key)
        i_HSS["fqa_implementation_index"] = fqa_index
        st.session_state["fqa_implementation_index"] = fqa_index
    else:
        i_HSS["fqa_implementation_index"] = 0.0
        st.session_state["fqa_implementation_index"] = 0.0

    selected_fqa_pulse_level = st.selectbox(
        "FQA amplification of PULSE effect",
        options=FQA_PULSE_MODIFIER_LEVELS,
        index=fqa_pulse_modifier_default_index(),
        key="preset_fqa_pulse_modifier_level",
    )
    i_HSS["fqa_pulse_modifier_level"] = selected_fqa_pulse_level
    i_HSS["fqa_pulse_modifier"] = FQA_PULSE_MODIFIER_OPTIONS[selected_fqa_pulse_level]
    st.session_state["fqa_pulse_modifier_level"] = selected_fqa_pulse_level

    selected_pulse_boost_level = st.selectbox(
        "PULSE implementation boost to other MOMISH interventions",
        options=PULSE_IMPLEMENTATION_BOOST_LEVELS,
        index=pulse_implementation_boost_default_index(),
        key="preset_pulse_implementation_boost_level",
        help="Maximum increase in implementation index a fully-implemented PULSE system "
             "lends to PROMPTS, MENTORS, and FQA when PULSE is enabled.",
    )
    i_HSS["pulse_implementation_boost_level"] = selected_pulse_boost_level
    i_HSS["pulse_implementation_boost"] = PULSE_IMPLEMENTATION_BOOST_OPTIONS[selected_pulse_boost_level]
    st.session_state["pulse_implementation_boost_level"] = selected_pulse_boost_level

    bl_col1, bl_col2 = st.columns([1, 2])
    with bl_col1:
        blood_on = st.checkbox("Enable Blood tracking", value=False, key="preset_blood_on")
    with bl_col2:
        blood_fidelity = st.radio(
            "Blood tracking fidelity",
            ["Current Fidelity", "Moderate Fidelity", "High Fidelity"],
            horizontal=True,
            key="preset_blood_fidelity",
            disabled=not blood_on,
        )

    i_flags["flag_blood"] = 1 if blood_on else 0
    i_flags["flag_blood_tracking"] = i_flags["flag_blood"]
    st.session_state["flag_blood"] = int(blood_on)
    if blood_on:
        if blood_fidelity.startswith("Current"):
            fid_key = "Current"
        elif blood_fidelity.startswith("Moderate"):
            fid_key = "Moderate"
        else:
            fid_key = "High"
        blood_index = get_blood_tracking_implementation_index(fid_key)
        i_HSS["blood_adoption"] = blood_index
        st.session_state["blood_adoption"] = blood_index
    else:
        i_HSS["blood_adoption"] = 0.0
        st.session_state["blood_adoption"] = 0.0

    te_col1, te_col2 = st.columns([1, 2])
    with te_col1:
        transfer_emt_on = st.checkbox("Enable Transfer & EMT", value=False, key="preset_transfer_emt_on")
    with te_col2:
        transfer_emt_fidelity = st.radio(
            "Transfer & EMT fidelity",
            ["Current Fidelity", "Moderate Fidelity", "High Fidelity"],
            horizontal=True,
            key="preset_transfer_emt_fidelity",
            disabled=not transfer_emt_on,
        )

    i_flags["flag_transfer_delay"] = 1 if transfer_emt_on else 0
    st.session_state["flag_transfer_delay"] = int(transfer_emt_on)
    if transfer_emt_on:
        if transfer_emt_fidelity.startswith("Current"):
            fid_key = "Current"
        elif transfer_emt_fidelity.startswith("Moderate"):
            fid_key = "Moderate"
        else:
            fid_key = "High"
        transfer_index = get_transfer_implementation_index(fid_key)
        i_HSS["referral_implementation_index"] = transfer_index
        st.session_state["referral_implementation_index"] = transfer_index
    else:
        i_HSS["referral_implementation_index"] = 0.0
        st.session_state["referral_implementation_index"] = 0.0


# Function to render HSS interventions
def render_hss(preset_demand_scenario, preset_supply_scenario):
    # Scenario default values
    Demand_scenarios = {
        "Conservative": {"P_ANC": 70, "P_L45": 53},
        "Moderate": {"P_ANC": 80, "P_L45": 68},
        "Aggressive": {"P_ANC": 90, "P_L45": 90}
    }

    Capacity_match = {
        "Conservative": 25.0,
        "Moderate": 50.0,
        "Aggressive": 85.0
    }

    Capacity_dismatch = {
        "Conservative": 12.5,
        "Moderate": 25.0,
        "Aggressive": 42.5
    }

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(":chart_with_upwards_trend: HSS interventions (Demand)",
                     help = "Goal: Increase pregnant mothers' demand for \n\n Antenatal Care (ANC) and deliveries at L4/5 facilties")

        if preset_demand_scenario is not None:
            Employ_CHV = 1
            Increase_ANC = 1
            Increase_LB45 = 1
            P_ANC_preset_value = Demand_scenarios[preset_demand_scenario]["P_ANC"]
            P_L45_preset_value = Demand_scenarios[preset_demand_scenario]["P_L45"]
        else:
            Employ_CHV = 0
            Increase_ANC = 0
            Increase_LB45 = 0
            P_ANC_preset_value = None
            P_L45_preset_value = None

        col1_1, col1_2 = st.columns(2)
        with col1_1:
            st.text('Apply interventions')
            CHVint = st.toggle('Employ CHVs', value = Employ_CHV,
                               help = "CHVs refer to Community Healthcare Workers")

            if CHVint:
                i_flags['flag_SDR'] = 1

        with col1_2:
            st.text('Adjust parameters')
            if CHVint:
                i_flags['flag_CHV'] = 1
            else:
                i_flags['flag_CHV'] = 0
        col1_3, col1_4 = st.columns(2)

        with col1_3:
            ANC_int = st.checkbox('Increasing 4+ANC visits', value = Increase_ANC,
                                  help = "Increase the percentage of pregnant women have 4+ANCs",) if CHVint else 0

        with col1_4:
            if ANC_int:
                i_flags['flag_ANC'] = 1
                if P_ANC_preset_value is None:
                    P_ANC_value = round(st.session_state.get('P_ANC', 0.9) * 100)
                else:
                    P_ANC_value = P_ANC_preset_value

                i_HSS["P_ANC"] = st.slider('Expected **4+ANC rate**', min_value=round(slider_params['p_ANC_base_slider'] * 100), max_value=100, step=2, value=P_ANC_value, format="%d%%",
                                     help = "Value = 90% means 90% of pregnant mothers will attend 4+ANCs (WHO target)")
                i_HSS["P_ANC"] /= 100
                st.session_state['P_ANC'] = i_HSS["P_ANC"]

                P_l45_exp = get_P_l45(i_HSS["P_ANC"], slider_params)
                P_L45_slider = round(P_l45_exp * 100) if P_l45_exp is not None else 0
            else:
                P_L45_slider = round(slider_params['base_p_45_slider'] * 100)

        col1_5, col1_6 = st.columns(2)
        with col1_5:
            LB_int = st.checkbox('Increasing live births in L4/5 facilities', value = Increase_LB45,
                                 ) if CHVint else 0

        with col1_6:
            if LB_int:
                i_flags['flag_LB'] = 1
                min_value_L45 = P_L45_slider
                # Ensure the stored value is within the new range
                if P_L45_preset_value is None:
                    P_L45_value = max(min_value_L45, round(st.session_state.get('P_L45', 0.9) * 100))
                else:
                    P_L45_value = max(min_value_L45, P_L45_preset_value)

                i_HSS["P_L45"] = st.slider("Expected **% live births at L4/5** before transfer", min_value=min_value_L45, max_value=100, step=1, value=P_L45_value, format="%d%%",
                                        help = "Value = 90% means 90% of deliveries will happen at L4/5 facilities before emergency transfer")

                i_HSS["P_L45"] /= 100

                i_HSS['tau_decay'] = st.slider(
                    "**Memory decay** period (months)",
                    min_value=1,
                    max_value=36,
                    value=6,
                    step=1,
                    help="Defines how long past capacity constraints affect intention to deliver at L4/5 (in months)."
                )

                # Update session state
                st.session_state['P_L45'] = i_HSS["P_L45"]

        # # # =========================
        # # # PROMPTS (Demand-side)
        # # # =========================
        # # st.markdown("---")
        # # st.text("PROMPTS (Engagement program)")

        # colP1, colP2 = st.columns(2)
        # with colP1:
        #     prompts_int = st.checkbox(
        #         "Enable PROMPTS",
        #         value=bool(st.session_state.get("flag_PROMPTS", 0)),
        #         help="Enable PROMPTS engagement program (affects engagement-related mechanisms in LB_effect)."
        #     )

        # with colP2:
        #     i_flags["flag_PROMPTS"] = 1 if prompts_int else 0
        #     st.session_state["flag_PROMPTS"] = int(prompts_int)

        # if prompts_int:
        #     colP3, colP4, colP5 = st.columns(3)

        #     with colP3:
        #         adoption_default = int(st.session_state.get("adoption_prompts", 100))
        #         adoption_val = st.slider(
        #             "PROMPTS adoption",
        #             min_value=0, max_value=100, step=5,
        #             value=adoption_default,
        #             format="%d%%",
        #             help="Program adoption level."
        #         )
        #         i_HSS["adoption_prompts"] = adoption_val / 100.0
        #         st.session_state["adoption_prompts"] = adoption_val

        #     with colP4:
        #         engage_default = int(st.session_state.get("chv_engagement", 100))
        #         engage_val = st.slider(
        #             "CHV engagement (PROMPTS)",
        #             min_value=0, max_value=100, step=5,
        #             value=engage_default,
        #             format="%d%%",
        #             help="CHV engagement level used ONLY inside PROMPTS."
        #         )
        #         i_HSS["chv_engagement"] = engage_val / 100.0
        #         st.session_state["chv_engagement"] = engage_val

        #     with colP5:
        #         prompts_effect_default = int(st.session_state.get("intervention_fidelity", 100))
        #         prompts_effect_val = st.slider(
        #             "Intervention Fidelity",
        #             min_value=87, max_value=100, step=5,
        #             value=prompts_effect_default,
        #             format="%d%%",
        #             help="Effectiveness of PROMPTS in increasing intention to deliver at L4/5."
        #         )
        #         i_HSS["prompts_effect"] = prompts_effect_val / 100.0
        #         st.session_state["prompts_effect"] = prompts_effect_val


        # else:
        #     i_HSS["adoption_prompts"] = 0.0
        #     i_HSS["chv_engagement"] = 0.0
        #     i_HSS["intervention_fidelity"] = 0.0

        col1_7, col1_8 = st.columns(2)
        with col1_7:
            if CHVint:
                i_HSS['CHV_memory'] = st.selectbox("**CHV memory decay model**",
                                            options=["Logistic Decay", "Always Forget", "Always Remember"],
                                            index=0,
                                            help="Defines how long CHVs remember past negative quality of care expressed by mothers \n\n"
                                                    "Logistic Decay: CHVs gradually forget past negative quality of care \n\n"
                                                    "Always Forget: CHVs always forget past negative quality of care \n\n"
                                                    "Always Remember: CHVs always remember past negative quality of care")


    with col2:
        st.subheader(":hospital: HSS interventions (Supply)",
                     help="Goal: Increase supply of L4/5 facilities and rescue network \n\n for supporting the increased demand"
                     )
        if preset_supply_scenario == "Match Demand":
            upgrade_L45_facilities = 1
            upgrade_performance = 1
            upgrade_capacity = 1
            upgrade_labor = 1
            upgrade_equipment = 1
            update_transport = 1
            update_refer = 1
            performance_value = 100
            capacity_added_value = Capacity_match[preset_demand_scenario]
            labor_value = 100
            equipment_value = 100
            refer_value = 100
        elif preset_supply_scenario == "Cannot Meet Demand":
            upgrade_L45_facilities = 1
            upgrade_performance = 1
            upgrade_capacity = 1
            upgrade_labor = 1
            upgrade_equipment = 1
            update_transport = 1
            update_refer = 1
            performance_value = 75
            capacity_added_value = Capacity_dismatch[preset_demand_scenario]
            labor_value = 50
            equipment_value = 50
            refer_value = 50
        elif preset_supply_scenario is None:
            upgrade_L45_facilities = 0
            upgrade_performance = 0
            upgrade_capacity = 0
            upgrade_labor = 0
            upgrade_equipment = 0
            update_transport = 0
            update_refer = 0
            performance_value = round(slider_params['base_knowledge_L45_slider'] * 100)
            capacity_added_value = 0.0
            labor_value = 0
            equipment_value = 0
            refer_value = 0

        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.text('Apply interventions')
            facint = st.toggle('Upgrade L4/5 facilities', value = upgrade_L45_facilities,
                                 help = "To ensure high standards of obstetric and newborn care services")
            if facint:
                i_flags['flag_SDR'] = 1

        with col2_2:
            st.text('Adjust parameters')

        col2_3, col2_4 = st.columns(2)
        with col2_3:
            if facint:
                performanceint = st.checkbox('Improve performance of healthcare workers', value = upgrade_performance,
                                            help = "To ensure healthcare workers follow protocols of single interventions")
            else:
                performanceint = 0

        with col2_4:
            if performanceint:
                i_flags['flag_performance'] = 1
                i_HSS["knowledge"] = st.slider("The **performance** level of healthcare workers",
                                               min_value=round(slider_params['base_knowledge_L45_slider'] * 100), max_value=100, step=5,
                                               value=performance_value, format="%d%%",
                                               help="Performance reflects the likelihood of following protocols of single interventions \n\n"
                                                    "Value = 80% means the performance level of healthcare workers is 80 out of 100")
                i_HSS["knowledge"] /= 100
            else:
                i_flags['flag_performance'] = 0
                i_HSS["knowledge"] = slider_params['base_knowledge_L45_slider']

        col2_5, col2_6 = st.columns(2)
        with col2_5:
            if facint:
                capacityint = st.checkbox('Increase facility capacity', value = upgrade_capacity,
                                        help = "Improving infrastructure to ensure the facility can handle the increased demand")
            else:
                capacityint = 0
        with col2_6:
            if capacityint:
                i_flags['flag_capacity'] = 1
                i_HSS["capacity_added"] = st.slider("The **facility capacity** increases by", min_value=0.0, max_value=100.0, step=5.0, value=capacity_added_value, format="%d%%",
                                       help = "Value = 100% means the capacity of L4/5 facilities will be increased by 100% (2 times)")

                i_HSS["capacity_added"] /= 100
            else:
                i_flags['flag_capacity'] = 0
                i_HSS["capacity_added"] = 0

        col2_7, col2_8 = st.columns(2)
        with col2_7:
            if facint:
                laborint = st.checkbox('Increase number of skilled labor force', value = upgrade_labor,
                                        help = "To ensure the facility has enough skilled labor force to handle the increased demand")
            else:
                laborint = 0
        with col2_8:
            if laborint:
                i_flags['flag_labor'] = 1
                i_HSS["labor_ratio"] = st.slider("The **ideal number of skilled labor force** in L4/5", min_value=0,
                                                 max_value=100, step=5, value=labor_value,
                                                 format="%d%%", help="Value = 50% means 50% of the ideal number of skilled labor force in L4/5 facilities")

                i_HSS["labor_ratio"] /= 100
            else:
                i_flags['flag_labor'] = 0
                i_HSS["labor_ratio"] = 0

        col2_9, col2_10 = st.columns(2)
        with col2_9:
            if facint:
                equipment_int = st.checkbox('Increasing equipment in L4/5 facilities', value = upgrade_equipment,
                                        help = "To ensure the facility has enough equipment to handle the increased demand")
            else:
                equipment_int = 0
        with col2_10:
            if equipment_int:
                i_flags['flag_equipment'] = 1
                i_HSS["sensor_ratio"] = st.slider("The **ideal number of intrapartum sensors** in L4/5", min_value=0,
                                                    max_value=100, step=5, value=equipment_value,
                                                    format="%d%%", help="Value = 50% means 50% of the ideal number of intrapartum sensors in L4/5 facilities")

                i_HSS["sensor_ratio"] /= 100
            else:
                i_flags['flag_equipment'] = 0
                i_HSS["sensor_ratio"] = 0

        st.markdown("---")
        col2_11, col2_12 = st.columns(2)
        with col2_11:
            refint = st.toggle('Upgrade Rescue network', value = update_transport,
                                 help="To support the increased referrals/transfers \n\n from home or L2/3 facilities to L4/5 facilities")

        with col2_12:
            if refint:
                i_flags['flag_SDR'] = 1

        col2_13, col2_14 = st.columns(2)
        with col2_13:
            if refint:
                referint = st.checkbox('Improve referral capacity', value = update_refer,
                                        help="Increasing taxies/bodas for referring mothers to L4/5 facilities")
            else:
                referint = 0
        with col2_14:
            if referint:
                i_flags['flag_refer_voucher'] = 1
                i_HSS["P_refer"] = st.slider("% referral with free taxies/bodas", min_value=0, max_value=100, step=1, value=refer_value,
                                      format="%d%%", help="Value = 100 means 100% of referral can use free taxies/bodas")
                i_HSS["P_refer"] /= 100
            else:
                i_flags['flag_refer_voucher'] = 0
                i_HSS["P_refer"] = 0

        col2_15, col2_16 = st.columns(2)
        with col2_15:
            if referint:
                transfer_capacity_int = st.checkbox(
                    'Referral with Emergency Transfer',
                    value=bool(update_refer),
                    help="Improve emergency transfer capacity from L2/3 to L4/5 facilities",
                )
            else:
                transfer_capacity_int = 0
        with col2_16:
            if transfer_capacity_int:
                i_flags['flag_transfer_capacity'] = 1
                i_HSS["transfer_capacity_target"] = st.slider(
                    "% emergency transfers supported",
                    min_value=0,
                    max_value=100,
                    step=1,
                    value=refer_value,
                    format="%d%%",
                    help="Target L2/3-to-L4/5 emergency transfer capacity. Existing baseline values are preserved when higher.",
                )
            else:
                i_flags['flag_transfer_capacity'] = 0
                i_HSS["transfer_capacity_target"] = 0.0



def sync_param_momish_from_hss(i_param, i_HSS, flags):
    """Align top-level param keys with dashboard i_HSS (LB_effect + intrapartum may read either)."""
    if i_HSS.get("prompts_implementation_index") is not None:
        i_param["prompts_implementation_index"] = float(i_HSS["prompts_implementation_index"])
    if i_HSS.get("prompts_rr_anc4p") is not None:
        i_param["prompts_rr_anc4p"] = float(i_HSS["prompts_rr_anc4p"])
    if i_HSS.get("fqa_pulse_modifier") is not None:
        i_param["fqa_pulse_modifier_level"] = i_HSS.get("fqa_pulse_modifier_level", "Medium")
        i_param["fqa_pulse_modifier"] = float(i_HSS["fqa_pulse_modifier"])
    if i_HSS.get("mentor_implementation_index") is not None:
        i_param["mentors_implementation_index"] = float(i_HSS["mentor_implementation_index"])
    if i_HSS.get("pulse_implementation_index") is not None:
        i_param["pulse_implementation_index"] = float(i_HSS["pulse_implementation_index"])
    if i_HSS.get("fqa_implementation_index") is not None:
        i_param["fqa_implementation_index"] = float(i_HSS["fqa_implementation_index"])
    if i_HSS.get("pulse_implementation_boost") is not None:
        i_param["pulse_implementation_boost_level"] = i_HSS.get("pulse_implementation_boost_level", "Moderate")
        i_param["pulse_implementation_boost"] = float(i_HSS["pulse_implementation_boost"])
    if i_HSS.get("referral_implementation_index") is not None:
        i_param["referral_implementation_index"] = float(i_HSS["referral_implementation_index"])

    # Snapshot FQA's own implementation index *before* PULSE's cross-boost below.
    # pulse_effect() (LB_effect.py) reads this snapshot for its own strength
    # formula -- if it read the boosted value instead, PULSE would indirectly
    # amplify itself through FQA's fqa_pulse_modifier term, which was
    # calibrated against FQA's un-boosted level, not a PULSE-inflated one.
    i_param["fqa_pulse_amplifier_index"] = float(i_param.get("fqa_implementation_index", 0.0))

    # PULSE implementation boost: when PULSE is enabled, it lends some of its
    # own implementation index to PROMPTS, MENTORS, FQA, and REFERRAL (not
    # Blood, and not itself). Computed once here rather than inside each
    # intervention's own per-month block, so none of them need to know PULSE
    # exists.
    if flags.get("flag_pulse", 0):
        pulse_implementation_index = float(i_param.get("pulse_implementation_index", 0.0))
        pulse_implementation_boost = float(i_param.get("pulse_implementation_boost", 0.0))
        fqa_pulse_modifier = float(i_param.get("fqa_pulse_modifier", 0.0))
        flag_fqa = float(flags.get("flag_fqa", 0))

        pulse_implementation_boost_effective = float(np.clip(
            pulse_implementation_index * pulse_implementation_boost * (1 + flag_fqa * fqa_pulse_modifier),
            0.0,
            1.0,
        ))

        for key in (
            "prompts_implementation_index",
            "mentors_implementation_index",
            "fqa_implementation_index",
            "referral_implementation_index",
        ):
            i_param[key] = float(np.clip(
                i_param.get(key, 0.0) + pulse_implementation_boost_effective,
                0.0,
                1.0,
            ))


def apply_momish_facility_delivery(i_flags, i_HSS, slider_params, choice_key, intervention_selection):
    """
    Map MOMISH "Overall facility delivery" to HSS infrastructure (no sliders here).
    - high: same numeric presets as render_hss(Aggressive, Match Demand)
    - low:  same as render_hss(Conservative, Match Demand)
    - off:  baseline / no HSS demand–supply stack (does not clear PROMPTS/MENTOR/PULSE fields)
    - follow: only used when intervention_selection == \"Both\" — keep i_HSS/i_flags from render_hss above
    """
    if choice_key == "follow":
        return

    Demand_scenarios = {
        "Conservative": {"P_ANC": 70, "P_L45": 53},
        "Moderate": {"P_ANC": 80, "P_L45": 68},
        "Aggressive": {"P_ANC": 90, "P_L45": 90},
    }
    Capacity_match = {
        "Conservative": 25.0,
        "Moderate": 50.0,
        "Aggressive": 85.0,
    }

    if choice_key == "off":
        i_flags["flag_SDR"] = 0
        i_flags["flag_CHV"] = 0
        i_flags["flag_ANC"] = 0
        i_flags["flag_LB"] = 0
        i_flags["flag_performance"] = 0
        i_flags["flag_capacity"] = 0
        i_flags["flag_labor"] = 0
        i_flags["flag_equipment"] = 0
        i_flags["flag_refer_voucher"] = 0
        i_flags["flag_transfer_capacity"] = 0
        i_HSS["P_ANC"] = slider_params["p_ANC_base_slider"]
        i_HSS["P_L45"] = slider_params["base_p_45_slider"]
        i_HSS["knowledge"] = slider_params["base_knowledge_L45_slider"]
        i_HSS["capacity_added"] = 0
        i_HSS["labor_ratio"] = 0
        i_HSS["sensor_ratio"] = 0
        i_HSS["P_refer"] = 0
        i_HSS["transfer_capacity_target"] = 0.0
        i_HSS["tau_decay"] = 6
        i_HSS["CHV_memory"] = "Always Forget"
        i_HSS["referadded"] = 0
        i_HSS["transadded"] = 0
        i_HSS["supply_level"] = 0
        st.session_state["P_ANC"] = i_HSS["P_ANC"]
        st.session_state["P_L45"] = i_HSS["P_L45"]
        return

    scenario = "Aggressive" if choice_key == "high" else "Conservative"
    d = Demand_scenarios[scenario]
    i_flags["flag_SDR"] = 1
    i_flags["flag_CHV"] = 1
    i_flags["flag_ANC"] = 1
    i_flags["flag_LB"] = 1
    i_flags["flag_performance"] = 1
    i_flags["flag_capacity"] = 1
    i_flags["flag_labor"] = 1
    i_flags["flag_equipment"] = 1
    i_flags["flag_refer_voucher"] = 1
    i_flags["flag_transfer_capacity"] = 1

    i_HSS["P_ANC"] = d["P_ANC"] / 100.0
    p_l45_exp = get_P_l45(i_HSS["P_ANC"], slider_params)
    min_l45 = round(p_l45_exp * 100) if p_l45_exp is not None else 0
    p_l45_pct = max(min_l45, d["P_L45"])
    i_HSS["P_L45"] = p_l45_pct / 100.0
    i_HSS["tau_decay"] = 6
    i_HSS["CHV_memory"] = "Logistic Decay"

    i_HSS["knowledge"] = 1.0
    i_HSS["capacity_added"] = Capacity_match[scenario] / 100.0
    i_HSS["labor_ratio"] = 1.0
    i_HSS["sensor_ratio"] = 1.0
    i_HSS["P_refer"] = 1.0
    i_HSS["transfer_capacity_target"] = 100.0

    st.session_state["P_ANC"] = i_HSS["P_ANC"]
    st.session_state["P_L45"] = i_HSS["P_L45"]


def render_prompts():
    st.subheader(":bulb: MOMISH Interventions")

    _intervention = st.session_state.get("intervention_selection")
    if _intervention == "Both":
        _facility_spec = [
            ("Use HSS from Scenario Settings", "follow"),
            ("Low overall facility delivery", "low"),
            ("High overall facility delivery", "high"),
            ("Not enabled (no HSS)", "off"),
        ]
    else:
        _facility_spec = [
            ("Not enabled (no HSS)", "off"),
            ("Low — Conservative preset", "low"),
            ("High — Aggressive preset", "high"),
        ]
    _facility_labels = [x[0] for x in _facility_spec]
    _facility_map = dict(_facility_spec)
    st.markdown("**Overall facility delivery**")
    _facility_choice_label = st.radio(
        "HSS demand/supply context for PROMPTS & MENTORS (no sliders)",
        _facility_labels,
        horizontal=True,
        key=f"momish_facility_delivery_radio_{_intervention}",
        help=(
            "High / Low apply the same numeric presets as HSS **Scenario** mode (Aggressive or Conservative) with **Match Demand** supply. "
            "Not enabled turns off the HSS demand–supply stack (baseline facility context). "
            + (
                "In **Both** mode, the first option keeps the HSS values from the Scenario Settings expander above."
                if _intervention == "Both"
                else "In **MOMISH-only** mode there is no separate HSS screen — pick Low/High here to embed facility context."
            )
        ),
    )
    _facility_choice_key = _facility_map[_facility_choice_label]

    prompts_enabled = st.toggle(
        "Enable PROMPTS",
        value=bool(st.session_state.get("flag_PROMPTS", 0)),
        help="Enable PROMPTS engagement",
        key="prompts_enable"
    )

    i_flags["flag_PROMPTS"] = 1 if prompts_enabled else 0
    st.session_state["flag_PROMPTS"] = int(prompts_enabled)

    if prompts_enabled:
        prompts_fidelity_options = ["Current", "Moderate", "High"]
        prompts_fidelity_choice_default = st.session_state.get(
            "prompts_implementation_level", "Current"
        )
        if prompts_fidelity_choice_default not in prompts_fidelity_options:
            prompts_fidelity_choice_default = "Current"

        prompts_fidelity_choice = st.selectbox(
            "PROMPTS implementation level",
            options=prompts_fidelity_options,
            index=prompts_fidelity_options.index(prompts_fidelity_choice_default),
            key="prompts_implementation_level_select",
            help="Choose the PROMPTS implementation level. The selected value is passed to the model "
                 "as an implementation index (share of mothers reached) plus a matching RR on 4+ ANC.",
        )
        st.session_state["prompts_implementation_level"] = prompts_fidelity_choice

        prompts_index = get_prompts_implementation_index(prompts_fidelity_choice)
        i_HSS["prompts_implementation_index"] = prompts_index
        i_HSS["prompts_rr_anc4p"] = PROMPTS_RR_OVERRIDE[prompts_fidelity_choice]
        st.session_state["prompts_implementation_index"] = prompts_index
    else:
        i_HSS["prompts_implementation_index"] = 0.0
        i_HSS["prompts_rr_anc4p"] = 1.0
        st.session_state["prompts_implementation_index"] = 0.0

    # ==========================================================
    # BLOCK 2 — MENTORS
    # ==========================================================
    col5, col6 = st.columns(2)

    with col5:
        mentor_on = st.toggle(
            "MENTORS Intervention",
            value=bool(st.session_state.get("flag_MENTOR", 0)),
            key="mentor_enable"
        )

    with col6:
        i_flags["flag_MENTOR"] = 1 if mentor_on else 0
        st.session_state["flag_MENTOR"] = int(mentor_on)

    if mentor_on:

        mentor_fidelity_options = ["Current", "Moderate", "High"]
        mentor_fidelity_choice_default = st.session_state.get(
            "mentor_session_fidelity_bundle", "Current"
        )
        if mentor_fidelity_choice_default not in mentor_fidelity_options:
            mentor_fidelity_choice_default = "Current"

        mentor_fidelity_choice = st.selectbox(
            "MENTORS implementation level",
            options=mentor_fidelity_options,
            index=mentor_fidelity_options.index(mentor_fidelity_choice_default),
            key="mentor_session_fidelity_bundle_select",
            help="Choose the MENTORS implementation level. The selected value is passed to the model as a single implementation index.",
        )
        st.session_state["mentor_session_fidelity_bundle"] = mentor_fidelity_choice

        mentor_index = get_mentor_implementation_index(mentor_fidelity_choice)
        i_HSS["mentor_implementation_index"] = mentor_index
        st.session_state["mentor_implementation_index"] = mentor_index
    else:
        i_HSS["mentor_implementation_index"] = 0.0
        st.session_state["mentor_implementation_index"] = 0.0

    # ==========================================================
    # BLOCK 3 — SMS
    # ==========================================================
    col9, col10 = st.columns(2)
    with col9:
        pulse_int = st.toggle(
            "PULSE Intervention",
            value=bool(st.session_state.get("flag_pulse", 0)),
            key="pulse_enable"
        )

    with col10:
        fqa_int = st.toggle(
            "FQA Intervention",
            value=bool(st.session_state.get("flag_fqa", 0)),
            key="fqa_enable",
            help="FQA directly improves healthcare worker knowledge. When PULSE is also on, it also strengthens PULSE actionable insights.",
        )

    i_flags["flag_pulse"] = 1 if pulse_int else 0
    i_flags["flag_fqa"] = 1 if fqa_int else 0
    st.session_state["flag_pulse"] = int(pulse_int)
    st.session_state["flag_fqa"] = int(fqa_int)

    if pulse_int:
        pulse_fidelity_options = ["Current", "Moderate", "High"]
        pulse_fidelity_choice_default = st.session_state.get(
            "pulse_implementation_level", "Current"
        )
        if pulse_fidelity_choice_default not in pulse_fidelity_options:
            pulse_fidelity_choice_default = "Current"

        pulse_fidelity_choice = st.selectbox(
            "PULSE implementation level",
            options=pulse_fidelity_options,
            index=pulse_fidelity_options.index(pulse_fidelity_choice_default),
            key="pulse_implementation_level_select",
            help="Choose the PULSE implementation level. The selected value is passed to the model as a single implementation index.",
        )
        st.session_state["pulse_implementation_level"] = pulse_fidelity_choice

        pulse_index = get_pulse_implementation_index(pulse_fidelity_choice)
        i_HSS["pulse_implementation_index"] = pulse_index
        st.session_state["pulse_implementation_index"] = pulse_index
    else:
        i_HSS["pulse_implementation_index"] = 0.0
        st.session_state["pulse_implementation_index"] = 0.0

    if fqa_int:
        fqa_fidelity_options = ["Current", "Moderate", "High"]
        fqa_fidelity_choice_default = st.session_state.get(
            "fqa_implementation_level", "Current"
        )
        if fqa_fidelity_choice_default not in fqa_fidelity_options:
            fqa_fidelity_choice_default = "Current"

        fqa_fidelity_choice = st.selectbox(
            "FQA implementation level",
            options=fqa_fidelity_options,
            index=fqa_fidelity_options.index(fqa_fidelity_choice_default),
            key="fqa_implementation_level_select",
            help="Choose the FQA implementation level. The selected value is passed to the model as a single implementation index.",
        )
        st.session_state["fqa_implementation_level"] = fqa_fidelity_choice

        fqa_index = get_fqa_implementation_index(fqa_fidelity_choice)
        i_HSS["fqa_implementation_index"] = fqa_index
        st.session_state["fqa_implementation_index"] = fqa_index
    else:
        i_HSS["fqa_implementation_index"] = 0.0
        st.session_state["fqa_implementation_index"] = 0.0

    selected_fqa_pulse_level = st.selectbox(
        "FQA amplification of PULSE effect",
        options=FQA_PULSE_MODIFIER_LEVELS,
        index=fqa_pulse_modifier_default_index(),
        key="fqa_pulse_modifier_level_select",
    )
    i_HSS["fqa_pulse_modifier_level"] = selected_fqa_pulse_level
    i_HSS["fqa_pulse_modifier"] = FQA_PULSE_MODIFIER_OPTIONS[selected_fqa_pulse_level]
    st.session_state["fqa_pulse_modifier_level"] = selected_fqa_pulse_level

    selected_pulse_boost_level = st.selectbox(
        "PULSE implementation boost to other MOMISH interventions",
        options=PULSE_IMPLEMENTATION_BOOST_LEVELS,
        index=pulse_implementation_boost_default_index(),
        key="pulse_implementation_boost_level_select",
        help="Maximum increase in implementation index a fully-implemented PULSE system "
             "lends to PROMPTS, MENTORS, and FQA when PULSE is enabled.",
    )
    i_HSS["pulse_implementation_boost_level"] = selected_pulse_boost_level
    i_HSS["pulse_implementation_boost"] = PULSE_IMPLEMENTATION_BOOST_OPTIONS[selected_pulse_boost_level]
    st.session_state["pulse_implementation_boost_level"] = selected_pulse_boost_level


    # ==========================================================
    # BLOCK 4 — BLOOD INTERVENTION
    # ==========================================================
    col13, col14 = st.columns(2)
    with col13:
        blood_int = st.toggle(
            "Blood Intervention",
            value=bool(st.session_state.get("flag_blood", 0)),
            key="blood_enable"
        )

    with col14:
        i_flags["flag_blood"] = 1 if blood_int else 0
        i_flags["flag_blood_tracking"] = i_flags["flag_blood"]
        st.session_state["flag_blood"] = int(blood_int)

    if blood_int:
        blood_fidelity_options = ["Current", "Moderate", "High"]
        blood_fidelity_choice_default = st.session_state.get(
            "blood_implementation_level", "Current"
        )
        if blood_fidelity_choice_default not in blood_fidelity_options:
            blood_fidelity_choice_default = "Current"

        blood_fidelity_choice = st.selectbox(
            "Blood tracking implementation level",
            options=blood_fidelity_options,
            index=blood_fidelity_options.index(blood_fidelity_choice_default),
            key="blood_implementation_level_select",
            help="Current = 0.25, Moderate = 0.5, High = 0.95. Scales PPH/APH maternal death weights in mortality.",
        )
        st.session_state["blood_implementation_level"] = blood_fidelity_choice

        blood_index = get_blood_tracking_implementation_index(blood_fidelity_choice)
        i_HSS["blood_adoption"] = blood_index
        st.session_state["blood_adoption"] = blood_index
    else:
        i_HSS["blood_adoption"] = 0.0
        st.session_state["blood_adoption"] = 0.0

    # ==========================================================
    # BLOCK 5 — TRANSFER & EMT
    # ==========================================================
    col17, col18 = st.columns(2)
    with col17:
        transfer_emt_on = st.toggle(
            "Transfer & EMT Intervention",
            value=bool(st.session_state.get("flag_transfer_delay", 0)),
            key="transfer_emt_enable"
        )

    with col18:
        i_flags["flag_transfer_delay"] = 1 if transfer_emt_on else 0
        st.session_state["flag_transfer_delay"] = int(transfer_emt_on)

    if transfer_emt_on:
        transfer_fidelity_options = ["Current", "Moderate", "High"]
        transfer_fidelity_choice_default = st.session_state.get(
            "transfer_implementation_level", "Current"
        )
        if transfer_fidelity_choice_default not in transfer_fidelity_options:
            transfer_fidelity_choice_default = "Current"

        transfer_fidelity_choice = st.selectbox(
            "Transfer & EMT implementation level",
            options=transfer_fidelity_options,
            index=transfer_fidelity_options.index(transfer_fidelity_choice_default),
            key="transfer_implementation_level_select",
            help="Choose the Transfer & EMT implementation level. Shifts some transferred "
                 "mothers one delay category shorter (e.g. 2+h to 1-2h).",
        )
        st.session_state["transfer_implementation_level"] = transfer_fidelity_choice

        transfer_index = get_transfer_implementation_index(transfer_fidelity_choice)
        i_HSS["referral_implementation_index"] = transfer_index
        st.session_state["referral_implementation_index"] = transfer_index
    else:
        i_HSS["referral_implementation_index"] = 0.0
        st.session_state["referral_implementation_index"] = 0.0

    apply_momish_facility_delivery(
        i_flags, i_HSS, slider_params, _facility_choice_key, _intervention
    )

# Function to render Single Interventions
def render_single():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(":pill: Treatment interventions (Drugs and Supplies)",
                     help = "Goal: Address leading biomedical causes of maternal and neonatal death")
        col1_1, col1_2 = st.columns(2)

        with col1_1:
            mat_interventions = {
                "PPH bundle": "Identify and reduce postpartum hemorrhage (PPH), including obsteric drape for identifying PPH, and treatments to stop bleeding (uterine massage, oxytocic drugs, tranexamic acid, IV fluids, and genital-tract examination)",
                "IV iron infusion": "Reduce the probability of anemia, which increases risk of maternal complications.",
                "Magnesium sulfate (MgSO4)": "Reduce maternal deaths due to eclampsia.",
                "Antibiotics for maternal sepsis": "Reduce maternal deaths due to maternal sepsis.",
                "Oxytocin for prolonged labor": "Reduce prolonged labor."
            }

            selected_mat_interventions = st.multiselect("Select **maternal interventions** to apply:", options=list(mat_interventions.keys()),
                                                    help="You can select multiple interventions.")

            if selected_mat_interventions:
                for intervention in selected_mat_interventions:
                    st.markdown(f"<h3 style='font-size:20px;'>{intervention}</h3>", unsafe_allow_html=True)

                    if intervention == "PPH bundle":
                        i_flags['flag_pph_bundle'] = 1
                        i_S["pph_bundle"] = st.slider("% mothers with PPH in L4/5 supplied",
                                                     min_value=round(slider_params['S_pph_bundle_slider'][3] * 100), max_value=100,
                                                     step=1, value=100, format="%d%%",
                                                     help=mat_interventions[intervention])
                        i_S["pph_bundle"] /= 100

                    elif intervention == "IV iron infusion":
                        i_flags['flag_iv_iron'] = 1
                        i_S["iv_iron"] = st.slider("% mothers with severe anemia supplied",
                                                   min_value=round(slider_params['S_iv_iron_slider'] * 100), max_value=100, step=1,
                                                   value=100, format="%d%%",
                                                   help=mat_interventions[intervention])
                        i_S["iv_iron"] /= 100

                    elif intervention == "Magnesium sulfate (MgSO4)":
                        i_flags['flag_MgSO4'] = 1
                        i_S["MgSO4"] = st.slider("% mothers with eclampsia in L4/5 supplied",
                                                 min_value=round(slider_params['S_MgSO4_slider'][3] * 100), max_value=100, step=1,
                                                 value=100, format="%d%%",
                                                 help=mat_interventions[intervention])
                        i_S["MgSO4"] /= 100

                    elif intervention == "Antibiotics for maternal sepsis":
                        i_flags['flag_antibiotics'] = 1
                        i_S["antibiotics"] = st.slider("% mothers with sepsis in L4/5 supplied",
                                                       min_value=round(slider_params['S_antibiotics_slider'][3] * 100), max_value=100,
                                                       step=1, value=100, format="%d%%",
                                                       help=mat_interventions[intervention])
                        i_S["antibiotics"] /= 100

                    elif intervention == "Oxytocin for prolonged labor":
                        i_flags['flag_oxytocin'] = 1
                        i_S["oxytocin"] = st.slider("% prolonged labor in L4/5 supplied",
                                                    min_value=round(slider_params['S_oxytocin_slider'][3] * 100), max_value=100, step=1,
                                                    value=100, format="%d%%",
                                                    help=mat_interventions[intervention])
                        i_S["oxytocin"] /= 100


                    st.markdown("---")  # Separator for each intervention

        with col1_2:
             neo_interventions = {
                 "Preterm complication treatments": "Include maternal corticosteroids for reducing RDS and IVH; antibiotics for reducing neonatal sepsis and NEC.",
             }

    with col2:
        st.subheader(":stethoscope: Diagnosis interventions (Sensors and Monitoring)",
                     help = "Goal: Increase diagnosis of high-risk pregnancies and complications")
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.text('Apply interventions')
            intus = st.toggle('AI portable ultrasound (AI-US)',
                               help = "It helps improve accuracy of gestational age estimation and risk stratification")

        with col2_2:
            st.text('Adjust parameters')
            if intus:
                i_flags['flag_us'] = 1
                i_E["sens_us"] = st.slider('Sensitivity of AI-US', min_value=0.00, max_value=1.00, step=0.05, value=0.95,
                                   help = "Value = 0.95 means AI-US can detect 95% of high-risk pregnancies")
                i_E["spec_us"] = st.slider('Specificity of AI-US', min_value=0.00, max_value=1.00, step=0.05, value=0.95,
                                     help = "Value = 0.95 means AI-US can detect 95% of low-risk pregnancies")
                i_S["US"] = 1
            else:
                i_flags['flag_us'] = 0
                i_S["US"] = 0

        st.markdown("---")
        intintrasensors = st.toggle('Intrapartum sensors',
                                    help="It helps predict pre-labor complications during delivery \n\n"
                                         "Sum of traditional monitoring and AI sensor coverage should be not larger than 100%")

        if intintrasensors:
            i_flags['flag_intrasensor'] = 1
            AI_sensor = st.checkbox('Apply AI algorithms')
            i_flags['flag_sensor_ai'] = 1 if AI_sensor else 0
        else:
            i_flags['flag_intrasensor'] = 0
            AI_sensor = 0
            i_flags['flag_sensor_ai'] = 0

        if AI_sensor:
            col2_3, col2_4 = st.columns(2)
            with col2_3:
                pass
            with col2_4:
                i_E["sens_sensor"] = st.slider('Sensitivity of AI-Sensor', min_value=0.00, max_value=1.00, step=0.05, value=0.95,
                                           help="Value = 0.95 means AI-sensor can detect 95% of pre-labor complications")
                i_E["spec_sensor"] = st.slider('Specificity of AI-Sensor', min_value=0.00, max_value=1.00, step=0.05, value=0.95,
                                           help="Value = 0.95 means AI-US can detect 95% of pre-labor complications")


# Display selected intervention type
# Initialize session state to manage navigation
if 'intervention_selection' not in st.session_state:
    st.session_state.intervention_selection = None
if 'hss_mode' not in st.session_state:
    st.session_state.hss_mode = None
if 'scenario_selected' not in st.session_state:
    st.session_state.scenario_selected = None
if 'model_finished' not in st.session_state:
    st.session_state.model_finished = False
if 'selected_outcomes' not in st.session_state:
    st.session_state.selected_outcomes = []
if 'b_df_multiple' not in st.session_state:
    st.session_state.b_df_multiple = None
if 'compare_two_interventions' not in st.session_state:
    st.session_state.compare_two_interventions = False
if 'dual_first_config' not in st.session_state:
    st.session_state.dual_first_config = None
if 'reference_label' not in st.session_state:
    st.session_state.reference_label = "Baseline"
if 'target_label' not in st.session_state:
    st.session_state.target_label = "Intervention"
if 'ab_base_df' not in st.session_state:
    st.session_state.ab_base_df = None
if 'ab_base_ind_outcomes' not in st.session_state:
    st.session_state.ab_base_ind_outcomes = None
if 'dl_table' not in st.session_state:
    st.session_state.dl_table = None




control_col, results_col = st.columns([0.36, 0.64], gap="large")

with control_col:
    with st.expander("⚙️ **Scenario Settings** (Click to expand/collapse)", expanded=True):
        # Leading Question
        if st.session_state.intervention_selection is None:
            st.title("Intervention Selection")
            st.subheader("1. Which types of interventions would you like to explore?")

            if st.button("All (Analyst Mode)"):
                st.session_state.intervention_selection = "Both"
            if st.button("All (Preset Mode)"):
                st.session_state.intervention_selection = "BothPreset"
            if st.button(":one: Health Systems Strengthening Interventions (Demand and Supply)"):
                st.session_state.intervention_selection = "HSS"
            if st.button(":two: Single Interventions (Treatment and Diagnosis)"):
                st.session_state.intervention_selection = "Single"
            if st.button(":three: MOMISH Interventions"):
                st.session_state.intervention_selection = "PROMPTS"


        if st.session_state.intervention_selection == "HSS":
            st.button("🔙 Back to Intervention Options", on_click=go_back_to_main)

            # Choose between scenarios or manual customization
            # if st.session_state.hss_mode is None:
            st.subheader("2. Choose how you want to explore HSS interventions:")

            if st.button("📊 Select Pre-set Scenarios"):
                st.session_state.hss_mode = "Scenarios"

            if st.button("🎛️ Customize Manually"):
                st.session_state.hss_mode = "Manual"

            # Scenario Selection Mode
            if st.session_state.hss_mode == "Scenarios":
                st.button("🔙 Back to HSS Options", on_click=go_back_to_hss)
                st.subheader("3. Select a scenario:")
                col1, col2 = st.columns(2)
                with col1:
                    demand_scenario = st.selectbox("**3.1 Choose a pre-set demand scenario**", ["Conservative", "Moderate", "Aggressive"])
                    supply_scenario = st.selectbox("**3.2 Choose a pre-set supply scenario**", ["Match Demand", "Cannot Meet Demand"])
                with col2:
                    pass
                render_hss(preset_demand_scenario=demand_scenario, preset_supply_scenario= supply_scenario)
                st.session_state.scenario_selected = True

            # --- Manual Customization Mode ---
            if st.session_state.hss_mode == "Manual":
                st.button("🔙 Back to HSS Options", on_click=go_back_to_hss)
                render_hss(preset_demand_scenario=None, preset_supply_scenario=None)
                st.session_state.scenario_selected = True

        elif st.session_state.intervention_selection == "Single":
            st.button("🔙 Back to Intervention Options", on_click=go_back_to_main)
            render_single()
            st.session_state.scenario_selected = True

        elif st.session_state.intervention_selection == "BothPreset":
            st.button("🔙 Back to Intervention Options", on_click=go_back_to_main)
            st.caption(
                "Preset mode: choose enabled interventions and strategy levels only. "
                "Parameters use predefined values (no sliders)."
            )
            render_hss_preset()
            st.markdown("---")
            render_single_preset()
            st.markdown("---")
            render_prompts_preset()
            st.session_state.scenario_selected = True

        elif st.session_state.intervention_selection == "Both":
            st.button("🔙 Back to Intervention Options", on_click=go_back_to_main)
            st.subheader("2. Choose how you want to explore HSS interventions (with Single + MOMISH below):")

            if st.button("📊 Select Pre-set Scenarios", key="both_btn_scenarios"):
                st.session_state.hss_mode = "Scenarios"

            if st.button("🎛️ Customize Manually", key="both_btn_manual"):
                st.session_state.hss_mode = "Manual"

            if st.session_state.hss_mode == "Scenarios":
                st.button("🔙 Back to HSS Options", on_click=go_back_to_hss, key="both_back_hss_scenarios")
                st.subheader("3. Select a scenario:")
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    demand_scenario_both = st.selectbox(
                        "**3.1 Choose a pre-set demand scenario**",
                        ["Conservative", "Moderate", "Aggressive"],
                        key="both_demand_preset",
                    )
                    supply_scenario_both = st.selectbox(
                        "**3.2 Choose a pre-set supply scenario**",
                        ["Match Demand", "Cannot Meet Demand"],
                        key="both_supply_preset",
                    )
                with col_b2:
                    pass
                render_hss(
                    preset_demand_scenario=demand_scenario_both,
                    preset_supply_scenario=supply_scenario_both,
                )
                st.markdown("---")
                render_single()
                st.markdown("---")
                st.subheader("MOMISH interventions")
                render_prompts()
                st.session_state.scenario_selected = True

            elif st.session_state.hss_mode == "Manual":
                st.button("🔙 Back to HSS Options", on_click=go_back_to_hss, key="both_back_hss_manual")
                render_hss(preset_demand_scenario=None, preset_supply_scenario=None)
                st.markdown("---")
                render_single()
                st.markdown("---")
                st.subheader("MOMISH interventions")
                render_prompts()
                st.session_state.scenario_selected = True

        elif st.session_state.intervention_selection == "PROMPTS":
            st.button("🔙 Back to Intervention Options", on_click=go_back_to_main)
            render_prompts()
            st.session_state.scenario_selected = True

    with (st.expander("⚙️ **Model Settings** (Click to expand/collapse)", expanded=True)):
        if st.session_state.scenario_selected == True:
            ##leading question on length of implementation phase, maintantance phase, multiple scenarios?
            st.subheader("How would you like to run the model?")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                MODEL["imple_time"] = st.slider("The length of implementation phase (years)?",
                                                min_value=3, max_value=6, step=1, value=3,
                                                help='Implementation phase will stop at the years you choose \n\n and continue the maintainance phase')

                MODEL["int_period"] = MODEL["imple_time"] * 12

            with col2:
                MODEL["main_time"] = st.slider("The length of maintenance phase (years)?",
                                               min_value=0, max_value=3, step=1, value=0,
                                               help='The model will simulate until maintenance phase finish')

                MODEL["n_months"] = MODEL["main_time"] * 12 + MODEL["int_period"]

            with col3:
                MODEL["multiple_run"] = st.checkbox("Run multiple scenarios?",
                                                    help="If checked, the model will run multiple scenarios with different random seeds")

            with col4:
                if MODEL["multiple_run"]:
                    MODEL["n_runs"] = st.number_input("Number of runs", min_value=1, max_value=300, step=1, value=1, placeholder="Type a number")

                    # st.slider("Number of runs", min_value=1, max_value=300, step=1, value=10,
                    #                             help="Number of simulation runs to average over")

            st.markdown("---")
            previous_compare_two_interventions = st.session_state.compare_two_interventions
            compare_two_interventions_selected = st.checkbox(
                "Compare two interventions directly (A vs B)",
                value=previous_compare_two_interventions,
                help="First click Run to capture Intervention A. Then adjust settings and click Run again for Intervention B."
            )
            if compare_two_interventions_selected != previous_compare_two_interventions:
                clear_model_runtime_state(clear_comparison=True)
            st.session_state.compare_two_interventions = compare_two_interventions_selected
            if st.session_state.compare_two_interventions:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.session_state.reference_label = st.text_input(
                        "Intervention A label",
                        value=st.session_state.reference_label if st.session_state.reference_label != "Baseline" else "Intervention A"
                    )
                with col_b:
                    st.session_state.target_label = st.text_input(
                        "Intervention B label",
                        value=st.session_state.target_label if st.session_state.target_label != "Intervention" else "Intervention B"
                    )

                if st.session_state.dual_first_config is None:
                    st.info("A/B mode: first click Run to capture Intervention A.")
                else:
                    st.success("Intervention A captured. Adjust settings and click Run for Intervention B.")

        if "b_df" not in st.session_state or "i_df" not in st.session_state or "n_months" not in st.session_state or "i_param" not in st.session_state or "n_runs" not in st.session_state or "int_period" not in st.session_state:
            st.session_state.b_df = None
            st.session_state.i_df = None
            st.session_state.n_months = None
            st.session_state.int_period = None
            st.session_state.i_param = None
            st.session_state.n_runs = None
            st.session_state.model_finished = False

        with st.form('Test'):
            col1, col2 = st.columns(2)

            with col1:
                submitted = st.form_submit_button("🚀 Run Model",
                                                  help="Click this button to run the model with the selected settings")

            with col2:
                clear = st.form_submit_button(
                    "🧹 Reset Model / Clear Cache",
                    help="Clear cached results and captured A/B comparison state without changing the selected intervention settings.",
                )

            if clear:
                clear_model_runtime_state(clear_comparison=True, clear_table=True)
                st.success("Model cache and captured A/B comparison state have been reset. Current intervention settings were kept.")

            if submitted:
                if not st.session_state.compare_two_interventions:
                    st.session_state.dual_first_config = None
                st.session_state.model_finished = False
                st.session_state.selected_outcomes = []

                current_config = {
                                "multiple_run": MODEL["multiple_run"],
                                "n_runs": MODEL["n_runs"],
                                "n_months": MODEL["n_months"],
                                "compare_two_interventions": st.session_state.compare_two_interventions
                            }

                if st.session_state.get("baseline_config") != current_config:
                                st.session_state.b_df = None
                                st.session_state.b_df_multiple = None
                                st.session_state.b_ind_outcomes = None
                                st.session_state.baseline_config = current_config

                # Display progress status
                status = st.empty()  # Placeholder for dynamic status messages
                progress_bar = st.progress(0)  # Initialize progress bar

                status.text("⏳ Running Model...")

                # Time tracking variables
                start_time = time.time()  # Record start time
                avg_time_per_run = None  # Will store estimated time per run

                # Parameter debug report snapshots (i_param at two points). In
                # multiple-run mode only the first run's snapshots are kept.
                debug_loader_param = None
                debug_final_param = None

                ### MODEL PARAMETERS ###
                n_months = MODEL["n_months"]
                int_period = MODEL["int_period"]
                compare_two_interventions = st.session_state.compare_two_interventions

                if compare_two_interventions and st.session_state.dual_first_config is None:
                    st.session_state.dual_first_config = {
                        "flags": copy.deepcopy(i_flags),
                        "E": copy.deepcopy(i_E),
                        "S": copy.deepcopy(i_S),
                        "HSS": copy.deepcopy(i_HSS),
                    }
                    st.session_state.ab_base_df = None
                    st.session_state.ab_base_ind_outcomes = None
                    st.session_state.model_finished = False
                    st.info("Intervention A has been captured. Please adjust settings and click Run again to execute Intervention B.")
                    st.stop()

                if not MODEL["multiple_run"]:  # SINGLE RUN MODE
                    num_seeds = n_months
                    base_seeds = make_shifted_run_seeds(1, num_seeds)[0]
                    base_seed = int(base_seeds[0])

                    b_param = get_parameters(rng=np.random.default_rng(base_seed), county=selected_county)
                    b_param = calculate_derived_parameters(b_param)

                    i_param = get_parameters(rng=np.random.default_rng(base_seed), county=selected_county)
                    i_param = calculate_derived_parameters(i_param)
                    if ENABLE_PARAM_DEBUG_REPORT:
                        debug_loader_param = copy.deepcopy(i_param)
                    # base_seed = np.random.default_rng().integers(low=0, high=1e6, size=1)[0]
                    # rng_param = np.random.default_rng(base_seed)

                    # b_param = get_parameters(rng = rng_param)
                    # b_param = calculate_derived_parameters(b_param)
                    b_flags = reset_flags()
                    b_HSS = reset_HSS(slider_params)
                    b_S = reset_S(slider_params)
                    b_E = reset_E()
                    if compare_two_interventions and st.session_state.dual_first_config is not None:
                        b_flags = copy.deepcopy(st.session_state.dual_first_config["flags"])
                        b_E = copy.deepcopy(st.session_state.dual_first_config["E"])
                        b_S = copy.deepcopy(st.session_state.dual_first_config["S"])
                        b_HSS = copy.deepcopy(st.session_state.dual_first_config["HSS"])

                    b_param.update({"E": b_E, "S": b_S, "HSS": b_HSS})
                    if compare_two_interventions:
                        sync_param_momish_from_hss(b_param, b_HSS, b_flags)
                    i_param.update({"E": i_E, "S": i_S, "HSS": i_HSS})
                    sync_param_momish_from_hss(i_param, i_HSS, i_flags)
                    if ENABLE_PARAM_DEBUG_REPORT:
                        debug_final_param = copy.deepcopy(i_param)

                    # rng_clone = np.random.default_rng(base_seed)
                    # i_param = get_parameters(rng = rng_clone)
                    # i_param = calculate_derived_parameters(i_param)
                    # i_param.update({"E": i_E, "S": i_S, "HSS": i_HSS})

                    # Run baseline model only if not already stored
                    # Always run baseline with the same base_seeds as intervention below.
                    # If we skipped this when b_df was cached, intervention would use newly drawn seeds
                    # while baseline stayed on old seeds (misaligned comparisons — e.g. MOMISH-only runs).
                    status.text("⏳ Running Baseline Model...")
                    b_df, b_ind_outcomes, _ = run_model_dash(b_param, b_flags, n_months, int_period, base_seed=base_seeds)
                    b_ind_outcomes["Run"] = 1
                    b_ind_outcomes["Scenario"] = st.session_state.reference_label if compare_two_interventions else "Baseline"
                    st.session_state.b_df = b_df
                    st.session_state.b_ind_outcomes = b_ind_outcomes
                    status.text("✅ Baseline Model Completed!")

                    # Run intervention model
                    status.text("⏳ Running Comparison Model...")
                    # rng_model = np.random.default_rng(base_seed)
                    i_df, i_ind_outcomes, _ = run_model_dash(i_param, i_flags, n_months, int_period, base_seed = base_seeds)
                    #i_df, i_ind_outcomes, _ = run_model_dash(i_param, i_flags, n_months, int_period, rng = rng_model)
                    i_ind_outcomes["Run"] = 1
                    i_ind_outcomes["Scenario"] = st.session_state.target_label if compare_two_interventions else "Intervention"

                    # In A/B mode, also run plain baseline once for column comparison
                    if compare_two_interventions:
                        status.text("⏳ Running Plain Baseline for A/B column comparison...")
                        base_param = get_parameters(rng=np.random.default_rng(base_seed), county=selected_county)
                        base_param = calculate_derived_parameters(base_param)
                        base_flags = reset_flags()
                        base_HSS = reset_HSS(slider_params)
                        base_S = reset_S(slider_params)
                        base_E = reset_E()
                        base_param.update({"E": base_E, "S": base_S, "HSS": base_HSS})
                        ab_base_df, ab_base_ind_outcomes, _ = run_model_dash(base_param, base_flags, n_months, int_period, base_seed=base_seeds)
                        ab_base_ind_outcomes["Run"] = 1
                        ab_base_ind_outcomes["Scenario"] = "Baseline"
                        st.session_state.ab_base_df = ab_base_df
                        st.session_state.ab_base_ind_outcomes = ab_base_ind_outcomes

                    # Retrieve cached baseline results
                    b_df = st.session_state.b_df
                    b_ind_outcomes = st.session_state.b_ind_outcomes
                    st.session_state.b_param = b_param
                    st.session_state.i_param = i_param

                    # Update progress bar to 100%
                    progress_bar.progress(100)

                else:  # MULTIPLE RUNS MODE
                    i_df, b_df, i_ind_outcomes, b_ind_outcomes = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                    total_runs = MODEL["n_runs"]
                    run_seeds_matrix = make_shifted_run_seeds(total_runs, n_months)

                    # --- BASELINE RUNS ---
                    # seeds = np.random.default_rng(2025).integers(low=0, high=1e6, size=total_runs * n_months)

                    # Always rerun the baseline on every "Run Model" click (single-run mode
                    # already does this unconditionally above) so that changing settings that
                    # aren't tracked in current_config -- e.g. the selected county -- can never
                    # leave a stale cached baseline in place.
                    status.text("⏳ Running Reference Model for Multiple Runs...")

                    temp_b_df = []  # Store results in list before concatenating (better performance)
                    temp_b_ind_outcomes = []

                    #for i in range(total_runs):
                    for run_index in range(total_runs):
                        iter_start_time = time.time()

                        # Reset flags and initialize parameters for each run
                        monthly_seeds_for_this_run = run_seeds_matrix[run_index]
                        # Use the VERY FIRST seed of this run's sequence to generate parameters
                        param_rng = np.random.default_rng(monthly_seeds_for_this_run[0])
                        b_param = get_parameters(rng=param_rng, county=selected_county)
                        b_param = calculate_derived_parameters(b_param)

                        b_flags, b_HSS, b_S, b_E = reset_flags(), reset_HSS(slider_params), reset_S(slider_params), reset_E()
                        if compare_two_interventions and st.session_state.dual_first_config is not None:
                            b_flags = copy.deepcopy(st.session_state.dual_first_config["flags"])
                            b_E = copy.deepcopy(st.session_state.dual_first_config["E"])
                            b_S = copy.deepcopy(st.session_state.dual_first_config["S"])
                            b_HSS = copy.deepcopy(st.session_state.dual_first_config["HSS"])
                        b_param.update({"E": b_E, "S": b_S, "HSS": b_HSS})
                        if compare_two_interventions:
                            sync_param_momish_from_hss(b_param, b_HSS, b_flags)

                        # Pass the ARRAY of monthly seeds to run_model_dash
                        b_df_i, b_ind_outcomes_i, _ = run_model_dash(b_param, b_flags, n_months, int_period, base_seed=monthly_seeds_for_this_run)

                        b_df_i["Run"] = run_index + 1
                        b_ind_outcomes_i["Run"] = run_index + 1
                        b_ind_outcomes_i["Scenario"] = st.session_state.reference_label if compare_two_interventions else "Baseline"
                        temp_b_df.append(b_df_i)
                        temp_b_ind_outcomes.append(b_ind_outcomes_i)

                        # rng_param = np.random.default_rng(base_seed)
                        # b_param = get_parameters(rng = rng_param)
                        # b_param = calculate_derived_parameters(b_param)
                        # #st.text(b_param)
                        # b_flags, b_HSS, b_S, b_E = reset_flags(), reset_HSS(slider_params), reset_S(slider_params), reset_E()
                        # b_param.update({"E": b_E, "S": b_S, "HSS": b_HSS})

                        # # Run baseline model only once per iteration
                        # rng_model = np.random.default_rng(base_seed)
                        # b_df_i, b_ind_outcomes_i, _ = run_model_dash(b_param, b_flags, n_months, int_period, base_seed = base_seeds)
                        # #b_df_i, b_ind_outcomes_i, _ = run_model_dash(b_param, b_flags, n_months, int_period,
                        #                                          rng=None)
                        # Time tracking & progress update
                        iter_time_taken = time.time() - iter_start_time
                        avg_time_per_run = iter_time_taken if avg_time_per_run is None else (
                                                                                                        avg_time_per_run * run_index + iter_time_taken) / (
                                                                                                        run_index + 1)
                        remaining_time = avg_time_per_run * (total_runs - (run_index + 1))
                        progress_bar.progress((run_index + 1) / total_runs)
                        status.text(f"⏳ Running Reference Model... {run_index + 1}/{total_runs} runs completed. "
                                    f"Estimated time left: {remaining_time / 60:.1f} min.")
                        # st.text(f"CPU usage: {psutil.cpu_percent()}%")
                        # usage_per_core = psutil.cpu_percent(percpu=True)
                        # for i, usage in enumerate(usage_per_core):
                        #     st.text(f"Core {i}: {usage}%")
                        #
                        # def print_resource_usage(interval=1, repeat=10):
                        #     process = psutil.Process(os.getpid())
                        #
                        #     for i in range(repeat):
                        #         cpu = psutil.cpu_percent(interval=interval)
                        #         mem_info = process.memory_info()
                        #         mem_mb = mem_info.rss / (1024 ** 2)  # Convert bytes to MB
                        #
                        #         st.text(f"[{i + 1}] CPU Usage: {cpu:.1f}% | Memory Usage: {mem_mb:.2f} MB")
                        #
                        # print_resource_usage()

                    # Store final baseline results in session state
                    st.session_state.b_df_multiple = pd.concat(temp_b_df, ignore_index=True)
                    st.session_state.b_ind_outcomes = pd.concat(temp_b_ind_outcomes, ignore_index=True)

                    status.text("✅ Reference Model Completed!")

                    # Retrieve baseline results just computed above
                    b_df = st.session_state.b_df_multiple
                    b_ind_outcomes = st.session_state.b_ind_outcomes

                    # Run Intervention Model for Each Run
                    temp_i_df = []
                    temp_i_ind_outcomes = []

                    #for i in range(total_runs):
                    for run_index in range(total_runs):
                        iter_start_time = time.time()

                        monthly_seeds_for_this_run = run_seeds_matrix[run_index]

                        # Use the exact same seed to generate identical starting parameters
                        param_rng = np.random.default_rng(monthly_seeds_for_this_run[0])
                        i_param = get_parameters(rng=param_rng, county=selected_county)
                        i_param = calculate_derived_parameters(i_param)
                        if ENABLE_PARAM_DEBUG_REPORT and run_index == 0:
                            debug_loader_param = copy.deepcopy(i_param)
                        i_param.update({"E": i_E, "S": i_S, "HSS": i_HSS})
                        sync_param_momish_from_hss(i_param, i_HSS, i_flags)
                        if ENABLE_PARAM_DEBUG_REPORT and run_index == 0:
                            debug_final_param = copy.deepcopy(i_param)

                        # Pass the exact same ARRAY of monthly seeds
                        i_df_i, i_ind_outcomes_i, _ = run_model_dash(i_param, i_flags, n_months, int_period, base_seed=monthly_seeds_for_this_run)

                        i_df_i["Run"] = run_index + 1
                        i_ind_outcomes_i["Run"] = run_index + 1
                        i_ind_outcomes_i["Scenario"] = st.session_state.target_label if compare_two_interventions else "Intervention"
                        temp_i_df.append(i_df_i)
                        temp_i_ind_outcomes.append(i_ind_outcomes_i)
                        # Reset flags and initialize parameters for each run
                        # rng_param = np.random.default_rng(base_seed)
                        # i_param = get_parameters(rng = rng_param)
                        # i_param = calculate_derived_parameters(i_param)
                        # #st.text(i_param)
                        # i_param.update({"E": i_E, "S": i_S, "HSS": i_HSS})

                        # # Run intervention model
                        # rng_model = np.random.default_rng(base_seed)
                        # i_df_i, i_ind_outcomes_i, _ = run_model_dash(i_param, i_flags, n_months, int_period, base_seed = base_seed)
                        # i_df_i["Run"] = run_index + 1
                        # i_ind_outcomes_i["Run"] = run_index + 1
                        # i_ind_outcomes_i["Scenario"] = "Intervention"
                        # temp_i_df.append(i_df_i)
                        # temp_i_ind_outcomes.append(i_ind_outcomes_i)

                        # Time tracking & progress update
                        iter_time_taken = time.time() - iter_start_time
                        avg_time_per_run = iter_time_taken if avg_time_per_run is None else (
                                                                                                        avg_time_per_run * run_index + iter_time_taken) / (
                                                                                                        run_index + 1)
                        remaining_time = avg_time_per_run * (total_runs - (run_index + 1))
                        progress_bar.progress((run_index + 1) / total_runs)
                        status.text(f"⏳ Running Comparison Model... {run_index + 1}/{total_runs} runs completed. "
                                    f"Estimated time left: {remaining_time / 60:.1f} min.")
                        # st.text(f"CPU usage: {psutil.cpu_percent()}%")
                        # usage_per_core = psutil.cpu_percent(percpu=True)
                        # for i, usage in enumerate(usage_per_core):
                        #     st.text(f"Core {i}: {usage}%")

                    # Store intervention results
                    i_df = pd.concat(temp_i_df, ignore_index=True)
                    i_ind_outcomes = pd.concat(temp_i_ind_outcomes, ignore_index=True)

                    # In A/B mode, also run plain baseline for multiple runs
                    if compare_two_interventions:
                        status.text("⏳ Running Plain Baseline for A/B column comparison...")
                        temp_ab_base_df = []
                        temp_ab_base_ind_outcomes = []
                        for run_index in range(total_runs):
                            monthly_seeds_for_this_run = run_seeds_matrix[run_index]
                            param_rng = np.random.default_rng(monthly_seeds_for_this_run[0])
                            base_param = get_parameters(rng=param_rng, county=selected_county)
                            base_param = calculate_derived_parameters(base_param)
                            base_flags = reset_flags()
                            base_HSS = reset_HSS(slider_params)
                            base_S = reset_S(slider_params)
                            base_E = reset_E()
                            base_param.update({"E": base_E, "S": base_S, "HSS": base_HSS})
                            ab_base_df_i, ab_base_ind_outcomes_i, _ = run_model_dash(base_param, base_flags, n_months, int_period, base_seed=monthly_seeds_for_this_run)
                            ab_base_df_i["Run"] = run_index + 1
                            ab_base_ind_outcomes_i["Run"] = run_index + 1
                            ab_base_ind_outcomes_i["Scenario"] = "Baseline"
                            temp_ab_base_df.append(ab_base_df_i)
                            temp_ab_base_ind_outcomes.append(ab_base_ind_outcomes_i)

                        st.session_state.ab_base_df = pd.concat(temp_ab_base_df, ignore_index=True)
                        st.session_state.ab_base_ind_outcomes = pd.concat(temp_ab_base_ind_outcomes, ignore_index=True)

                    status.text("✅ Model Run Completed!")

                # Total execution time
                total_time = time.time() - start_time
                total_minutes = total_time / 60

                # Store results in session state
                st.session_state.b_df = b_df
                st.session_state.i_df = i_df
                st.session_state.b_ind_outcomes = b_ind_outcomes
                st.session_state.i_ind_outcomes = i_ind_outcomes
                st.session_state.n_months = n_months
                st.session_state.int_period = int_period
                st.session_state.b_param = b_param
                st.session_state.i_param = i_param
                st.session_state.n_runs = MODEL["n_runs"]
                st.session_state.model_finished = True

                if ENABLE_PARAM_DEBUG_REPORT and debug_loader_param is not None and debug_final_param is not None:
                    st.session_state.param_debug_report = build_parameter_debug_report(
                        debug_loader_param,
                        debug_final_param,
                        county=selected_county,
                        scenario=st.session_state.get("target_label", "Intervention"),
                        flags=i_flags,
                    )

                # Success message with total runtime
                st.success(f"🎉 Model execution completed! Total runtime: {total_minutes:.1f} minutes.")

with results_col:
    if  st.session_state.model_finished == True:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            download_option = st.checkbox("Download the individual outcomes")
            if download_option:
                if "b_ind_outcomes" in st.session_state and not st.session_state.b_ind_outcomes.empty:
                    csv_baseline = st.session_state.b_ind_outcomes.to_csv(index=False)
                    baseline_filename = f"{st.session_state.reference_label.replace(' ', '_')}.csv"
                    st.download_button(label=f"📥 Download {baseline_filename}", data=csv_baseline, file_name=baseline_filename,
                                       mime="text/csv")
                else:
                    st.warning("⚠️ Reference scenario data is not available. Please run the model first.")

                scenario_name = st.text_input("**Enter Intervention Scenario Name:**", placeholder="e.g., Scenario_1")
                st.session_state.scenario_name = scenario_name
                # Ensure a scenario name is entered
                if scenario_name:
                    if "i_ind_outcomes" in st.session_state and not st.session_state.i_ind_outcomes.empty:
                        file_name = f"{scenario_name}.csv"
                        st.session_state.i_ind_outcomes["Scenario"] = scenario_name
                        csv_intervention = st.session_state.i_ind_outcomes.to_csv(index=False)
                        st.download_button(label=f"📥 Download {file_name}", data=csv_intervention, file_name=file_name,
                                           mime="text/csv")
                    else:
                        st.warning("⚠️ Intervention data is not available. Please run the model first.")
                else:
                    st.warning("⚠️ Please enter a scenario name before downloading the intervention data.")

            if ENABLE_PARAM_DEBUG_REPORT and st.session_state.get("param_debug_report"):
                st.download_button(
                    label="🛠️ Download Parameter Debug Report",
                    data=st.session_state.param_debug_report,
                    file_name="param_debug_report.txt",
                    mime="text/plain",
                )


    if "b_df" in st.session_state and "i_df" in st.session_state and st.session_state.model_finished:
        st.subheader("Select outcomes for visualization")

        # Initialize session state for storing selected outcomes
        if "selected_outcomes" not in st.session_state:
            st.session_state.selected_outcomes = []

        col1, col2 = st.columns([2, 1])

        with col1:
            col1_1, col1_2 = st.columns(2)

            with col1_1:
                category_options = ('System Features',
                                    'Process Indicators',
                                    'Implementation Outcomes',
                                    'Maternal Outcomes',
                                    'Neonatal Outcomes')

                plot_category = st.selectbox(
                    label="(1) Select category of outcomes:",
                    options=category_options,
                    key="plot_category"
                )

            with col1_2:
                # Define outcome options for each category
                outcomes_dict = {
                    'System Features': (
                        "Facility capacity ratio",
                        "Labor force ratio",
                        "Equipment inventory ratio",
                    ),
                    'Process Indicators': (
                        "Distribution of live births",
                        "High-risk pregnancies",
                        "ANC rate",
                        "C-section rate",
                        "Normal referral",
                        "Emergency transfer"
                    ),
                    'Implementation Outcomes': (
                        "Cost Effectiveness",
                        "DALYs",
                        "DALYs averted"
                    ),
                    'Maternal Outcomes': (
                        "Maternal complication rate",
                        "Severe maternal outcomes",
                        "Maternal mortality rate"
                    ),
                    'Neonatal Outcomes': (
                        "Preterm rate",
                        "Neonatal complication rate",
                        "Neonatal mortality rate",
                        "Stillbirth rate"
                    )
                }

                plot_options = outcomes_dict.get(plot_category, [])

                selected_plot = st.selectbox(
                    label="(2) Select outcome of interest:",
                    options=plot_options,
                    key="selected_plot"
                )

        with col2:
            st.markdown("### Actions")
            # Button to add the selected outcome to visualization list
            if st.button("➕ Add Outcome"):
                if selected_plot and (selected_plot not in st.session_state.selected_outcomes):
                    st.session_state.selected_outcomes.append((plot_category, selected_plot))
                    st.rerun()  # Force UI to refresh to reflect added outcomes

        st.markdown("---")

        # Show selected outcomes
        if st.session_state.selected_outcomes:
            st.subheader("📌 Selected Outcomes")

            for i, (category, outcome) in enumerate(st.session_state.selected_outcomes):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{category} → {outcome}**")
                with col2:
                    if st.button(f"❌ Remove {i + 1}", key=f"remove_{i}"):
                        st.session_state.selected_outcomes.pop(i)
                        st.rerun()  # Refresh UI after removal

            st.markdown("---")

        else:
            st.info("No outcomes selected yet. Please add at least one outcome.")

    if st.session_state.b_df is not None and st.session_state.i_df is not None:
        # Use the stored dataframes instead of recalculating
        b_df = st.session_state.b_df
        i_df = st.session_state.i_df
        n_months = st.session_state.n_months
        int_period = st.session_state.int_period
        i_param = st.session_state.i_param
        n_runs = st.session_state.n_runs

        if st.session_state.selected_outcomes:
            for _, selected_plot in st.session_state.selected_outcomes:

                ##Functions##
                def prepare_indicator_df(df, indicator, n_months, n_runs, scenario):
                    indicator = np.concatenate(df[indicator].values).reshape(-1, 4)
                    df_indicator = pd.DataFrame(indicator, columns=['Home', 'L2/3', 'L4', 'L5'])
                    df_indicator['All'] = np.sum(df_indicator, axis=1)
                    df_indicator['Facilities'] = df_indicator['L2/3'] + df_indicator['L4'] + df_indicator['L5']
                    df_indicator['L4/5'] = df_indicator['L4'] + df_indicator['L5']
                    df_indicator = df_indicator.drop(columns=['L4', 'L5'])

                    # df_indicator['Month'] = np.tile(np.arange(1, n_months+1), n_runs)
                    actual_runs_in_df = len(df_indicator) // n_months
                    df_indicator['Month'] = np.tile(np.arange(1, n_months+1), actual_runs_in_df)

                    df_indicator['Run'] = np.repeat(np.arange(1, actual_runs_in_df + 1), n_months)
                    df_indicator['Scenario'] = scenario
                    #df_indicator = df_indicator.groupby(['Month', 'Scenario'], as_index=False).sum()
                    df_indicator = df_indicator.melt(id_vars=['Month', 'Run', 'Scenario'], var_name='Level', value_name='Counts')
                    return df_indicator

                def prepare_chart_data(b_df, i_df, numerator, dominator, n_months, n_runs, multiplier):
                    b_df_ind = prepare_indicator_df(b_df, numerator, n_months, n_runs, st.session_state.reference_label)
                    i_df_ind = prepare_indicator_df(i_df, numerator, n_months, n_runs, st.session_state.target_label)
                    df_ind = pd.concat([b_df_ind, i_df_ind], ignore_index=True)

                    b_df_lb = prepare_indicator_df(b_df, dominator, n_months, n_runs, st.session_state.reference_label)
                    i_df_lb = prepare_indicator_df(i_df, dominator, n_months, n_runs, st.session_state.target_label)
                    df_lb = pd.concat([b_df_lb, i_df_lb], ignore_index=True)

                    # In Both + A/B mode, add plain baseline as the 3rd scenario for bar/column comparisons
                    if (
                        st.session_state.get("compare_two_interventions", False)
                        and st.session_state.get("intervention_selection") in ("Both", "BothPreset")
                        and st.session_state.get("ab_base_df") is not None
                    ):
                        ab_base_df = st.session_state.ab_base_df
                        base_df_ind = prepare_indicator_df(ab_base_df, numerator, n_months, n_runs, "Baseline")
                        base_df_lb = prepare_indicator_df(ab_base_df, dominator, n_months, n_runs, "Baseline")
                        df_ind = pd.concat([df_ind, base_df_ind], ignore_index=True)
                        df_lb = pd.concat([df_lb, base_df_lb], ignore_index=True)

                    df = df_ind.merge(df_lb, on=['Month', 'Run', 'Scenario', 'Level'], suffixes=('_ind', '_lb'))
                    df.columns = ['Month', 'Run', 'Scenario', 'Level', 'Counts', 'Denominator']
                    df['Rate'] = df['Counts'] / df['Denominator'] * multiplier
                    return df

                def add_poisson_ci(df, multiplier, confidence=0.95):
                    alpha = 1 - confidence

                    # lower = 0 if df['Counts'] == 0 else stats.chi2.ppf(alpha / 2, 2 * df['Counts']) / 2
                    # upper = stats.chi2.ppf(1 - alpha / 2, 2 * (df['Counts'] + 1)) / 2
                    # Vectorized Poisson CI calculation
                    lower_bound = np.where(
                        df['Counts'] == 0,
                        0,
                        stats.chi2.ppf(alpha / 2, 2 * df['Counts']) / 2
                    )
                    upper_bound = stats.chi2.ppf(1 - alpha / 2, 2 * (df['Counts'] + 1)) / 2

                    # Apply rates per 1,000 or 100,000 births
                    # df['Lower_rate'] = lower / df['Denominator'] * multiplier if df['Denominator'] > 0 else 0
                    # df['Upper_rate'] = upper / df['Denominator'] * multiplier if df['Denominator'] > 0 else 0
                    df['Lower_rate'] = np.where(df['Denominator'] > 0, lower_bound / df['Denominator'] * multiplier, 0)
                    df['Upper_rate'] = np.where(df['Denominator'] > 0, upper_bound / df['Denominator'] * multiplier, 0)
                    return df

                def create_line_data(data, multiplier):
                    line_data = data[data['Scenario'] == st.session_state.target_label].copy()
                    # line_data = line_data.apply(add_poisson_ci, axis=1, multiplier=multiplier)
                    # line_data = line_data.apply(lambda x: round(x, 2) if x.name in ['Rate', 'Lower_rate', 'Upper_rate'] else x)

                    # Step 1: Add Poisson CI per run × month × level
                    line_data = add_poisson_ci(line_data, multiplier=multiplier)

                    # Step 2: Mean rate and mean bounds across runs
                    line_data = line_data.groupby(['Month', 'Level'], as_index=False).agg({
                        'Rate': 'mean',
                        'Lower_rate': 'mean',
                        'Upper_rate': 'mean'
                    })

                    # Step 3: Add back Scenario label and round
                    line_data['Scenario'] = st.session_state.target_label
                    line_data[['Rate', 'Lower_rate', 'Upper_rate']] = line_data[['Rate', 'Lower_rate', 'Upper_rate']].round(
                        2)
                    return line_data

                def create_bar_data(data, multiplier):
                    # bar_data = data[data['Month'] > n_months - 12]
                    # bar_data = bar_data.groupby(['Scenario', 'Level'], as_index=False).sum()
                    # bar_data['Rate'] = bar_data['Counts'] / bar_data['Denominator'] * multiplier
                    # bar_data = bar_data.apply(add_poisson_ci, axis=1, multiplier=multiplier)
                    # bar_data = bar_data.apply(lambda x: round(x, 2) if x.name in ['Rate', 'Lower_rate', 'Upper_rate'] else x)

                    # Keep only the last year of simulation
                    #bar_data = data[data['Month'] > n_months - 12].copy()
                    bar_data = data.copy()
                    bar_data = bar_data.groupby(['Scenario', 'Run', 'Level'], as_index=False).agg({
                        'Counts': 'sum',
                        'Denominator': 'sum',
                    })
                    bar_data['Rate'] = bar_data['Counts'] / bar_data['Denominator'] * multiplier

                    # Step 1: Add Poisson CI per run × month × level
                    bar_data = add_poisson_ci(bar_data, multiplier=multiplier)

                    # Step 2: Taking average across runs and scenarios
                    bar_data = bar_data.groupby(['Scenario', 'Level'], as_index=False).agg({
                        'Rate': 'mean',
                        'Lower_rate': 'mean',
                        'Upper_rate': 'mean'
                    })

                    # Step 3: Round for display
                    bar_data[['Rate', 'Lower_rate', 'Upper_rate']] = bar_data[['Rate', 'Lower_rate', 'Upper_rate']].round(2)
                    return bar_data

                def line_chart_ci(line_data, title, ytitle, ydomain):
                    chart = (
                            alt.Chart(line_data, title=title)
                            .mark_line()
                            .encode(
                                x=alt.X("Month:Q", axis=alt.Axis(
                                        title="Time since the start of intervention implementation (Months)",
                                        titleFontSize=16, labelFontSize=14,)
                                        ),
                                y=alt.Y("Rate:Q", axis=alt.Axis(
                                    title=ytitle,
                                    titleFontSize=16, labelFontSize=14),
                                        scale=alt.Scale(domain=ydomain)
                                        ),
                                color=alt.Color("Level", legend=alt.Legend(title="Level", titleFontSize=16, labelFontSize=14)),
                                tooltip=["Month:N", "Level:N", "Rate:Q", "Lower_rate:Q", "Upper_rate:Q"]
                            )
                            + alt.Chart(line_data)
                            .mark_area(opacity=0.2)
                            .encode(
                        x="Month:Q",
                        y=alt.Y("Lower_rate:Q", axis=alt.Axis(title=ytitle, titleFontSize=16, labelFontSize=14)),
                        y2="Upper_rate:Q",
                        color=alt.Color("Level", legend=None),  # Match color to Scenario
                        )
                    )

                    chart = chart.properties(width=700, height=400).interactive()
                    chart = chart.configure_title(
                        anchor='middle', fontSize=18
                    )
                    return chart

                def bar_chart_ci(bar_data, title, ytitle, ydomain):
                    num_facets = len(bar_data["Level"].unique())
                    scenario_values = bar_data["Scenario"].dropna().unique().tolist()
                    preferred_order = ["Baseline", st.session_state.reference_label, st.session_state.target_label]
                    scenario_domain = list(dict.fromkeys(s for s in preferred_order if s in scenario_values))
                    scenario_domain.extend(s for s in scenario_values if s not in scenario_domain)
                    scenario_color_map = {
                        "Baseline": "#1F3A93",
                        st.session_state.reference_label: "#4C78A8",
                        st.session_state.target_label: "#72B7B2",
                    }
                    scenario_colors = [scenario_color_map.get(s, "#54A24B") for s in scenario_domain]
                    layered_chart = (
                            alt.Chart(bar_data)
                            .mark_bar()
                            .encode(
                                x=alt.X("Scenario:N", axis=None),  # X-axis for Scenario
                                y=alt.Y("Rate:Q", axis=alt.Axis(title=ytitle, titleFontSize=16, labelFontSize=14), scale=alt.Scale(domain=ydomain)),
                                color=alt.Color(
                                    "Scenario:N",
                                    legend=alt.Legend(title="Scenario", titleFontSize=16, labelFontSize=14),
                                    scale=alt.Scale(domain=scenario_domain, range=scenario_colors),
                                ),  # Color by Scenario
                                tooltip=["Scenario:N", "Level:N", "Rate:Q", "Lower_rate:Q", "Upper_rate:Q"]
                            )
                            + alt.Chart(bar_data)
                            .mark_errorbar(color="black", ticks=True)  # Black color for error bars
                            .encode(
                        x=alt.X("Scenario:N"),  # Match bar chart X-axis
                        y=alt.Y("Lower_rate:Q", axis=alt.Axis(title=ytitle, titleFontSize=16, labelFontSize=14)),  # Lower bound of the CI
                        y2="Upper_rate:Q",  # Upper bound of the CI
                        )
                    ).properties(width=int(600 / num_facets), height=300)  # Set width and height for each column

                    chart = layered_chart.facet(
                        column=alt.Column(
                            "Level:N",
                            header=alt.Header(labelOrient="bottom", labelFontSize=14,
                                              title="Delivery Location", titleOrient="bottom", titleFontSize=16)
                        )
                    ).configure_facet(
                            spacing=10  # Adjust spacing between columns to avoid crowding
                    ).interactive().properties(
                        title=alt.TitleParams(text=title, anchor="middle", fontSize=18)
                    )

                    return chart

                def aggregate_deaths_by_cause(ind_outcomes):
                    """Mean maternal death count by cause across simulation runs."""
                    df = ind_outcomes.loc[
                        ind_outcomes["death_cause"] != "none", ["Run", "death_cause"]
                    ].copy()
                    per_run = (
                        df.groupby(["Run", "death_cause"], as_index=False)
                        .size()
                        .rename(columns={"size": "count"})
                    )
                    if per_run.empty:
                        return per_run
                    if per_run["Run"].nunique() <= 1:
                        return per_run.groupby("death_cause", as_index=False)["count"].sum()
                    return per_run.groupby("death_cause", as_index=False)["count"].mean()

                def death_by_cause_comparison_chart(b_ind_outcomes, i_ind_outcomes, ref_name, tgt_name):
                    """
                    Stacked bar: base = reference deaths by cause; overlay = change to target (count delta).
                    Labels show percent change (target vs reference).
                    """
                    cause_order = ["pph", "sepsis", "eclampsia", "ol", "aph", "other"]
                    cause_labels = {
                        "pph": "PPH",
                        "sepsis": "Sepsis",
                        "eclampsia": "Eclampsia",
                        "ol": "Obstructed labor",
                        "aph": "APH",
                        "other": "Other",
                    }

                    ref_counts = aggregate_deaths_by_cause(b_ind_outcomes)
                    tgt_counts = aggregate_deaths_by_cause(i_ind_outcomes)
                    if ref_counts.empty and tgt_counts.empty:
                        return None

                    merged = pd.DataFrame({"death_cause": cause_order})
                    merged = merged.merge(
                        ref_counts.rename(columns={"count": "count_ref"}),
                        on="death_cause",
                        how="left",
                    ).merge(
                        tgt_counts.rename(columns={"count": "count_tgt"}),
                        on="death_cause",
                        how="left",
                    )
                    merged[["count_ref", "count_tgt"]] = merged[["count_ref", "count_tgt"]].fillna(0)
                    merged["delta"] = merged["count_tgt"] - merged["count_ref"]
                    merged["pct_change"] = np.where(
                        merged["count_ref"] > 0,
                        merged["delta"] / merged["count_ref"] * 100,
                        np.where(merged["count_tgt"] > 0, np.inf, 0),
                    )
                    merged["cause_label"] = merged["death_cause"].map(cause_labels)

                    change_label = f"Change ({tgt_name} − {ref_name})"
                    base_layer = merged.assign(segment=ref_name, y=merged["count_ref"])
                    delta_layer = merged.assign(segment=change_label, y=merged["delta"])
                    stack_df = pd.concat(
                        [
                            base_layer[["cause_label", "death_cause", "segment", "y", "count_ref", "count_tgt", "delta", "pct_change"]],
                            delta_layer[["cause_label", "death_cause", "segment", "y", "count_ref", "count_tgt", "delta", "pct_change"]],
                        ],
                        ignore_index=True,
                    )
                    stack_df["segment_order"] = np.where(stack_df["segment"] == ref_name, 0, 1)
                    stack_df["bar_color"] = np.where(
                        stack_df["segment"] == ref_name,
                        ref_name,
                        np.where(stack_df["delta"] > 0, "Increase", "Decrease"),
                    )

                    bars = (
                        alt.Chart(stack_df)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "cause_label:N",
                                sort=[cause_labels[c] for c in cause_order],
                                title="Cause of death",
                            ),
                            y=alt.Y("y:Q", stack="zero", title="Maternal deaths (mean count)"),
                            color=alt.Color(
                                "bar_color:N",
                                title="Component",
                                scale=alt.Scale(
                                    domain=[ref_name, "Increase", "Decrease"],
                                    range=["#4C78A8", "#E45756", "#54A24B"],
                                ),
                            ),
                            order=alt.Order("segment_order:O", sort="ascending"),
                            tooltip=[
                                alt.Tooltip("cause_label:N", title="Cause"),
                                alt.Tooltip("count_ref:Q", format=".1f", title=ref_name),
                                alt.Tooltip("count_tgt:Q", format=".1f", title=tgt_name),
                                alt.Tooltip("delta:Q", format="+.1f", title="Absolute change"),
                                alt.Tooltip("pct_change:Q", format="+.1f", title="% change"),
                            ],
                        )
                    )

                    label_df = merged.copy()

                    def _pct_label(row):
                        if row["count_ref"] > 0:
                            return f"{row['pct_change']:+.0f}%"
                        if row["count_tgt"] > 0:
                            return "new"
                        return "0%"

                    label_df["pct_label"] = label_df.apply(_pct_label, axis=1)
                    label_df["y_top"] = label_df["count_tgt"]
                    label_df["trend"] = np.select(
                        [label_df["pct_change"] > 0, label_df["pct_change"] < 0],
                        ["Increase", "Decrease"],
                        default="No change",
                    )

                    labels = (
                        alt.Chart(label_df)
                        .mark_text(dy=-8, fontSize=12, fontWeight="bold")
                        .encode(
                            x=alt.X(
                                "cause_label:N",
                                sort=[cause_labels[c] for c in cause_order],
                            ),
                            y=alt.Y("y_top:Q"),
                            text="pct_label:N",
                            color=alt.Color(
                                "trend:N",
                                scale=alt.Scale(
                                    domain=["Increase", "Decrease", "No change"],
                                    range=["#C44E52", "#2CA02C", "#444444"],
                                ),
                                legend=None,
                            ),
                        )
                    )

                    title = f"Maternal deaths by cause: {ref_name} with change to {tgt_name}"
                    return (bars + labels).properties(
                        width=700,
                        height=420,
                        title=alt.TitleParams(text=title, anchor="middle", fontSize=16),
                    ).configure_axis(labelFontSize=12, titleFontSize=14)

                def line_chart_ci_matplotlib(line_data, title, ytitle, ydomain):
                    """
                    Create a line plot with confidence intervals using Matplotlib.
                    """
                    fig, ax = plt.subplots(figsize=(8, 5))  # Define figure size

                    # Define unique levels (categories)
                    levels = line_data["Level"].unique()
                    colors = sns.color_palette("tab10", len(levels))  # Use seaborn color palette

                    for i, level in enumerate(levels):
                        subset = line_data[line_data["Level"] == level]
                        ax.plot(subset["Month"], subset["Rate"], label=level, color=colors[i], linewidth=2)
                        ax.fill_between(subset["Month"], subset["Lower_rate"], subset["Upper_rate"],
                                        color=colors[i], alpha=0.2)  # Confidence interval shading

                    # Formatting
                    ax.set_xlabel("Time since the start of intervention implementation (Months)", fontsize=14)
                    ax.set_ylabel(ytitle, fontsize=14)
                    ax.set_ylim(ydomain)  # Set y-axis domain
                    ax.set_title(title, fontsize=16)
                    ax.legend(title="Level", fontsize=12)

                    plt.grid(True, linestyle="--", alpha=0.5)
                    plt.tight_layout()

                    return fig  # Return figure for saving

                def bar_chart_ci_matplotlib(bar_data, title, ytitle, ydomain):
                    """
                    Create a bar chart with error bars using Matplotlib.
                    """
                    fig, ax = plt.subplots(figsize=(8, 5))  # Define figure size

                    # Define unique scenarios
                    scenarios = bar_data["Scenario"].unique()
                    levels = bar_data["Level"].unique()
                    colors = sns.color_palette("tab10", len(scenarios))

                    bar_width = 0.35  # Adjust bar width

                    # Create bars for each scenario
                    for i, scenario in enumerate(scenarios):
                        subset = bar_data[bar_data["Scenario"] == scenario]
                        x_positions = np.arange(len(subset))
                        ax.bar(x_positions + i * bar_width, subset["Rate"], yerr=[subset["Rate"] - subset["Lower_rate"],
                                                                                  subset["Upper_rate"] - subset["Rate"]],
                               capsize=5, label=scenario, color=colors[i], width=bar_width, alpha=0.8)

                    # Formatting
                    ax.set_xticks(np.arange(len(levels)) + bar_width / 2)
                    ax.set_xticklabels(levels, fontsize=12)
                    ax.set_xlabel("Delivery Location", fontsize=14)
                    ax.set_ylabel(ytitle, fontsize=14)
                    ax.set_ylim(ydomain)
                    ax.set_title(title, fontsize=16)
                    ax.legend(title="Scenario", fontsize=12)

                    plt.grid(True, linestyle="--", alpha=0.5)
                    plt.tight_layout()

                    return fig  # Return figure for saving

                def Acum_DALY_df(indicator, dominator, multiplier):
                    # Prepare chart data
                    df_DALY = prepare_chart_data(b_df, i_df, indicator, dominator, n_months, n_runs, multiplier)

                    #filter out level = "All"
                    df_DALY = df_DALY[df_DALY['Level'] == 'All']

                    # Aggregate for Baseline and Intervention scenarios
                    df_DALY_base = df_DALY[df_DALY['Scenario'] == 'Baseline'].groupby(['Month'], as_index=False).sum()
                    df_DALY_int = df_DALY[df_DALY['Scenario'] == 'Intervention'].groupby(['Month'], as_index=False).sum()

                    # Calculate cumulative sums for Counts and LiveBirths
                    df_DALY_base['Cumulative_Counts'] = df_DALY_base['Counts'].cumsum()
                    df_DALY_base['Cumulative_Denominator'] = df_DALY_base['Denominator'].cumsum()
                    df_DALY_int['Cumulative_Counts'] = df_DALY_int['Counts'].cumsum()
                    df_DALY_int['Cumulative_Denominator'] = df_DALY_int['Denominator'].cumsum()

                    # Calculate cumulative rates for Baseline and Intervention
                    df_DALY_base['Rate'] = df_DALY_base['Cumulative_Counts'] / df_DALY_base['Cumulative_Denominator']
                    df_DALY_int['Rate'] = df_DALY_int['Cumulative_Counts'] / df_DALY_int['Cumulative_Denominator']

                    # Calculate cumulative DALYs averted
                    df_DALY_avt = df_DALY_base[['Month']].copy()
                    df_DALY_avt['Rate'] = (df_DALY_base['Rate'] - df_DALY_int['Rate']) * multiplier

                    return df_DALY_avt

                def prepare_referral_df(df, indicator, n_months, n_runs, scenario):
                    indicator = np.concatenate(df[indicator].values).reshape(-1, 2)
                    df_indicator = pd.DataFrame(indicator, columns=['Low SES', 'High SES'])
                    df_indicator['All'] = np.sum(df_indicator, axis=1)

                    df_indicator['Month'] = np.tile(np.arange(1, n_months+1), n_runs)
                    df_indicator['Run'] = np.repeat(np.arange(n_runs), n_months)
                    df_indicator['Scenario'] = scenario
                    #df_indicator = df_indicator.groupby(['Month', 'Scenario'], as_index=False).sum()
                    df_indicator = df_indicator.melt(id_vars=['Month', 'Run', 'Scenario'], var_name='Level', value_name='Counts')
                    return df_indicator

                def prepare_lb_df(df, indicator, n_months, n_runs, scenario):
                    indicator = np.concatenate(df[indicator].values).reshape(-1, 4).sum(axis=1)
                    df_indicator = pd.DataFrame(indicator, columns=['Denominator'])
                    df_indicator['Month'] = np.tile(np.arange(1, n_months+1), n_runs)
                    df_indicator['Run'] = np.repeat(np.arange(n_runs), n_months)
                    df_indicator['Scenario'] = scenario
                    #df_indicator = df_indicator.groupby(['Month', 'Scenario'], as_index=False).sum()
                    return df_indicator

                def prepare_referral_data(b_df, i_df, indicator, n_months, n_runs, multiplier):
                    b_df_ind = prepare_referral_df(b_df, indicator, n_months, n_runs, 'Baseline')
                    i_df_ind = prepare_referral_df(i_df, indicator, n_months, n_runs, 'Intervention')
                    df_ind = pd.concat([b_df_ind, i_df_ind], ignore_index=True)

                    b_df_lb = prepare_lb_df(b_df, 'Live Births Final', n_months, n_runs, 'Baseline')
                    i_df_lb = prepare_lb_df(i_df, 'Live Births Final', n_months, n_runs, 'Intervention')
                    df_lb = pd.concat([b_df_lb, i_df_lb], ignore_index=True)

                    df_ind = df_ind.merge(df_lb, on=['Month', 'Run', 'Scenario'], suffixes=('_ind', '_lb'))
                    df_ind.columns = ['Month', 'Run', 'Scenario', 'Level', 'Counts', 'Denominator']
                    df_ind['Rate'] = df_ind['Counts'] / df_ind['Denominator'] * multiplier
                    return df_ind

                if selected_plot == "Cost Effectiveness":
                    st.markdown("<h3 style='text-align: left;'>Cost per DALY averted</h3>",
                                unsafe_allow_html=True)
                    tab1, tab2, tab3, tab4 = st.tabs(["Single Interventions (Drugs and Supplies)", "POCUS", "Intrapartum Sensors", "HSS Interventions"])

                    cost_dic = i_param['cost_dict']

                    cost_dic = {key: value / i_param['USD_to_Ksh'] for key, value in cost_dic.items()}  #convert Ksh to USD

                    def prepare_df_ce(df, indicator, n_months, n_runs, scenario, n_cols):
                        indicator = np.concatenate(df[indicator].values).reshape(-1, n_cols)
                        column_names = [f'Col{i + 1}' for i in range(n_cols)]
                        df_indicator = pd.DataFrame(indicator, columns=column_names)
                        df_indicator['All'] = np.sum(df_indicator, axis=1)
                        df_indicator = df_indicator.drop(columns=column_names)
                        df_indicator = df_indicator.rename(columns={'All': 'Counts'})
                        df_indicator['Month'] = np.tile(np.arange(1, n_months+1), n_runs)
                        df_indicator['Scenario'] = scenario
                        df_indicator = df_indicator.groupby(['Month', 'Scenario'], as_index=False).mean()
                        return df_indicator

                    def acum_ind_df(indicator, ncols, order):
                        df_ind_base = prepare_df_ce(b_df, indicator, n_months, n_runs, 'Baseline', ncols)
                        df_ind_int = prepare_df_ce(i_df, indicator, n_months, n_runs, 'Intervention', ncols)
                        df_ind_base['Cumulative_Counts'] = df_ind_base['Counts'].cumsum()
                        df_ind_int['Cumulative_Counts'] = df_ind_int['Counts'].cumsum()
                        df_ind_diff = df_ind_base[['Month']].copy()
                        df_ind_diff['Cum_Count_Diff'] = df_ind_base['Cumulative_Counts'] - df_ind_int['Cumulative_Counts'] if order == 'baseline first' else df_ind_int['Cumulative_Counts'] - df_ind_base['Cumulative_Counts']
                        df_ind_diff['Count_Diff'] = df_ind_base['Counts'] - df_ind_int['Counts'] if order == 'baseline first' else df_ind_int['Counts'] - df_ind_base['Counts']
                        df_ind_diff['Count_Int'] = df_ind_int['Counts']
                        df_ind_diff['Cum_Count_Int'] = df_ind_int['Cumulative_Counts']
                        return df_ind_diff

                    df_daly_avt = acum_ind_df('DALYs', 4,'baseline first')
                    df_daly_avt = df_daly_avt.rename(columns={'Cum_Count_Diff': 'DALY averted'})

                    months = list(range(1, n_months + 1))
                    # FIXED COST
                    def fixed_cost_by_month(cost_type):
                        cost = np.array([(cost_type / int_period) * month \
                                             if month <= int_period else cost_type for month in months
                                         ])
                        return cost

                    #COST FOR SINGLE INTERVENTIONS
                    df_pph_bundle = acum_ind_df('Mothers with pph_bundle', 4,'intervention first')
                    df_pph_bundle['Cost'] = df_pph_bundle['Cum_Count_Diff'] * cost_dic['pph_bundle']
                    df_pph_bundle['Cost'] = df_pph_bundle['Cost'].clip(lower=0) if i_flags["flag_pph_bundle"] == 1 else 0
                    df_pph_bundle['Type'] = "PPH bundle"

                    df_iv_iron = acum_ind_df('Mothers with iv_iron', 4,'intervention first')
                    df_iv_iron['Cost'] = df_iv_iron['Cum_Count_Diff'] * cost_dic['iv_iron'] if i_flags["flag_iv_iron"] == 1 else 0
                    df_iv_iron['Cost'] = df_iv_iron['Cost'].clip(lower=0)
                    df_iv_iron['Type'] = "IV iron"

                    df_MgSO4 = acum_ind_df('Mothers with MgSO4', 4,'intervention first')
                    df_MgSO4['Cost'] = df_MgSO4['Cum_Count_Diff'] * cost_dic['MgSO4'] if i_flags["flag_MgSO4"] == 1 else 0
                    df_MgSO4['Cost'] = df_MgSO4['Cost'].clip(lower=0)
                    df_MgSO4['Type'] = "MgSO4"

                    df_antibiotics = acum_ind_df('Mothers with antibiotics', 4,'intervention first')
                    df_antibiotics['Cost'] = df_antibiotics['Cum_Count_Diff'] * cost_dic['antibiotics']
                    df_antibiotics['Cost'] = df_antibiotics['Cost'].clip(lower=0) if i_flags["flag_antibiotics"] == 1 else 0
                    df_antibiotics['Type'] = "Antibiotics"

                    df_single_cost = pd.concat([df_pph_bundle, df_iv_iron, df_MgSO4, df_antibiotics], ignore_index=True)
                    df_single_cost_all = df_single_cost.groupby('Month', as_index=False)['Cost'].sum()

                    ##COST FOR POCUS
                    if i_flags['flag_us'] == 1:
                        num_pocus = i_param['num_L4'] + i_param['num_L5'] + i_param['num_L2/3']  #each facility has one POCUS machine
                        pocus_cost = cost_dic['POCUS'] * num_pocus
                        cum_pocus_cost = np.array([pocus_cost for month in months])
                    else:
                        num_pocus = 0
                        cum_pocus_cost = np.zeros(n_months - 1)


                    # COST FOR 4+ANC VISITS
                    df_anc = acum_ind_df('ANC', 4, 'intervention first')
                    df_anc['Cost'] = df_anc['Cum_Count_Diff'] * cost_dic['SDR ANC']
                    df_anc['Cost'] = df_anc['Cost'].clip(lower=0)
                    df_anc['Type'] = "4+ANCs"

                    # COST FOR NORMAL DELIVERY
                    df_fac_delivery = acum_ind_df('Fac non-CS', 4,'intervention first')
                    df_fac_delivery['Cost'] = df_fac_delivery['Cum_Count_Diff'] * cost_dic['SDR Fac Delivery']
                    df_fac_delivery['Cost'] = df_fac_delivery['Cost'].clip(lower=0)
                    df_fac_delivery['Type'] = "Normal deliveries"

                    # COST FOR C-SECTION DELIVERY
                    df_cs_delivery = acum_ind_df('CS', 4,'intervention first')
                    df_cs_delivery['Cost'] = df_cs_delivery['Cum_Count_Diff'] * cost_dic['SDR CS Delivery']
                    df_cs_delivery['Cost'] = df_cs_delivery['Cost'].clip(lower=0)
                    df_cs_delivery['Type'] = "C-sections"

                    # COST FOR NORMAL REFERRALS
                    df_refer = acum_ind_df('Free_referrals', 2,'intervention first')
                    max_monthly_refers = max(df_refer['Count_Int'].max(), 0)
                    num_taxes_needed = math.ceil(max_monthly_refers / i_param['dispatches_per_vehicle'])
                    setup_cost = num_taxes_needed * cost_dic['SDR Taxi Setup']

                    cum_setup_cost = fixed_cost_by_month(setup_cost)
                    cum_monthly_cost = np.array([cost_dic['SDR Taxi Monthly'] * month for month in months]) * num_taxes_needed
                    cum_dispatch_cost = np.array([cost_dic['SDR Taxi Dispatch'] * df_refer['Cum_Count_Int'].loc[month - 1] for month in months])
                    cum_refer_cost = cum_setup_cost + cum_monthly_cost + cum_dispatch_cost
                    df_refer['Cost'] = cum_refer_cost
                    df_refer['Type'] = "Normal referrals"

                    # COST FOR EMERGENCY TRANSFERS - assuming all of them supported by facility-own ambulances
                    df_transfer = acum_ind_df("Emergency transfers", 4,'intervention first')
                    #max_monthly_transfers = max(df_transfer['Count_Int'].max(), 0)
                    max_monthly_transfers = max(df_transfer['Count_Diff'].max(), 0)
                    num_ambulances_needed = math.ceil(max_monthly_transfers / i_param['dispatches_per_vehicle'])
                    setup_cost = num_ambulances_needed * cost_dic['SDR Ambulance Setup']

                    cum_setup_cost = fixed_cost_by_month(setup_cost)
                    cum_monthly_cost = np.array([cost_dic['SDR Ambulance Monthly'] * month for month in months]) * num_ambulances_needed
                    cum_dispatch_cost = np.array([cost_dic['SDR Ambulance Dispatch'] * df_transfer['Cum_Count_Diff'].loc[month - 1] for month in months])
                    cum_dispatch_cost = np.maximum(cum_dispatch_cost, 0)
                    cum_transfer_cost = cum_setup_cost + cum_monthly_cost + cum_dispatch_cost
                    df_transfer['Cost'] = cum_transfer_cost
                    df_transfer['Type'] = "Emergency transfers"

                    # COST FOR WORKFORCE
                    df_surgical_labor = acum_ind_df('Surgical_actual', 2,'intervention first')
                    num_surgical_needed = max(df_surgical_labor['Count_Diff'].max(), 0)
                    monthly_surgical_salary = num_surgical_needed * cost_dic['SDR surgical staff']
                    cum_surgical_salary = np.array([monthly_surgical_salary * month for month in months])

                    df_nurse_labor = acum_ind_df('Nurse_actual', 2,'intervention first')
                    num_nurse_needed = max(df_nurse_labor['Count_Diff'].max(), 0)
                    monthly_nurse_salary = num_nurse_needed * cost_dic['SDR nurse staff']
                    cum_nurse_salary = np.array([monthly_nurse_salary * month for month in months])

                    df_anesthetist_labor = acum_ind_df('Anesthetist_actual', 2,'intervention first')
                    num_anesthetist_needed = max(df_anesthetist_labor['Count_Diff'].max(), 0)
                    monthly_anesthetist_salary = num_anesthetist_needed * cost_dic['SDR anesthetist']
                    cum_anesthetist_salary = np.array([monthly_anesthetist_salary * month for month in months])

                    cum_labor_cost = cum_surgical_salary + cum_nurse_salary + cum_anesthetist_salary
                    df_labor = pd.DataFrame({'Month': months,
                                            'Cost': cum_labor_cost,
                                            'Type': 'Direct Labor'
                                            }
                                        )

                    # COST FOR SENSORS
                    df_doppler = acum_ind_df('Doppler_Actual', 3,'intervention first')
                    num_doppler_needed = max(df_doppler['Count_Diff'].max(), 0)
                    doppler_cost = num_doppler_needed * cost_dic['Doppler']
                    cum_doppler_cost = np.array([doppler_cost for month in months])

                    df_ctg = acum_ind_df('CTG_Actual', 3,'intervention first')
                    num_ctg_needed = max(df_ctg['Count_Diff'].max(), 0)
                    ctg_cost = num_ctg_needed * cost_dic['CTG']
                    cum_ctg_cost = np.array([ctg_cost for month in months])
                    cum_sensor_cost = cum_doppler_cost + cum_ctg_cost
                    df_sensor = pd.DataFrame({'Month': months,
                                            'Cost': cum_sensor_cost,
                                            'Type': 'Intrapartum Sensors'
                                            }
                                        )

                    flag_single = 1 if (
                                i_flags['flag_pph_bundle'] or i_flags['flag_iv_iron'] or i_flags['flag_MgSO4'] or i_flags[
                            'flag_antibiotics']) else 0
                    flag_HSS = 1 if i_flags['flag_SDR'] else 0
                    flag_pocus = 1 if i_flags['flag_us'] else 0
                    flag_sensor = 1 if i_flags['flag_intrasensor'] else 0

                    with tab1:
                        if flag_single and not flag_HSS and not flag_pocus and not flag_sensor:
                            df_cost = pd.concat([df_pph_bundle, df_iv_iron, df_MgSO4, df_antibiotics], ignore_index=True)
                            df_cost_all = df_cost.groupby('Month', as_index=False)['Cost'].sum()
                            df_ce = df_daly_avt.copy()
                            df_ce['Cost per DALY averted'] = df_cost_all['Cost'] / df_daly_avt['DALY averted']

                            col1, col2 = st.columns(2)
                            with col1:
                                #plot cost effectiveness over month
                                chart = (
                                    alt.Chart(df_ce)
                                    .mark_line()
                                    .encode(
                                        x=alt.X("Month:Q", title="Time since the start of intervention implementation (Months)"),
                                        y=alt.Y("Cost per DALY averted:Q", axis=alt.Axis(title="USD")),
                                        tooltip=["Month:N", "Cost per DALY averted:Q"]
                                    )
                                    .properties(width=700, height=400)
                                    .interactive()
                                )

                                chart = chart.properties(
                                    title=alt.TitleParams(text="Accumulated Cost per DALY averted", anchor="middle")
                                )

                                st.altair_chart(chart)

                                num = df_ce['Cost per DALY averted'].iloc[-1]
                                st.markdown(
                                    f"<p style='font-size:30px;'>Cost per DALY averted by the end: USD {num:.0f}</p>",
                                    unsafe_allow_html=True
                                )

                            with col2:
                                #plot cost by type over month
                                chart = (
                                    alt.Chart(df_cost)
                                    .mark_line()
                                    .encode(
                                        x=alt.X("Month:Q", title="Time since the start of intervention implementation (Months)"),
                                        y=alt.Y("Cost:Q", axis=alt.Axis(title="USD")),
                                        color=alt.Color("Type:N", legend=alt.Legend(title="Intervention")),
                                        tooltip=["Month:N", "Type:N", "Cost:Q"]
                                    )
                                    .properties(width=700, height=400)
                                    .interactive()
                                )

                                chart = chart.properties(
                                    title=alt.TitleParams(text="Accumulated Cost by Single Interventions", anchor="middle")
                                )

                                st.altair_chart(chart)

                                num_mothers = np.sum(np.array([23729, 18196, 20709, 5127])) / 12 * n_months

                                cost_pph_bundle = df_cost[df_cost['Type'] == 'PPH bundle']['Cost'].iloc[-1] / num_mothers
                                cost_iv_iron = df_cost[df_cost['Type'] == 'IV iron']['Cost'].iloc[-1] / num_mothers
                                cost_MgSO4 = df_cost[df_cost['Type'] == 'MgSO4']['Cost'].iloc[-1] / num_mothers
                                cost_antibiotics = df_cost[df_cost['Type'] == 'Antibiotics']['Cost'].iloc[-1] / num_mothers

                                st.markdown(
                                    f"<p style='font-size:30px;'>Accumulated Cost for {num_mothers:.0f} dyads over {n_months} months</p>"
                                    f"<p style='font-size:24px;'>PPH bundle: USD {cost_pph_bundle:.2f} per dyad</p>"
                                    f"<p style='font-size:24px;'>IV iron infusion: USD {cost_iv_iron:.2f} per dyad</p>"
                                    f"<p style='font-size:24px;'>MgSO4: USD {cost_MgSO4:.2f} per dyad</p>"
                                    f"<p style='font-size:24px;'>Antibiotics: USD {cost_antibiotics:.2f} per dyad</p>"
                                    ,
                                    unsafe_allow_html=True
                                )
                        else:
                            st.markdown("<p style='font-size:24px;'>Please only adjust sliders under Treatment interventions (Drugs and Supplies) to view cost effectiveness</p>",
                                        unsafe_allow_html=True)

                    with tab2:
                        if flag_pocus and not flag_HSS and not flag_single and not flag_sensor:
                            df_pocus_ce = pd.DataFrame({'Month': months,
                                                        'Cost': cum_pocus_cost,
                                                        'DALY averted': df_daly_avt['DALY averted']
                                                        })

                            df_pocus_ce['Cost per DALY averted'] = df_pocus_ce['Cost'] / df_pocus_ce['DALY averted']

                            chart = (alt.Chart(
                                df_pocus_ce
                            ).mark_line().encode(
                                x=alt.X('Month:Q', title='Month'),
                                y=alt.Y('Cost per DALY averted:Q', axis=alt.Axis(title='USD')),
                                tooltip=['Month:N', 'Cost per DALY averted:Q']
                            ).properties(width=700, height=400).interactive())

                            chart = chart.properties(
                                title=alt.TitleParams(text='Accumulated Cost per DALY averted', anchor='middle')
                            )

                            st.altair_chart(chart)

                            num = df_pocus_ce['Cost per DALY averted'].iloc[-1]
                            st.markdown(
                                f"<p style='font-size:30px;'>Cost per DALY averted by the end: USD {num:.0f} with {num_pocus} POCUS machines</p>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown("<p style='font-size:24px;'>Please only adjust sliders under AI portable ultrasound to view cost effectiveness</p>",
                                        unsafe_allow_html=True)

                    with tab3:
                        if flag_sensor and not flag_HSS and not flag_single and not flag_pocus:
                            df_sensor_ce = pd.DataFrame({'Month': months,
                                                        'Cost': cum_sensor_cost,
                                                        'DALY averted': df_daly_avt['DALY averted']
                                                        })

                            df_sensor_ce['Cost per DALY averted'] = df_sensor_ce['Cost'] / df_sensor_ce['DALY averted']

                            chart = (alt.Chart(
                                df_sensor_ce
                            ).mark_line().encode(
                                x=alt.X('Month:Q', title='Month'),
                                y=alt.Y('Cost per DALY averted:Q', axis=alt.Axis(title='USD')),
                                tooltip=['Month:N', 'Cost per DALY averted:Q']
                            ).properties(width=700, height=400).interactive())

                            chart = chart.properties(
                                title=alt.TitleParams(text='Accumulated Cost per DALY averted', anchor='middle')
                            )

                            st.altair_chart(chart)

                            num = df_sensor_ce['Cost per DALY averted'].iloc[-1]
                            st.markdown(
                                f"<p style='font-size:30px;'>Cost per DALY averted by the end: USD {num:.0f} with {num_doppler_needed:.0f} Doppler machines and {num_ctg_needed:.0f} CTG machines</p>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown("<p style='font-size:24px;'>Please only adjust sliders under Intrapartum sensors to view cost effectiveness</p>",
                                        unsafe_allow_html=True)

                    with tab4:
                        if flag_HSS:
                            added_capacity = i_param["Capacity"] * i_param["HSS"]["capacity_added"]
                            beds_needed = math.ceil (added_capacity / (83 / 12))

                            fixed_cost_total = cost_dic['SDR PM'] + cost_dic['SDR Infra'] * beds_needed + cost_dic['SDR Equip']

                            # CONTINOUS COST
                            def maintain_cost_PM(cost_type):
                                cost = np.array([(cost_type / 12) * month \
                                            if month > int_period else 0 for month in months
                                        ])
                                return cost

                            maintain_cost_PM = maintain_cost_PM(cost_dic['SDR PM2'])

                            df_sdr_cost = pd.DataFrame({'Month': months,
                                                        'Program Management': fixed_cost_by_month(cost_dic['SDR PM']) + maintain_cost_PM,
                                                        'Infrastructure': fixed_cost_by_month(cost_dic['SDR Infra'] * beds_needed),
                                                        'Equipment': fixed_cost_by_month(cost_dic['SDR Equip']) * i_flags['flag_equipment'] + df_sensor['Cost'],
                                                        'Direct Labor': df_labor['Cost'],
                                                        'Normal Referrals': df_refer['Cost'],
                                                        'Emergency Transfers': df_transfer['Cost'],
                                                        'ANCs': df_anc['Cost'],
                                                        'Single Interventions': df_single_cost_all['Cost'],
                                                        'Normal Deliveries': df_fac_delivery['Cost'],
                                                        'C-sections': df_cs_delivery['Cost']
                                                        }
                                                       )

                            SDR_cost_total = df_sdr_cost['Program Management'] + df_sdr_cost['Infrastructure'] \
                                             + df_sdr_cost['Equipment'] + df_sdr_cost['Direct Labor'] \
                                             + df_sdr_cost['Normal Referrals'] + df_sdr_cost['Emergency Transfers'] \
                                             + df_sdr_cost['ANCs'] + df_sdr_cost['Single Interventions'] \
                                             + df_sdr_cost['Normal Deliveries'] + df_sdr_cost['C-sections']

                            df_sdr_cost = df_sdr_cost.melt(id_vars=['Month'], var_name='Type', value_name='Cost')

                            df_sdr_ce = pd.DataFrame({'Month': months,
                                                        'Cost': SDR_cost_total,
                                                        'DALY averted': df_daly_avt['DALY averted']
                                                        })

                            df_sdr_ce['Cost per DALY averted'] = df_sdr_ce['Cost'] / df_sdr_ce['DALY averted']

                            # Replace negative values with None (to exclude them from the plot)
                            df_sdr_ce.loc[df_sdr_ce['Cost per DALY averted'] < 0, 'Cost per DALY averted'] = None

                            col1, col2 = st.columns(2)
                            with col1:
                                # plot cost effectiveness over month
                                chart = (
                                    alt.Chart(df_sdr_ce)
                                    .mark_line()
                                    .encode(
                                        x=alt.X("Month:Q", title="Time since the start of intervention implementation (Months)"),
                                        y=alt.Y("Cost per DALY averted:Q", axis=alt.Axis(title="Cost per DALY averted in USD")),
                                        tooltip=["Month:N", "Cost per DALY averted:Q"]
                                    )
                                    .properties(width=700, height=400)
                                    .interactive()
                                )

                                chart = chart.properties(
                                    title=alt.TitleParams(text="Accumulated Cost per DALY averted", anchor="middle")
                                )

                                st.altair_chart(chart)

                                num = df_sdr_ce['Cost per DALY averted'].iloc[-1]
                                st.markdown(
                                    f"<p style='font-size:30px;'>Cost per DALY averted by the end: USD {num:.0f}</p>",
                                    unsafe_allow_html=True
                                )


                                def cost_per_daly_matplotlib(df_sdr_ce):
                                    fig, ax = plt.subplots(figsize=(8, 5))  # Define figure size

                                    # Plot the cost per DALY averted
                                    ax.plot(df_sdr_ce["Month"], df_sdr_ce["Cost per DALY averted"], linewidth=2,
                                            color="tab:blue")

                                    # Labels and Title
                                    ax.set_xlabel("Time since the start of intervention implementation (Months)",
                                                  fontsize=14)
                                    ax.set_ylabel("Cost per DALY averted in USD", fontsize=14)
                                    ax.set_title("Accumulated Cost per DALY Averted", fontsize=16)

                                    # Grid, Formatting, and Layout
                                    ax.grid(True, linestyle="--", alpha=0.5)
                                    plt.xticks(fontsize=12)
                                    plt.yticks(fontsize=12)
                                    plt.tight_layout()

                                    return fig


                                # fig = cost_per_daly_matplotlib(df_sdr_ce)
                                # st.pyplot(fig)  # Show plot in Streamlit
                                #
                                # # Enable Download of the Chart
                                # img_buffer = io.BytesIO()
                                # fig.savefig(img_buffer, format="png", dpi=300)
                                # st.download_button(label="Download Cost per DALY Averted Plot", data=img_buffer.getvalue(),
                                #                    file_name="cost_per_daly.png", mime="image/png",
                                #                    key="download_cost_daly")

                            with col2:
                                # plot cost by type over month
                                chart = (
                                    alt.Chart(df_sdr_cost)
                                    .mark_line()
                                    .encode(
                                        x=alt.X("Month:Q", title="Time since the start of intervention implementation (Months)"),
                                        y=alt.Y("Cost:Q", axis=alt.Axis(title="USD")),
                                        color=alt.Color("Type:N", legend=alt.Legend(title="Intervention")),
                                        tooltip=["Month:N", "Type:N", "Cost:Q"]
                                    )
                                    .properties(width=700, height=400)
                                    .interactive()
                                )

                                chart = chart.properties(
                                    title=alt.TitleParams(text="Accumulated Cost by SDR Interventions", anchor="middle")
                                )

                                st.altair_chart(chart)

                                cost_pm = df_sdr_cost[df_sdr_cost['Type'] == 'Program Management']['Cost'].iloc[-1]
                                cost_infra = df_sdr_cost[df_sdr_cost['Type'] == 'Infrastructure']['Cost'].iloc[-1]
                                cost_equip = df_sdr_cost[df_sdr_cost['Type'] == 'Equipment']['Cost'].iloc[-1]
                                cost_labor = df_sdr_cost[df_sdr_cost['Type'] == 'Direct Labor']['Cost'].iloc[-1]
                                cost_refer = df_sdr_cost[df_sdr_cost['Type'] == 'Normal Referrals']['Cost'].iloc[-1]
                                cost_transfer = df_sdr_cost[df_sdr_cost['Type'] == 'Emergency Transfers']['Cost'].iloc[-1]
                                cost_single = df_sdr_cost[df_sdr_cost['Type'] == 'Single Interventions']['Cost'].iloc[-1]
                                cost_anc = df_sdr_cost[df_sdr_cost['Type'] == 'ANCs']['Cost'].iloc[-1]
                                cost_normal = df_sdr_cost[df_sdr_cost['Type'] == 'Normal Deliveries']['Cost'].iloc[-1]
                                cost_cs = df_sdr_cost[df_sdr_cost['Type'] == 'C-sections']['Cost'].iloc[-1]

                                st.markdown(
                                    f"<p style='font-size:30px;'>Accumulated Cost by SDR Interventions</p>"
                                    f"<p style='font-size:24px;'>Program Management: USD {cost_pm:.0f}</p>"
                                    f"<p style='font-size:24px;'>Infrastructure: USD {cost_infra:.0f} for expanding {beds_needed} beds</p>"
                                    f"<p style='font-size:24px;'>Equipment: USD {cost_equip:.0f}, including {num_doppler_needed:.0f} Doppler machines and {num_ctg_needed:.0f} CTG machines</p>"
                                    f"<p style='font-size:24px;'>Direct Labor: USD {cost_labor:.0f} with additional {num_surgical_needed:.0f} surgical staff, {num_nurse_needed:.0f} nurses/midwifes, and {num_anesthetist_needed:.0f} anesthetists</p>"
                                    f"<p style='font-size:24px;'>Normal Referrals: USD {cost_refer:.0f} with {num_taxes_needed:.0f} taxis</p>"
                                    f"<p style='font-size:24px;'>Emergency Transfers: USD {cost_transfer:.0f} with {num_ambulances_needed:.0f} ambulances</p>"
                                    f"<p style='font-size:24px;'>Single Interventions: USD {cost_single:.0f}</p>"
                                    f'<p style="font-size:24px;">ANCs: USD {cost_anc:.0f}</p>'
                                    f'<p style="font-size:24px;">Normal Deliveries: USD {cost_normal:.0f}</p>'
                                    f'<p style="font-size:24px;">C-sections: USD {cost_cs:.0f}</p>'
                                    ,
                                    unsafe_allow_html=True
                                )

                            col3, col4, col5, col6 = st.columns(4)
                            with col3:
                                scenario_name = st.text_input("**Enter Intervention Scenario Name:**", placeholder="e.g., Scenario_1")

                                # Ensure a scenario name is entered
                                if scenario_name:
                                    file_name1 = f"{scenario_name}ICER.csv"
                                    csv_sdr_ce = df_sdr_ce.to_csv(index=False)
                                    st.download_button(label=f"📥 Download {file_name1}", data=csv_sdr_ce,
                                                       file_name=file_name1,
                                                       mime="text/csv")
                                    file_name2 = f"{scenario_name}COST.csv"
                                    csv_sdr_cost = df_sdr_cost.to_csv(index=False)
                                    st.download_button(label=f"📥 Download {file_name2}", data=csv_sdr_cost,
                                                       file_name=file_name2,
                                                       mime="text/csv")

                        else:
                            st.markdown("<p style='font-size:24px;'>Please adjust sliders under Health System Strengthening (HSS) Interventions to view cost effectiveness</p>",
                                        unsafe_allow_html=True)
                            st.markdown("<p style='font-size:18px;'>---Sliders under single interventions can also be adjusted together to see combined cost effectiveness</p>",
                                        unsafe_allow_html=True)

                if selected_plot == "Labor force ratio":
                    st.markdown("<h3 style='text-align: left;'>Labor force ratio</h3>",
                                unsafe_allow_html=True)

                    def prepare_indicator_df(df, indicator, level, n_months, n_runs, scenario):
                        df_indicator = np.concatenate(df[indicator].values).reshape(-1, 2)
                        df_indicator = pd.DataFrame(df_indicator, columns=['L4', 'L5'])
                        if level == 0:
                            df_indicator = df_indicator[['L4']]
                        else:
                            df_indicator = df_indicator[['L5']]
                        df_indicator['Month'] = np.tile(np.arange(1, n_months+1), n_runs)
                        df_indicator['Scenario'] = scenario
                        df_indicator = df_indicator.groupby(['Month', 'Scenario'], as_index=False).mean()
                        return df_indicator

                    def prepare_chart_data(b_df, i_df, indicator, level, n_months, n_runs):
                        b_df_ind = prepare_indicator_df(b_df, indicator, level, n_months, n_runs, 'Baseline')
                        i_df_ind = prepare_indicator_df(i_df, indicator, level, n_months, n_runs, 'Intervention')
                        df_ind = pd.concat([b_df_ind, i_df_ind], ignore_index=True)
                        return df_ind

                    def plot_labor(df, title, yvariable, ytitle, ymax):
                        chart = (
                            alt.Chart(df)
                            .mark_line()
                            .encode(
                                x=alt.X("Month:Q", title="Time since the start of intervention implementation (Months)"),
                                y=alt.Y(yvariable, axis=alt.Axis(title=ytitle), scale=alt.Scale(domain=[0, ymax])),
                                color=alt.Color("Scenario:N", legend=alt.Legend(title="Scenario")),
                                tooltip=["Month:N", "Scenario:N", yvariable]
                            )
                            .properties(width=600, height=400)
                            .interactive()
                        )
                        chart = chart.properties(
                            title=alt.TitleParams(text=title, anchor="middle")
                        )
                        return chart


                    tab1, tab2, tab3 = st.tabs(["Surgical staff", "Nurse staff", "Anesthetist staff"])

                    with tab1:
                        df_surgical_ratio_l4 = prepare_chart_data(b_df, i_df, 'Surgical_ratio', 0, n_months, n_runs)
                        df_surgical_ratio_l5 = prepare_chart_data(b_df, i_df, 'Surgical_ratio', 1, n_months, n_runs)
                        df_surgical_actual_l4 = prepare_chart_data(b_df, i_df, 'Surgical_actual', 0, n_months, n_runs)
                        df_surgical_actual_l5 = prepare_chart_data(b_df, i_df, 'Surgical_actual', 1, n_months, n_runs)
                        df_surgical_needed_l4 = prepare_chart_data(b_df, i_df, 'Surgical_needed', 0, n_months, n_runs)
                        df_surgical_needed_l5 = prepare_chart_data(b_df, i_df, 'Surgical_needed', 1, n_months, n_runs)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            ymax = max(df_surgical_ratio_l4['L4'].max(), df_surgical_ratio_l5['L5'].max())
                            chart1 = plot_labor(df_surgical_ratio_l4, "Ratio of surgical staff (actual versus needed) at L4", "L4", "Ratio", ymax)
                            st.altair_chart(chart1)
                            st.markdown("~~~")
                            chart2 = plot_labor(df_surgical_ratio_l5, "Ratio of surgical staff (actual versus needed) at L5", "L5", "Ratio", ymax)
                            st.altair_chart(chart2)
                        with col2:
                            ymax = df_surgical_needed_l4['L4'].max()
                            chart1 = plot_labor(df_surgical_actual_l4, "Actual surgical staff at L4", "L4", "# Surgical staff", ymax)
                            st.altair_chart(chart1)
                            st.markdown("~~~")
                            ymax = df_surgical_needed_l5['L5'].max()
                            chart2 = plot_labor(df_surgical_actual_l5, "Actual surgical staff at L5", "L5", "# Surgical staff", ymax)
                            st.altair_chart(chart2)
                        with col3:
                            ymax = df_surgical_needed_l4['L4'].max()
                            chart1 = plot_labor(df_surgical_needed_l4, "Needed surgical staff at L4", "L4", "# Surgical staff", ymax)
                            st.altair_chart(chart1)
                            st.markdown("~~~")
                            ymax = df_surgical_needed_l5['L5'].max()
                            chart2 = plot_labor(df_surgical_needed_l5, "Needed surgical staff at L5", "L5", "# Surgical staff", ymax)
                            st.altair_chart(chart2)

                    with tab2:
                        df_nurse_ratio_l4 = prepare_chart_data(b_df, i_df, 'Nurse_ratio', 0, n_months, n_runs)
                        df_nurse_ratio_l5 = prepare_chart_data(b_df, i_df, 'Nurse_ratio', 1, n_months, n_runs)
                        df_nurse_actual_l4 = prepare_chart_data(b_df, i_df, 'Nurse_actual', 0, n_months, n_runs)
                        df_nurse_actual_l5 = prepare_chart_data(b_df, i_df, 'Nurse_actual', 1, n_months, n_runs)
                        df_nurse_needed_l4 = prepare_chart_data(b_df, i_df, 'Nurse_needed', 0, n_months, n_runs)
                        df_nurse_needed_l5 = prepare_chart_data(b_df, i_df, 'Nurse_needed', 1, n_months, n_runs)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            ymax = max(df_nurse_ratio_l4['L4'].max(), df_nurse_ratio_l5['L5'].max())
                            chart1 = plot_labor(df_nurse_ratio_l4, "Ratio of nurse staff (actual versus needed) at L4", "L4", "Ratio", ymax)
                            st.altair_chart(chart1)
                            st.markdown("~~~")
                            chart2 = plot_labor(df_nurse_ratio_l5, "Ratio of nurse staff (actual versus needed) at L5", "L5", "Ratio", ymax)
                            st.altair_chart(chart2)
                        with col2:
                            ymax = df_nurse_needed_l4['L4'].max()
                            chart1 = plot_labor(df_nurse_actual_l4, "Actual nurse staff at L4", "L4", "# Nurse staff", ymax)
                            st.altair_chart(chart1)
                            st.markdown("~~~")
                            ymax = df_nurse_needed_l5['L5'].max()
                            chart2 = plot_labor(df_nurse_actual_l5, "Actual nurse staff at L5", "L5", "# Nurse staff", ymax)
                            st.altair_chart(chart2)
                        with col3:
                            ymax = df_nurse_needed_l4['L4'].max()
                            chart1 = plot_labor(df_nurse_needed_l4, "Needed nurse staff at L4", "L4", "# Nurse staff", ymax)
                            st.altair_chart(chart1)
                            st.markdown("~~~")
                            ymax = df_nurse_needed_l5['L5'].max()
                            chart2 = plot_labor(df_nurse_needed_l5, "Needed nurse staff at L5", "L5", "# Nurse staff", ymax)
                            st.altair_chart(chart2)

                    with tab3:
                        df_anesthetist_ratio_l4 = prepare_chart_data(b_df, i_df, 'Anesthetist_ratio', 0, n_months, n_runs)
                        df_anesthetist_ratio_l5 = prepare_chart_data(b_df, i_df, 'Anesthetist_ratio', 1, n_months, n_runs)
                        df_anesthetist_actual_l4 = prepare_chart_data(b_df, i_df, 'Anesthetist_actual', 0, n_months, n_runs)
                        df_anesthetist_actual_l5 = prepare_chart_data(b_df, i_df, 'Anesthetist_actual', 1, n_months, n_runs)
                        df_anesthetist_needed_l4 = prepare_chart_data(b_df, i_df, 'Anesthetist_needed', 0, n_months, n_runs)
                        df_anesthetist_needed_l5 = prepare_chart_data(b_df, i_df, 'Anesthetist_needed', 1, n_months, n_runs)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            ymax = max(df_anesthetist_ratio_l4['L4'].max(), df_anesthetist_ratio_l5['L5'].max())
                            chart1 = plot_labor(df_anesthetist_ratio_l4, "Ratio of anesthetist staff (actual versus needed) at L4", "L4", "Ratio", ymax)
                            st.altair_chart(chart1)
                            st.markdown("~~~")
                            chart2 = plot_labor(df_anesthetist_ratio_l5, "Ratio of anesthetist staff (actual versus needed) at L5", "L5", "Ratio", ymax)
                            st.altair_chart(chart2)
                        with col2:
                            ymax = df_anesthetist_needed_l4['L4'].max()
                            chart1 = plot_labor(df_anesthetist_actual_l4, "Actual anesthetist staff at L4", "L4", "# Anesthetist staff", ymax)
                            st.altair_chart(chart1)
                            st.markdown("~~~")
                            ymax = df_anesthetist_needed_l4['L4'].max()
                            chart2 = plot_labor(df_anesthetist_actual_l5, "Actual anesthetist staff at L5", "L5", "# Anesthetist staff", ymax)
                            st.altair_chart(chart2)
                        with col3:
                            ymax = df_anesthetist_needed_l4['L4'].max()
                            chart1 = plot_labor(df_anesthetist_needed_l4, "Needed anesthetist staff at L4", "L4", "# Anesthetist staff", ymax)
                            st.altair_chart(chart1)
                            st.markdown("~~~")
                            ymax = df_anesthetist_needed_l4['L4'].max()
                            chart2 = plot_labor(df_anesthetist_needed_l5, "Needed anesthetist staff at L5", "L5", "# Anesthetist staff", ymax)
                            st.altair_chart(chart2)

                if selected_plot == "Equipment inventory ratio":
                    st.markdown("<h3 style='text-align: left;'>Equipment inventory ratio</h3>",
                                  unsafe_allow_html=True)

                    def prepare_indicator_df(df, indicator, level, n_months, n_runs, scenario):
                        df_indicator = np.concatenate(df[indicator].values).reshape(-1, 3)
                        df_indicator = pd.DataFrame(df_indicator, columns=['L2/3', 'L4', 'L5'])
                        if level == 0:
                            df_indicator = df_indicator[['L2/3']]
                        elif level == 1:
                            df_indicator = df_indicator[['L4']]
                        else:
                            df_indicator = df_indicator[['L5']]
                        df_indicator['Month'] = np.tile(np.arange(1, n_months+1), n_runs)
                        df_indicator['Scenario'] = scenario
                        df_indicator = df_indicator.groupby(['Month', 'Scenario'], as_index=False).mean()
                        return df_indicator

                    def prepare_chart_data(b_df, i_df, indicator, level, n_months, n_runs):
                        b_df_ind = prepare_indicator_df(b_df, indicator, level, n_months, n_runs, 'Baseline')
                        i_df_ind = prepare_indicator_df(i_df, indicator, level, n_months, n_runs, 'Intervention')
                        df_ind = pd.concat([b_df_ind, i_df_ind], ignore_index=True)
                        return df_ind

                    def plot_equipment(df, title, yvariable, ytitle, ymax):
                        chart = (
                            alt.Chart(df)
                            .mark_line()
                            .encode(
                                x=alt.X("Month:Q", title="Time since the start of intervention implementation (Months)"),
                                y=alt.Y(yvariable, axis=alt.Axis(title=ytitle), scale=alt.Scale(domain=[0, ymax])),
                                color=alt.Color("Scenario:N", legend=alt.Legend(title="Scenario")),
                                tooltip=["Month:N", "Scenario:N", yvariable]
                            )
                            .properties(width=600, height=400)
                            .interactive()
                        )
                        chart = chart.properties(
                            title=alt.TitleParams(text=title, anchor="middle")
                        )
                        return chart

                    tab1, tab2 = st.tabs(["Hand-held dopplers", "CTGs"])
                    with tab1:
                        df_doppler_ratio_l23 = prepare_chart_data(b_df, i_df, 'Doppler_Ratio', 0, n_months, n_runs)
                        df_doppler_ratio_l4 = prepare_chart_data(b_df, i_df, 'Doppler_Ratio', 1, n_months, n_runs)
                        df_doppler_ratio_l5 = prepare_chart_data(b_df, i_df, 'Doppler_Ratio', 2, n_months, n_runs)
                        df_doppler_actual_l23 = prepare_chart_data(b_df, i_df, 'Doppler_Actual', 0, n_months, n_runs)
                        df_doppler_actual_l4 = prepare_chart_data(b_df, i_df, 'Doppler_Actual', 1, n_months, n_runs)
                        df_doppler_actual_l5 = prepare_chart_data(b_df, i_df, 'Doppler_Actual', 2, n_months, n_runs)
                        df_doppler_needed_l23 = prepare_chart_data(b_df, i_df, 'Doppler_Needed', 0, n_months, n_runs)
                        df_doppler_needed_l4 = prepare_chart_data(b_df, i_df, 'Doppler_Needed', 1, n_months, n_runs)
                        df_doppler_needed_l5 = prepare_chart_data(b_df, i_df, 'Doppler_Needed', 2, n_months, n_runs)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            ymax = max(df_doppler_ratio_l23['L2/3'].max(), df_doppler_ratio_l4['L4'].max(), df_doppler_ratio_l5['L5'].max())
                            chart1 = plot_equipment(df_doppler_ratio_l23, "Ratio of hand-held dopplers (actual versus needed) at L2/3", "L2/3", "Ratio", ymax)
                            st.altair_chart(chart1)
                            st.markdown("~~~")
                            chart2 = plot_equipment(df_doppler_ratio_l4, "Ratio of hand-held dopplers (actual versus needed) at L4", "L4", "Ratio", ymax)
                            st.altair_chart(chart2)
                            st.markdown("~~~")
                            chart3 = plot_equipment(df_doppler_ratio_l5, "Ratio of hand-held dopplers (actual versus needed) at L5", "L5", "Ratio", ymax)
                            st.altair_chart(chart3)
                        with col2:
                            ymax = max(df_doppler_needed_l23['L2/3'].max(), df_doppler_actual_l23['L2/3'].max())
                            chart1 = plot_equipment(df_doppler_actual_l23, "Actual hand-held dopplers at L2/3", "L2/3", "# Hand-held dopplers", ymax)
                            st.altair_chart(chart1)
                            st.markdown("~~~")
                            ymax = max(df_doppler_needed_l4['L4'].max(), df_doppler_actual_l4['L4'].max())
                            chart2 = plot_equipment(df_doppler_actual_l4, "Actual hand-held dopplers at L4", "L4", "# Hand-held dopplers", ymax)
                            st.altair_chart(chart2)
                            st.markdown("~~~")
                            ymax = max(df_doppler_needed_l5['L5'].max(), df_doppler_actual_l5['L5'].max())
                            chart3 = plot_equipment(df_doppler_actual_l5, "Actual hand-held dopplers at L5", "L5", "# Hand-held dopplers", ymax)
                            st.altair_chart(chart3)
                        with col3:
                            ymax = max(df_doppler_needed_l23['L2/3'].max(), df_doppler_actual_l23['L2/3'].max())
                            chart1 = plot_equipment(df_doppler_needed_l23, "Needed hand-held dopplers at L2/3", "L2/3", "# Hand-held dopplers", ymax)
                            st.altair_chart(chart1)
                            st.markdown("~~~")
                            ymax = max(df_doppler_needed_l4['L4'].max(), df_doppler_actual_l4['L4'].max())
                            chart2 = plot_equipment(df_doppler_needed_l4, "Needed hand-held dopplers at L4", "L4", "# Hand-held dopplers", ymax)
                            st.altair_chart(chart2)
                            st.markdown("~~~")
                            ymax = max(df_doppler_needed_l5['L5'].max(), df_doppler_actual_l5['L5'].max())
                            chart3 = plot_equipment(df_doppler_needed_l5, "Needed hand-held dopplers at L5", "L5", "# Hand-held dopplers", ymax)
                            st.altair_chart(chart3)

                    with tab2:
                        df_ctg_ratio_l23 = prepare_chart_data(b_df, i_df, 'CTG_Ratio', 0, n_months, n_runs)
                        df_ctg_ratio_l4 = prepare_chart_data(b_df, i_df, 'CTG_Ratio', 1, n_months, n_runs)
                        df_ctg_ratio_l5 = prepare_chart_data(b_df, i_df, 'CTG_Ratio', 2, n_months, n_runs)
                        df_ctg_actual_l23 = prepare_chart_data(b_df, i_df, 'CTG_Actual', 0, n_months, n_runs)
                        df_ctg_actual_l4 = prepare_chart_data(b_df, i_df, 'CTG_Actual', 1, n_months, n_runs)
                        df_ctg_actual_l5 = prepare_chart_data(b_df, i_df, 'CTG_Actual', 2, n_months, n_runs)
                        df_ctg_needed_l23 = prepare_chart_data(b_df, i_df, 'CTG_Needed', 0, n_months, n_runs)
                        df_ctg_needed_l4 = prepare_chart_data(b_df, i_df, 'CTG_Needed', 1, n_months, n_runs)
                        df_ctg_needed_l5 = prepare_chart_data(b_df, i_df, 'CTG_Needed', 2, n_months, n_runs)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            ymax = max(df_ctg_ratio_l23['L2/3'].max(), df_ctg_ratio_l4['L4'].max(), df_ctg_ratio_l5['L5'].max())
                            chart1 = plot_equipment(df_ctg_ratio_l23, "Ratio of CTGs (actual versus needed) at L2/3", "L2/3", "Ratio", ymax)
                            st.altair_chart(chart1)
                            st.markdown("~~~")
                            chart2 = plot_equipment(df_ctg_ratio_l4, "Ratio of CTGs (actual versus needed) at L4", "L4", "Ratio", ymax)
                            st.altair_chart(chart2)
                            st.markdown("~~~")
                            chart3 = plot_equipment(df_ctg_ratio_l5, "Ratio of CTGs (actual versus needed) at L5", "L5", "Ratio", ymax)
                            st.altair_chart(chart3)
                        with col2:
                            ymax = max(df_ctg_needed_l23['L2/3'].max(), df_ctg_actual_l23['L2/3'].max())
                            chart1 = plot_equipment(df_ctg_actual_l23, "Actual CTGs at L2/3", "L2/3", "# CTGs", ymax)
                            st.altair_chart(chart1)
                            st.markdown("~~~")
                            ymax = max(df_ctg_needed_l4['L4'].max(), df_ctg_actual_l4['L4'].max())
                            chart2 = plot_equipment(df_ctg_actual_l4, "Actual CTGs at L4", "L4", "# CTGs", ymax)
                            st.altair_chart(chart2)
                            st.markdown("~~~")
                            ymax = max(df_ctg_needed_l5['L5'].max(), df_ctg_actual_l5['L5'].max())
                            chart3 = plot_equipment(df_ctg_actual_l5, "Actual CTGs at L5", "L5", "# CTGs", ymax)
                            st.altair_chart(chart3)
                        with col3:
                            ymax = max(df_ctg_needed_l23['L2/3'].max(), df_ctg_actual_l23['L2/3'].max())
                            chart1 = plot_equipment(df_ctg_needed_l23, "Needed CTGs at L2/3", "L2/3", "# CTGs", ymax)
                            st.altair_chart(chart1)
                            st.markdown("~~~")
                            ymax = max(df_ctg_needed_l4['L4'].max(), df_ctg_actual_l4['L4'].max())
                            chart2 = plot_equipment(df_ctg_needed_l4, "Needed CTGs at L4", "L4", "# CTGs", ymax)
                            st.altair_chart(chart2)
                            ymax = max(df_ctg_needed_l5['L5'].max(), df_ctg_actual_l5['L5'].max())
                            st.markdown("~~~")
                            chart3 = plot_equipment(df_ctg_needed_l5, "Needed CTGs at L5", "L5", "# CTGs", ymax)
                            st.altair_chart(chart3)

                if selected_plot == "Facility capacity ratio":
                    st.markdown("<h3 style='text-align: left;'>Facility capacity ratio</h3>",
                                unsafe_allow_html=True)

                    def prepare_indicator_df(df, indicator, n_months, n_runs, scenario):
                        df_indicator = pd.DataFrame(df[indicator])
                        df_indicator['Month'] = np.tile(np.arange(1, n_months+1), n_runs)
                        df_indicator['Run'] = np.repeat(np.arange(n_runs), n_months)
                        df_indicator['Scenario'] = scenario
                        #df_indicator = df_indicator.groupby(['Month', 'Scenario'], as_index=False).mean()
                        return df_indicator

                    def prepare_chart_data(b_df, i_df, indicator, n_months, n_runs):
                        b_df_ind = prepare_indicator_df(b_df, indicator, n_months, n_runs, 'Baseline')
                        i_df_ind = prepare_indicator_df(i_df, indicator, n_months, n_runs, 'Intervention')
                        df_ind = pd.concat([b_df_ind, i_df_ind], ignore_index=True)
                        df_ind = df_ind.groupby(['Month', 'Scenario'], as_index=False)[indicator].mean()
                        #df_ind.columns = ['Month', 'Scenario', indicator]
                        return df_ind

                    def prepare_ratio_data(b_df, i_df, numerator, dominator, n_months, n_runs):
                        b_df_num = prepare_indicator_df(b_df, numerator, n_months, n_runs, 'Baseline')
                        i_df_num = prepare_indicator_df(i_df, numerator, n_months, n_runs, 'Intervention')
                        df_num = pd.concat([b_df_num, i_df_num], ignore_index=True)

                        b_df_dom = prepare_indicator_df(b_df, dominator, n_months, n_runs, 'Baseline')
                        i_df_dom = prepare_indicator_df(i_df, dominator, n_months, n_runs, 'Intervention')
                        df_dom = pd.concat([b_df_dom, i_df_dom], ignore_index=True)

                        df = df_num.merge(df_dom, on=['Month', 'Run', 'Scenario'], suffixes=('_num', '_dom'))
                        df.columns = ['Counts', 'Month', 'Run', 'Scenario', 'Denominator']
                        df['Rate'] = df['Counts'] / df['Denominator']
                        df['Counts'] = df['Counts'].astype(float)
                        df['Denominator'] = df['Denominator'].astype(float)
                        return df

                    def create_ratio_line_data(data):
                        line_data = data.copy()
                        line_data = add_poisson_ci(line_data, multiplier=1)

                        line_data = line_data.groupby(['Month', 'Scenario'], as_index=False).agg({
                            'Rate': 'mean',
                            'Lower_rate': 'mean',
                            'Upper_rate': 'mean'
                        })

                        line_data[['Rate', 'Lower_rate', 'Upper_rate']] = line_data[
                            ['Rate', 'Lower_rate', 'Upper_rate']].round(
                            2)

                        return line_data

                    #df_cap_ratio = prepare_chart_data(b_df, i_df, 'Capacity Ratio', n_months, n_runs)
                    df_cap_ratio = prepare_ratio_data(b_df, i_df, 'Facility_capacity_ideal', 'Facility_capacity_actual', n_months, n_runs)
                    line_data = create_ratio_line_data(df_cap_ratio)

                    df_fac_cap_actual = prepare_chart_data(b_df, i_df, 'Facility_capacity_actual', n_months, n_runs)
                    df_fac_cap_ideal = prepare_chart_data(b_df, i_df, 'Facility_capacity_ideal', n_months, n_runs)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        def line_chart_ci(line_data, title, ytitle, ydomain):
                            chart = (
                                    alt.Chart(line_data, title=title)
                                    .mark_line()
                                    .encode(
                                        x=alt.X("Month:Q", axis=alt.Axis(
                                            title="Time since the start of intervention implementation (Months)",
                                            titleFontSize=16, labelFontSize=14, )
                                                ),
                                        y=alt.Y("Rate:Q", axis=alt.Axis(
                                            title=ytitle,
                                            titleFontSize=16, labelFontSize=14),
                                                scale=alt.Scale(domain=ydomain)
                                                ),
                                        color=alt.Color("Scenario", legend=alt.Legend(title="Scenario", titleFontSize=16,
                                                                                   labelFontSize=14)),
                                        tooltip=["Month:N", "Scenario:N", "Rate:Q", "Lower_rate:Q", "Upper_rate:Q"]
                                    )
                                    + alt.Chart(line_data)
                                    .mark_area(opacity=0.2)
                                    .encode(
                                x="Month:Q",
                                y=alt.Y("Lower_rate:Q", axis=alt.Axis(title=ytitle, titleFontSize=16, labelFontSize=14)),
                                y2="Upper_rate:Q",
                                color=alt.Color("Scenario", legend=None),  # Match color to Scenario
                            )
                            )

                            chart = chart.properties(width=700, height=400).interactive()
                            chart = chart.configure_title(
                                anchor='middle', fontSize=18
                            )
                            return chart
                        chart = line_chart_ci(line_data, "Facility Capacity Ratio", "Capacity ratio in L4/5 facilities", [0,1])
                        st.altair_chart(chart)
                        # chart = (
                        #     alt.Chart(df_cap_ratio)
                        #     .mark_line()
                        #     .encode(
                        #         x=alt.X("Month:Q", title="Time since the start of intervention implementation (Months)"),
                        #         y=alt.Y("Capacity Ratio:Q", axis=alt.Axis(title="Capacity ratio in L4/5 facilities")),
                        #         color=alt.Color("Scenario:N", legend=alt.Legend(title="Scenario")),
                        #         tooltip=["Month:N", "Scenario:N", "Capacity Ratio:Q"]
                        #     )
                        #     .properties(width=600, height=400)
                        #     .interactive()
                        # )
                        # chart = chart.properties(
                        #     title=alt.TitleParams(text="Facility Capacity Ratio", anchor="middle")
                        # )
                        # st.altair_chart(chart)
                        #
                        #
                        # def facility_capacity_ratio_matplotlib(df_cap_ratio):
                        #     fig, ax = plt.subplots(figsize=(8, 5))  # Define figure size
                        #
                        #     # Get unique scenarios and assign colors
                        #     scenarios = df_cap_ratio["Scenario"].unique()
                        #     colors = sns.color_palette("tab10", len(scenarios))
                        #
                        #     # Plot each scenario without markers
                        #     for i, scenario in enumerate(scenarios):
                        #         subset = df_cap_ratio[df_cap_ratio["Scenario"] == scenario]
                        #         ax.plot(subset["Month"], subset["Capacity Ratio"], label=scenario, color=colors[i],
                        #                 linewidth=2)
                        #
                        #     # Labels and Title
                        #     ax.set_xlabel("Time since the start of intervention implementation (Months)", fontsize=14)
                        #     ax.set_ylabel("Capacity Ratio in L4/5 Facilities", fontsize=14)
                        #     ax.set_title("Facility Capacity Ratio", fontsize=16)
                        #
                        #     # Move legend to upper right
                        #     ax.legend(title="Scenario", fontsize=12, loc="upper right", bbox_to_anchor=(1, 0.95))
                        #
                        #     # Grid, Formatting, and Layout
                        #     ax.grid(True, linestyle="--", alpha=0.5)
                        #     plt.xticks(fontsize=12)
                        #     plt.yticks(fontsize=12)
                        #     plt.tight_layout()
                        #
                        #     return fig

                        # fig = facility_capacity_ratio_matplotlib(df_cap_ratio)
                        # st.pyplot(fig)  # Show plot in Streamlit
                        #
                        # # Enable Download of the Chart
                        # img_buffer = io.BytesIO()
                        # fig.savefig(img_buffer, format="png", dpi=300)
                        # st.download_button(label="Download Facility Capacity Ratio Plot", data=img_buffer.getvalue(),
                        #                    file_name="facility_capacity_ratio.png", mime="image/png",
                        #                    key="download_facility_capacity")

                    ymax = max(df_fac_cap_actual['Facility_capacity_actual'].max(), df_fac_cap_ideal['Facility_capacity_ideal'].max())
                    with col2:
                        chart = (
                            alt.Chart(df_fac_cap_actual)
                            .mark_line()
                            .encode(
                                x=alt.X("Month:Q", title="Time since the start of intervention implementation (Months)"),
                                y=alt.Y("Facility_capacity_actual:Q", axis=alt.Axis(title="Live births"), scale=alt.Scale(domain=[0, ymax])),
                                color=alt.Color("Scenario:N", legend=alt.Legend(title="Scenario")),
                                tooltip=["Month:N", "Scenario:N", "Facility_capacity_actual:Q"]
                            )
                            .properties(width=600, height=400)
                            .interactive()
                        )
                        chart = chart.properties(
                            title=alt.TitleParams(text="Actual Facility Capacity", anchor="middle")
                        )
                        st.altair_chart(chart)

                    with col3:
                        chart = (
                            alt.Chart(df_fac_cap_ideal)
                            .mark_line()
                            .encode(
                                x=alt.X("Month:Q", title="Time since the start of intervention implementation (Months)"),
                                y=alt.Y("Facility_capacity_ideal:Q", axis=alt.Axis(title="Live births"), scale=alt.Scale(domain=[0, ymax])),
                                color=alt.Color("Scenario:N", legend=alt.Legend(title="Scenario")),
                                tooltip=["Month:N", "Scenario:N", "Facility_capacity_ideal:Q"]
                            )
                            .properties(width=600, height=400)
                            .interactive()
                        )
                        chart = chart.properties(
                            title=alt.TitleParams(text="Ideal Facility Capacity", anchor="middle")
                        )
                        st.altair_chart(chart)

                if selected_plot == "C-section rate":
                    st.markdown("<h3 style='text-align: left;'>C-sections among 100 live births</h3>",
                                unsafe_allow_html=True)

                    tab1, tab2, tab3 = st.tabs(["Emergency C-sections", "Elective C-sections", "All C-sections"])

                    def prepare_indicator_df(df, indicator, n_months, n_runs, scenario):
                        indicator = np.concatenate(df[indicator].values).reshape(-1, 4)
                        df_indicator = pd.DataFrame(indicator, columns=['Home', 'L2/3', 'L4', 'L5'])
                        df_indicator['L4/5'] = df_indicator['L4'] + df_indicator['L5']
                        df_indicator = df_indicator.drop(columns=['Home', 'L4', 'L5'])
                        df_indicator['All'] = np.sum(df_indicator, axis=1)

                        df_indicator['Month'] = np.tile(np.arange(1, n_months+1), n_runs)
                        df_indicator['Run'] = np.repeat(np.arange(n_runs), n_months)
                        df_indicator['Scenario'] = scenario
                        #df_indicator = df_indicator.groupby(['Month', 'Scenario'], as_index=False).sum()
                        df_indicator = df_indicator.melt(id_vars=['Month', 'Run', 'Scenario'], var_name='Level',
                                                         value_name='Counts')
                        return df_indicator

                    def prepare_elecCS_df(df, indicator, n_months, n_runs, scenario):
                        indicator = np.concatenate(df[indicator].values).reshape(-1, 2)
                        df_indicator = pd.DataFrame(indicator, columns=['Lowrisk', 'Highrisk'])
                        df_indicator['All'] = np.sum(df_indicator, axis=1)
                        df_indicator['Month'] = np.tile(np.arange(1, n_months+1), n_runs)
                        df_indicator['Run'] = np.repeat(np.arange(n_runs), n_months)
                        df_indicator['Scenario'] = scenario
                        #df_indicator = df_indicator.groupby(['Month', 'Scenario'], as_index=False).sum()
                        df_indicator = df_indicator.melt(id_vars=['Month', 'Run', 'Scenario'], var_name='Level',
                                                         value_name='Counts')
                        return df_indicator

                    def prepare_elecCS_data(b_df, i_df, indicator, n_months, n_runs, multiplier):
                        b_df_ind = prepare_elecCS_df(b_df, indicator, n_months, n_runs, 'Baseline')
                        i_df_ind = prepare_elecCS_df(i_df, indicator, n_months, n_runs, 'Intervention')
                        df_ind = pd.concat([b_df_ind, i_df_ind], ignore_index=True)

                        b_df_lb = prepare_elecCS_df(b_df, 'Risk status', n_months, n_runs, 'Baseline')
                        i_df_lb = prepare_elecCS_df(i_df, 'Risk status', n_months, n_runs, 'Intervention')
                        df_lb = pd.concat([b_df_lb, i_df_lb], ignore_index=True)

                        df_ind = df_ind.merge(df_lb, on=['Month', 'Run', 'Scenario', 'Level'], suffixes=('_ind', '_lb'))
                        df_ind.columns = ['Month', 'Run', 'Scenario', 'Level', 'Counts', 'Denominator']
                        df_ind['Rate'] = df_ind['Counts'] / df_ind['Denominator'] * multiplier
                        return df_ind

                    def create_bar_data_type(data, multiplier):
                        bar_data = data[data['Month'] > n_months - 12].copy()

                        # bar_data = bar_data.groupby(['Scenario', 'Level', 'Type'], as_index=False).sum()
                        # bar_data['Rate'] = bar_data['Counts'] / bar_data['Denominator'] * multiplier
                        # bar_data = bar_data.apply(add_poisson_ci, axis=1, multiplier=multiplier)
                        # bar_data = bar_data.apply(
                        #     lambda x: round(x, 2) if x.name in ['Rate', 'Lower_rate', 'Upper_rate'] else x)

                        # Step 1: Add Poisson CI per run × month × level
                        bar_data = add_poisson_ci(bar_data, multiplier=multiplier)

                        # Step 2: Taking average across runs and scenarios
                        bar_data = bar_data.groupby(['Scenario', 'Level', 'Type'], as_index=False).agg({
                            'Rate': 'mean',
                            'Lower_rate': 'mean',
                            'Upper_rate': 'mean'
                        })

                        # Step 3: Round for display
                        bar_data[['Rate', 'Lower_rate', 'Upper_rate']] = bar_data[
                            ['Rate', 'Lower_rate', 'Upper_rate']].round(2)
                        return bar_data

                    def bar_chart_type(data, title, ymax):
                        chart = (
                            alt.Chart(data)
                            .mark_bar()
                            .encode(
                                x=alt.X("Scenario:N", axis=alt.Axis(title=None)),  # X-axis for Scenario
                                y=alt.Y("Rate:Q", axis=alt.Axis(title="Rate"), scale=alt.Scale(domain=[0, ymax])),
                                color=alt.Color("Type:N", legend=alt.Legend(title="Type")),  # Color by Scenario
                                tooltip=["Scenario:N", "Level:N", "Type:N", "Rate:Q", "Lower_rate:Q", "Upper_rate:Q"]
                            )
                        ).properties(width=150, height=300)  # Set width and height for each column

                        chart = chart.facet(
                            column=alt.Column(
                                "Level:N",  # Facet by Level
                                title=None,  # Remove column title
                                header=alt.Header(labelOrient="top", labelFontSize=12)  # Customize header
                            )
                        ).configure_title(anchor="middle")  # Center-align the title

                        chart = chart.properties(
                            title=alt.TitleParams(text=title, anchor="middle")
                        ).interactive()
                        return chart

                    def create_pie_chart(data, scenario, title, color, width=300, height=500):
                        # Filter data for the given scenario
                        filtered_data = data[data['Scenario'] == scenario]

                        # Calculate the proportion for each type
                        total_rate = filtered_data['Rate'].sum()
                        filtered_data['Proportion'] = round(filtered_data['Rate'] / total_rate * 100)

                        # Create the pie chart
                        chart = (
                            alt.Chart(filtered_data)
                            .mark_arc()
                            .encode(
                                theta='Rate:Q',
                                color=color,
                                tooltip=[color, "Rate:Q", "Proportion:Q"]
                            )
                        )

                        # Combine the chart and text labels
                        chart = chart.properties(
                            width=width,
                            height=height,
                            title=alt.TitleParams(text=title, anchor="middle")
                        ).interactive()

                        # Display the chart in Streamlit
                        st.altair_chart(chart)

                    with tab2:
                        df_elec_CS = prepare_elecCS_data(b_df, i_df, 'Elective CS risk status', n_months, n_runs, 100)

                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_elec_CS, 100)
                            ymax = line_data['Upper_rate'].max()
                            chart = line_chart_ci(line_data, "Elective CS rate among all live births by month", "Rate", [0, ymax])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_elec_CS, 100)
                            chart = bar_chart_ci(bar_data, "Elective CS rate among all live births annually", "Rate", [0, ymax])
                            st.altair_chart(chart)

                        st.markdown("~~~")
                        col3, col4 = st.columns(2)
                        pie_data = bar_data[bar_data['Level'] != 'All']
                        with col3:
                            create_pie_chart(pie_data, 'Baseline', "Baseline", 'Level:N')
                        with col4:
                            create_pie_chart(pie_data, 'Intervention', "Intervention", 'Level:N')


                    with tab1:
                        df_CS_all = prepare_chart_data(b_df, i_df, 'Emergency CS', 'Live Births Final', n_months, n_runs, 100)
                        df_CS_all['Type'] = "All"
                        df_CS_unnecessary = prepare_chart_data(b_df, i_df, 'CS_unnessary', 'Live Births Final', n_months, n_runs, 100)
                        df_CS_unnecessary['Type'] = "Unnecessary"
                        df_CS_necessary = df_CS_all.copy()
                        df_CS_necessary['Counts'] = df_CS_all['Counts'] - df_CS_unnecessary['Counts']
                        df_CS_necessary['Rate'] = df_CS_necessary['Counts'] / df_CS_necessary['Denominator'] * 100
                        df_CS_necessary['Type'] = "Necessary"
                        df_CS_type = pd.concat([df_CS_unnecessary, df_CS_necessary], ignore_index=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_CS_all, 100)
                            ymax = line_data['Rate'].max()
                            chart = line_chart_ci(line_data, "Emergency CS rate by month", "Rate", [0, ymax])
                            st.altair_chart(chart)

                        with col2:
                            bar_data_type = create_bar_data_type(df_CS_type, 100)
                            chart = bar_chart_type(bar_data_type, "Emergency CS rate by type", ymax)
                            st.altair_chart(chart)

                        st.markdown("~~~")
                        col3, col4 = st.columns(2)
                        pie_data_type = bar_data_type[bar_data_type['Level'] == 'All']

                        # Assuming col3 and col4 are defined and pie_data_type is already defined
                        with col3:
                            create_pie_chart(pie_data_type, 'Baseline', "Baseline", 'Type:N')
                        with col4:
                            create_pie_chart(pie_data_type, 'Intervention', "Intervention", 'Type:N')

                    with tab3:
                        df_CS_all = prepare_chart_data(b_df, i_df, 'CS', 'Live Births Final', n_months, n_runs, 100)
                        df_CS_elective = prepare_chart_data(b_df, i_df, 'Elective CS', 'Live Births Final', n_months, n_runs, 100)
                        df_CS_elective['Type'] = "Elective"
                        df_CS_emergency = prepare_chart_data(b_df, i_df, 'Emergency CS', 'Live Births Final', n_months, n_runs, 100)
                        df_CS_emergency['Type'] = "Emergency"

                        df_CS_type = pd.concat([df_CS_elective, df_CS_emergency], ignore_index=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_CS_all, 100)
                            ymax = line_data['Rate'].max()
                            chart = line_chart_ci(line_data, "CS rate among facility live births by month", "Rate", [0, ymax])
                            st.altair_chart(chart)
                        with col2:
                            bar_data_type = create_bar_data_type(df_CS_type, 100)
                            chart = bar_chart_type(bar_data_type, "CS rate among facility live births by type", ymax)
                            st.altair_chart(chart)

                        st.markdown("~~~")
                        col3, col4 = st.columns(2)
                        pie_data_type = bar_data_type[bar_data_type['Level'] == 'All']

                        # Assuming col3 and col4 are defined and pie_data_type is already defined
                        with col3:
                            create_pie_chart(pie_data_type, 'Baseline', "Baseline", 'Type:N')
                        with col4:
                            create_pie_chart(pie_data_type, 'Intervention', "Intervention", 'Type:N')

                if selected_plot == "Distribution of live births":
                    st.markdown("<h3 style='text-align: left;'>% Live Births Delivered at Location</h3>",
                                unsafe_allow_html=True)

                    def prepare_plot_data(b_df, i_df, n_months, n_runs):
                        bLB = np.concatenate(b_df['Live Births Final'].values).reshape(-1, 4)
                        iLB = np.concatenate(i_df['Live Births Final'].values).reshape(-1, 4)

                        b_df = pd.DataFrame(bLB, columns=['Home', 'L2/3', 'L4', 'L5'])
                        i_df = pd.DataFrame(iLB, columns=['Home', 'L2/3', 'L4', 'L5'])

                        for df in [b_df, i_df]:
                            df_all = np.sum(df, axis=1)
                            df['All'] = df_all
                            df['L4/5'] = df['L4'] + df['L5']
                            df['Month'] = np.tile(np.arange(1, n_months+1), n_runs)
                            df['Run'] = np.repeat(np.arange(n_runs), n_months)

                        b_df['Scenario'] = 'Baseline'
                        i_df['Scenario'] = 'Intervention'
                        b_df = b_df.drop(columns=['L4', 'L5'])
                        i_df = i_df.drop(columns=['L4', 'L5'])

                        combined_df = pd.concat([b_df, i_df], ignore_index=True)
                        combined_df = combined_df.melt(id_vars=['Month', 'Run', 'Scenario'], var_name='Level', value_name='Counts')
                        #combined_df = combined_df.groupby(['Month', 'Scenario', 'Level'], as_index=False).sum()
                        return combined_df

                    df_LB = prepare_plot_data(b_df, i_df, n_months, n_runs)
                    df_LB_level = df_LB[df_LB['Level'] != 'All']
                    df_LB_all = df_LB[df_LB['Level'] == 'All']

                    data = pd.merge(df_LB_level, df_LB_all, on=['Month', 'Run', 'Scenario'], how='left')
                    data['Rate'] = data['Counts_x'] / data['Counts_y'] * 100
                    data = data.drop(columns = ['Level_y'])
                    data.columns = ['Month', 'Run', 'Scenario', 'Level', 'Counts', 'Denominator', 'Rate']

                    col1, col2 = st.columns(2)
                    with col1:
                        line_data = create_line_data(data, 100)
                        chart = line_chart_ci(line_data,"% Live births delivered at location by month", "Percentage", [0, 100])
                        st.altair_chart(chart)

                        # fig1 = line_chart_ci_matplotlib(line_data, "% Live Births Delivered at Location by Month", "Percentage",
                        #                                 [0, 100])
                        # st.pyplot(fig1)  # Display plot in Streamlit
                        #
                        # # Enable download of the line chart
                        # img_buffer1 = io.BytesIO()
                        # fig1.savefig(img_buffer1, format="png", dpi=300)
                        # st.download_button(label="Download Line Chart", data=img_buffer1.getvalue(),
                        #                    file_name="line_chart.png", mime="image/png", key="download_line_lbs")

                    with col2:
                        bar_data = create_bar_data(data, 100)
                        chart = bar_chart_ci(bar_data, "% Live births delivered at location annually", "Percentage", [0, 100])
                        st.altair_chart(chart)

                if selected_plot == "ANC rate":
                    st.markdown("<h3 style='text-align: left;'>4+ANC visits per 100 live births</h3>",
                                unsafe_allow_html=True)
                    df_ANC = prepare_chart_data(b_df, i_df, 'ANC', 'Live Births Final', n_months, n_runs, 100)
                    col1, col2 = st.columns(2)
                    with col1:
                        line_data = create_line_data(df_ANC, 100)
                        chart = line_chart_ci(line_data, "4+ANC rate by month", "Rate", [0, 100])
                        st.altair_chart(chart)
                    with col2:
                        bar_data = create_bar_data(df_ANC, 100)
                        chart = bar_chart_ci(bar_data, "4+ANC rate annually", "Rate", [0, 100])
                        st.altair_chart(chart)

                if selected_plot == "Maternal mortality rate":
                    st.markdown("<h3 style='text-align: left;'>Maternal deaths per 100,000 live births (MMR)</h3>",
                               unsafe_allow_html=True)
                    tab1, tab2, tab3 = st.tabs(
                        ["MMR by location", "Distribution of Causes", "Death by cause"]
                    )

                    with tab1:

                        df_MMR = prepare_chart_data(b_df, i_df, 'Deaths', 'Live Births Final', n_months, n_runs, 100000)

                        # Get unique levels and create a multiselect box
                        all_levels = df_MMR['Level'].unique().tolist()
                        selected_levels = st.multiselect("Select Delivery Location Level(s) to Show", options=all_levels,
                                                         default=all_levels)

                        # Filter the data based on selected levels
                        df_MMR = df_MMR[df_MMR["Level"].isin(selected_levels)]

                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_MMR, 100000)
                            ymax = line_data['Upper_rate'].max()
                            chart = line_chart_ci(line_data, "MMR by month", "MMR (deaths per 100,000 live births)", [0, ymax])

                            st.altair_chart(chart)

                        with col2:
                            bar_data = create_bar_data(df_MMR, 100000)
                            ymax = bar_data['Upper_rate'].max()
                            chart = bar_chart_ci(bar_data, "MMR annually", "MMR (deaths per 100,000 live births)", [0, ymax])

                            st.altair_chart(chart)

                            # fig2 = bar_chart_ci_matplotlib(bar_data, "MMR by Delivery Location",
                            #                               "MMR (deaths per 100,000 live births)", [0, 1000])
                            # st.pyplot(fig2)
                            # # Enable download of the bar chart
                            # img_buffer2 = io.BytesIO()
                            # fig2.savefig(img_buffer2, format="png", dpi=300)
                            # st.download_button(label="Download Bar Chart", data=img_buffer2.getvalue(),
                            #                    file_name="bar_chart.png", mime="image/png", key="download_bar")

                    with tab2:
                        b_ind_outcomes = st.session_state['b_ind_outcomes']
                        i_ind_outcomes = st.session_state['i_ind_outcomes']
                        data1 = b_ind_outcomes[["Run", "Scenario", "death_cause"]]
                        data2 = i_ind_outcomes[["Run", "Scenario", "death_cause"]]
                        df = pd.concat([data1, data2])
                        df_filtered = df[df["death_cause"] != "none"]

                        death_counts = df_filtered.groupby(["Scenario", "death_cause"]).size().reset_index(name="count")
                        death_counts["proportion"] = death_counts.groupby("Scenario")["count"].transform(
                            lambda x: x / x.sum())

                        chart = alt.Chart(death_counts).mark_bar().encode(
                            x=alt.X("Scenario:N", title="Scenario"),
                            y=alt.Y("proportion:Q", stack="normalize", title="Proportion of Deaths"),
                            color=alt.Color("death_cause:N", title="Cause of Death"),
                            tooltip=["Scenario", "death_cause", alt.Tooltip("proportion:Q", format=".1%")]
                        ).properties(
                            width=600,
                            height=400,
                            title="Proportion of Maternal Death Causes by Scenario"
                        )

                        st.altair_chart(chart)

                    with tab3:
                        ref_name = (
                            st.session_state.reference_label
                            if st.session_state.compare_two_interventions
                            else "Baseline"
                        )
                        tgt_name = (
                            st.session_state.target_label
                            if st.session_state.compare_two_interventions
                            else "Intervention"
                        )
                        st.caption(
                            f"Bar base: **{ref_name}** maternal deaths by cause (mean count across runs when applicable). "
                            f"Stacked segment: absolute change from {ref_name} to **{tgt_name}**. "
                            f"Labels above bars show percent change."
                        )
                        cause_chart = death_by_cause_comparison_chart(
                            b_ind_outcomes, i_ind_outcomes, ref_name, tgt_name
                        )
                        if cause_chart is None:
                            st.info("No attributed maternal deaths in the current run results.")
                        else:
                            st.altair_chart(cause_chart, use_container_width=True)


                if selected_plot == "Severe maternal outcomes":
                    st.markdown("<h3 style='text-align: left;'>Severe maternal outcomes, including near misses and maternal deaths</h3>",
                                unsafe_allow_html=True)

                    tab1, tab2 = st.tabs(["SMO rate", "Death rate among SMOs"])

                    with tab1:
                        df_severe_comps = prepare_chart_data(b_df, i_df, 'severe_comps', 'Live Births Final', n_months, n_runs, 100)
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_severe_comps, 100)
                            ymax = line_data['Rate'].max()
                            chart = line_chart_ci(line_data, "Severe complications rate by month", "Rate", [0, ymax])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_severe_comps, 100)
                            chart = bar_chart_ci(bar_data, "Severe complications rate annually", "Rate", [0, ymax])
                            st.altair_chart(chart)

                    with tab2:
                        df_death_severe = prepare_chart_data(b_df, i_df, 'Deaths', 'severe_comps', n_months, n_runs, 100)
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_death_severe, 100)
                            ymax = line_data['Rate'].max()
                            chart = line_chart_ci(line_data, "Death rate among severe complications by month", "Rate", [0, ymax])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_death_severe, 100)
                            ymax = bar_data['Rate'].max()
                            chart = bar_chart_ci(bar_data, "Death rate among severe complications annually", "Rate", [0, ymax])
                            st.altair_chart(chart)


                if selected_plot == "Maternal complication rate":
                    st.markdown("<h3 style='text-align: left;'>Complications per 100 live births</h3>",
                                unsafe_allow_html=True)

                    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["Death-related", "Anemia", "Prolonged labor", "PPH", "Sepsis", "Eclampsia", "Obstructed labor", "Ruptured uterus", "APH"])

                    with tab1:
                        df_MCR = prepare_chart_data(b_df, i_df, "Comps after transfer", 'Live Births Final', n_months, n_runs, 100)
                        # Get unique levels and create a multiselect box
                        all_levels = df_MCR['Level'].unique().tolist()
                        selected_levels = st.multiselect("Select Delivery Location Level(s) to Show", options=all_levels,
                                                         default=all_levels, key="mcr_level_multiselect")

                        # Filter the data based on selected levels
                        df_MCR = df_MCR[df_MCR["Level"].isin(selected_levels)]
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_MCR, 100)
                            ymax = line_data['Upper_rate'].max()
                            chart = line_chart_ci(line_data, "Complications rate by month", "Rate", [0, ymax])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_MCR, 100)
                            chart = bar_chart_ci(bar_data, "Complications rate annually", "Rate", [0, ymax])
                            st.altair_chart(chart)
                        ## MEIBIN ALTERED ##
                        st.text(f'complications: {i_df["severe_comps"]}')
                        st.text(f'complications: {b_df["severe_comps"]}')
                    with tab2:
                        df_anemia = prepare_chart_data(b_df, i_df, 'Anemia', 'Live Births Final', n_months, n_runs, 100)
                        # Get unique levels and create a multiselect box
                        all_levels = df_anemia['Level'].unique().tolist()
                        selected_levels = st.multiselect("Select Delivery Location Level(s) to Show", options=all_levels,
                                                         default=all_levels, key="anemia_level_multiselect")

                        # Filter the data based on selected levels
                        df_anemia = df_anemia[df_anemia["Level"].isin(selected_levels)]
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_anemia, 100)
                            ymax = line_data['Upper_rate'].max()
                            chart = line_chart_ci(line_data, "Anemia rate by month", "Rate", [0, ymax])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_anemia, 100)
                            ymax = bar_data['Upper_rate'].max()
                            chart = bar_chart_ci(bar_data, "Anemia rate annually", "Rate", [0, ymax])
                            st.altair_chart(chart)

                    with tab3:
                        df_PL = prepare_chart_data(b_df, i_df, 'PL', 'Live Births Final', n_months, n_runs, 100)
                        # Get unique levels and create a multiselect box
                        all_levels = df_PL['Level'].unique().tolist()
                        selected_levels = st.multiselect("Select Delivery Location Level(s) to Show", options=all_levels,
                                                         default=all_levels, key="PL_level_multiselect")

                        # Filter the data based on selected levels
                        df_PL = df_PL[df_PL["Level"].isin(selected_levels)]
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_PL, 100)
                            ymax = line_data['Upper_rate'].max()
                            chart = line_chart_ci(line_data, "Prolonged labor rate by month", "Rate", [0, ymax])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_PL, 100)
                            ymax = bar_data['Upper_rate'].max()
                            chart = bar_chart_ci(bar_data, "Prolonged labor rate annually", "Rate", [0, ymax])
                            st.altair_chart(chart)

                    with tab4:
                        df_PPH = prepare_chart_data(b_df, i_df, 'pph', 'Live Births Final', n_months, n_runs, 100)
                        # Get unique levels and create a multiselect box
                        all_levels = df_PPH['Level'].unique().tolist()
                        selected_levels = st.multiselect("Select Delivery Location Level(s) to Show", options=all_levels,
                                                         default=all_levels, key="PPH_level_multiselect")

                        # Filter the data based on selected levels
                        df_PPH = df_PPH[df_PPH["Level"].isin(selected_levels)]
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_PPH, 100)
                            ymax = line_data['Upper_rate'].max()
                            chart = line_chart_ci(line_data, "Postpartum hemorrhage rate by month", "Rate", [0, ymax])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_PPH, 100)
                            ymax = bar_data['Upper_rate'].max()
                            chart = bar_chart_ci(bar_data, "Postpartum hemorrhage rate annually", "Rate", [0, ymax])
                            st.altair_chart(chart)

                    with tab5:
                        df_sepsis = prepare_chart_data(b_df, i_df, 'mat_sepsis', 'Live Births Final', n_months, n_runs, 100)
                        # Get unique levels and create a multiselect box
                        all_levels = df_sepsis['Level'].unique().tolist()
                        selected_levels = st.multiselect("Select Delivery Location Level(s) to Show", options=all_levels,
                                                         default=all_levels, key="sepsis_level_multiselect")

                        # Filter the data based on selected levels
                        df_sepsis = df_sepsis[df_sepsis["Level"].isin(selected_levels)]
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_sepsis, 100)
                            ymax = line_data['Upper_rate'].max()
                            chart = line_chart_ci(line_data, "Sepsis rate by month", "Rate", [0, ymax])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_sepsis, 100)
                            ymax = bar_data['Upper_rate'].max()
                            chart = bar_chart_ci(bar_data, "Sepsis rate annually", "Rate", [0, ymax])
                            st.altair_chart(chart)

                    with tab6:
                        df_eclampsia = prepare_chart_data(b_df, i_df, 'eclampsia', 'Live Births Final', n_months, n_runs, 100)
                        # Get unique levels and create a multiselect box
                        all_levels = df_eclampsia['Level'].unique().tolist()
                        selected_levels = st.multiselect("Select Delivery Location Level(s) to Show", options=all_levels,
                                                         default=all_levels, key="eclampsia_level_multiselect")

                        # Filter the data based on selected levels
                        df_eclampsia = df_eclampsia[df_eclampsia["Level"].isin(selected_levels)]
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_eclampsia, 100)
                            ymax = line_data['Upper_rate'].max()
                            chart = line_chart_ci(line_data, "Eclampsia rate by month", "Rate", [0, ymax])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_eclampsia, 100)
                            ymax = bar_data['Upper_rate'].max()
                            chart = bar_chart_ci(bar_data, "Eclampsia rate annually", "Rate", [0, ymax])
                            st.altair_chart(chart)

                    with tab7:
                        df_obstructed = prepare_chart_data(b_df, i_df, 'OL', 'Live Births Final', n_months, n_runs, 100)
                        # Get unique levels and create a multiselect box
                        all_levels = df_obstructed['Level'].unique().tolist()
                        selected_levels = st.multiselect("Select Delivery Location Level(s) to Show", options=all_levels,
                                                         default=all_levels, key="ol_level_multiselect")

                        # Filter the data based on selected levels
                        df_obstructed = df_obstructed[df_obstructed["Level"].isin(selected_levels)]
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_obstructed, 100)
                            ymax = line_data['Upper_rate'].max()
                            chart = line_chart_ci(line_data, "Obstructed labor rate by month", "Rate", [0, ymax])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_obstructed, 100)
                            ymax = bar_data['Upper_rate'].max()
                            chart = bar_chart_ci(bar_data, "Obstructed labor rate annually", "Rate", [0, ymax])
                            st.altair_chart(chart)

                        # bar_data_number = create_bar_data(df_obstructed, 1)
                        # bar_data_number['Counts'] = bar_data_number['Counts'] / n_runs
                        # bar_data_number['Counts'] = bar_data_number['Counts'].astype(int)
                        #
                        # def bar_chart_num(bar_data, title, ytitle, ydomain):
                        #     layered_chart = (
                        #             alt.Chart(bar_data)
                        #             .mark_bar()
                        #             .encode(
                        #                 x=alt.X("Scenario:N", axis=None),  # X-axis for Scenario
                        #                 y=alt.Y("Counts:Q", axis=alt.Axis(title=ytitle), scale=alt.Scale(domain=ydomain)),
                        #                 color=alt.Color("Scenario:N", legend=alt.Legend(title="Scenario")),
                        #                 # Color by Scenario
                        #                 tooltip=["Scenario:N", "Level:N", "Counts:Q"]
                        #             )
                        #     ).properties(width=150, height=300)  # Set width and height for each column
                        #
                        #     chart = layered_chart.facet(
                        #         column=alt.Column(
                        #             "Level:N",  # Facet by Level
                        #             title=None,  # Remove column title
                        #             header=alt.Header(labelOrient="bottom", labelFontSize=12)  # Customize header
                        #         )
                        #     ).configure_title(anchor="middle")  # Center-align the title
                        #
                        #     chart = chart.properties(
                        #         title=alt.TitleParams(text=title, anchor="middle")
                        #     ).interactive()
                        #     return chart
                        # ymax = bar_data_number['Counts'].max()
                        # chart = bar_chart_num(bar_data_number, "Number of obstructed labor annually", "Count", [0, ymax])
                        # st.altair_chart(chart)

                    with tab8:
                        df_ruptured_uterus = prepare_chart_data(b_df, i_df, 'ruptured_uterus', 'Live Births Final', n_months, n_runs, 100)
                        # Get unique levels and create a multiselect box
                        all_levels = df_ruptured_uterus['Level'].unique().tolist()
                        selected_levels = st.multiselect("Select Delivery Location Level(s) to Show", options=all_levels,
                                                         default=all_levels, key="ruptured_uterus_level_multiselect")

                        # Filter the data based on selected levels
                        df_ruptured_uterus = df_ruptured_uterus[df_ruptured_uterus["Level"].isin(selected_levels)]
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_ruptured_uterus, 100)
                            ymax = line_data['Upper_rate'].max()
                            chart = line_chart_ci(line_data, "Ruptured uterus rate by month", "Rate", [0, ymax])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_ruptured_uterus, 100)
                            ymax = bar_data['Upper_rate'].max()
                            chart = bar_chart_ci(bar_data, "Ruptured uterus rate annually", "Rate", [0, ymax])
                            st.altair_chart(chart)

                    with tab9:
                        df_aph = prepare_chart_data(b_df, i_df, 'aph', 'Live Births Final', n_months, n_runs, 100)
                        # Get unique levels and create a multiselect box
                        all_levels = df_aph['Level'].unique().tolist()
                        selected_levels = st.multiselect("Select Delivery Location Level(s) to Show", options=all_levels,
                                                         default=all_levels, key="aph_level_multiselect")

                        # Filter the data based on selected levels
                        df_aph = df_aph[df_aph["Level"].isin(selected_levels)]
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_aph, 100)
                            ymax = line_data['Upper_rate'].max()
                            chart = line_chart_ci(line_data, "Antepartum hemorrhage rate by month", "Rate", [0, ymax])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_aph, 100)
                            ymax = bar_data['Upper_rate'].max()
                            chart = bar_chart_ci(bar_data, "Antepartum hemorrhage rate annually", "Rate", [0, ymax])
                            st.altair_chart(chart)

                if selected_plot == "Preterm rate":
                    st.markdown("<h3 style='text-align: left;'>Preterm births per 100 live births</h3>",
                                unsafe_allow_html=True)

                    df_PT = prepare_chart_data(b_df, i_df, 'Preterm', 'Live Births Final', n_months, n_runs, 100)

                    col1, col2 = st.columns(2)
                    with col1:
                        line_data = create_line_data(df_PT, 100)
                        chart = line_chart_ci(line_data, "Preterm rate by month", "Rate", [0, 50])
                        st.altair_chart(chart)

                    with col2:
                        bar_data = create_bar_data(df_PT, 100)
                        chart = bar_chart_ci(bar_data, "Preterm rate annually", "Rate", [0, 50])
                        st.altair_chart(chart)


                if selected_plot == "Neonatal complication rate":
                    st.markdown("<h3 style='text-align: left;'>Neonatal complications per 100 live births</h3>",
                                unsafe_allow_html=True)

                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["RDS", "IVH", "Sepsis", "NEC", "hypoxia", "asphyxia"])
                    with tab1:
                        df_RDS = prepare_chart_data(b_df, i_df, 'RDS', 'Live Births Final', n_months, n_runs, 100)
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_RDS, 100)
                            chart = line_chart_ci(line_data, "RDS rate by month", "Rate", [0, 6])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_RDS, 100)
                            chart = bar_chart_ci(bar_data, "RDS rate annually", "Rate", [0, 6])
                            st.altair_chart(chart)
                    with tab2:
                        df_IVH = prepare_chart_data(b_df, i_df, 'IVH', 'Live Births Final', n_months, n_runs, 100)
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_IVH, 100)
                            chart = line_chart_ci(line_data, "IVH rate by month", "Rate", [0, 4])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_IVH, 100)
                            chart = bar_chart_ci(bar_data, "IVH rate annually", "Rate", [0, 4])
                            st.altair_chart(chart)
                    with tab3:
                        df_Sepsis = prepare_chart_data(b_df, i_df, 'neo_sepsis', 'Live Births Final', n_months, n_runs, 100)
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_Sepsis, 100)
                            chart = line_chart_ci(line_data, "Sepsis rate by month", "Rate", [0, 5])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_Sepsis, 100)
                            chart = bar_chart_ci(bar_data, "Sepsis rate annually", "Rate", [0, 5])
                            st.altair_chart(chart)
                    with tab4:
                        df_NEC = prepare_chart_data(b_df, i_df, 'NEC', 'Live Births Final', n_months, n_runs, 100)
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_NEC, 100)
                            chart = line_chart_ci(line_data, "NEC rate by month", "Rate", [0, 1])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_NEC, 100)
                            chart = bar_chart_ci(bar_data, "NEC rate annually", "Rate", [0, 1])
                            st.altair_chart(chart)
                    with tab5:
                        df_hypoxia = prepare_chart_data(b_df, i_df, 'hypoxia', 'Live Births Final', n_months, n_runs, 100)
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_hypoxia, 100)
                            chart = line_chart_ci(line_data, "Hypoxia rate by month", "Rate", [0, 12])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_hypoxia, 100)
                            chart = bar_chart_ci(bar_data, "Hypoxia rate annually", "Rate", [0, 12])
                            st.altair_chart(chart)
                    with tab6:
                        df_asphyxia = prepare_chart_data(b_df, i_df, 'asphyxia', 'Live Births Final', n_months, n_runs, 100)
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_asphyxia, 100)
                            chart = line_chart_ci(line_data, "Asphyxia rate by month", "Rate", [0, 8])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_asphyxia, 100)
                            chart = bar_chart_ci(bar_data, "Asphyxia rate annually", "Rate", [0, 8])
                            st.altair_chart(chart)

                if selected_plot == "High-risk pregnancies":
                    st.markdown("<h3 style='text-align: left;'>High-risk pregnancies per 100 live births</h3>",
                                unsafe_allow_html=True)

                    df_HRP = prepare_chart_data(b_df, i_df, 'High risk', 'Live Births Final', n_months, n_runs, 100)
                    col1, col2 = st.columns(2)
                    with col1:
                        line_data = create_line_data(df_HRP, 100)
                        chart = line_chart_ci(line_data, "High-risk pregnancies rate by month", "Rate", [0, 100])
                        st.altair_chart(chart)
                    with col2:
                        bar_data = create_bar_data(df_HRP, 100)
                        chart = bar_chart_ci(bar_data, "High-risk pregnancies rate annually", "Rate", [0, 100])
                        st.altair_chart(chart)

                if selected_plot == "Neonatal mortality rate":
                    st.markdown("<h3 style='text-align: left;'>Neonatal deaths per 1,000 live births</h3>",
                                unsafe_allow_html=True)

                    df_NM = prepare_chart_data(b_df, i_df, 'Neonatal Deaths', 'Live Births Final', n_months, n_runs, 1000)
                    col1, col2 = st.columns(2)
                    with col1:
                        line_data = create_line_data(df_NM, 1000)
                        ymax = line_data['Upper_rate'].max() + 5
                        chart = line_chart_ci(line_data, "Neonatal mortality rate by month", "NMR", [0, ymax])
                        st.altair_chart(chart)
                    with col2:
                        bar_data = create_bar_data(df_NM, 1000)
                        ymax = bar_data['Rate'].max() + 5
                        chart = bar_chart_ci(bar_data, "Neonatal mortality rate annually", "NMR", [0, ymax])
                        st.altair_chart(chart)

                if selected_plot == "Stillbirth rate":
                    st.markdown("<h3 style='text-align: left;'>Stillbirths per 1,000 live births</h3>",
                                unsafe_allow_html=True)

                    tab1, tab2 = st.tabs(["Intrapartum", "Antepartum"])

                    with tab1:
                        df_SB = prepare_chart_data(b_df, i_df, 'stillbirths', 'Live Births Final', n_months, n_runs, 1000)
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_SB, 1000)
                            chart = line_chart_ci(line_data, "Stillbirth rate by month", "Rate (per 1000)", [0, 100])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_SB, 1000)
                            chart = bar_chart_ci(bar_data, "Stillbirth rate annually", "Rate (per 1000)", [0, 100])
                            st.altair_chart(chart)

                    with tab2:
                        pass


                if selected_plot == "Normal referral":
                    st.markdown("<h3 style='text-align: left;'>Normal referrals to L4/5 facilities per 100 live births</h3>",
                                unsafe_allow_html=True)

                    tab1, tab2, tab3 = st.tabs(["Self referrals", "Referrals using free bodas", "Total referrals"])

                    with tab1:
                        df_self_refer = prepare_referral_data(b_df, i_df, 'Self_referrals', n_months, n_runs, 100)
                        col1, col2 = st.columns(2)
                        ymax = 100 #df_self_refer['Rate'].max()
                        with col1:
                            line_data = create_line_data(df_self_refer, 100)
                            chart = line_chart_ci(line_data, "Self-referral rate by month", "Rate", [0, ymax])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_self_refer, 100)
                            chart = bar_chart_ci(bar_data, "Self-referral rate annually", "Rate", [0, ymax])
                            st.altair_chart(chart)

                    with tab2:
                        df_free_refer = prepare_referral_data(b_df, i_df, 'Free_referrals', n_months, n_runs, 100)
                        col1, col2 = st.columns(2)
                        ymax = 100
                        #ymax = df_free_refer['Rate'].max()
                        with col1:
                            line_data = create_line_data(df_free_refer, 100)
                            chart = line_chart_ci(line_data, "Public referral rate by month", "Rate", [0, ymax])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_free_refer, 100)
                            chart = bar_chart_ci(bar_data, "Public referral rate annually", "Rate", [0, ymax])
                            st.altair_chart(chart)

                    with tab3:
                        df_refer = prepare_referral_data(b_df, i_df, 'Normal_referrals', n_months, n_runs, 100)
                        col1, col2 = st.columns(2)
                        ymax = 100
                        #ymax = df_refer['Rate'].max()
                        with col1:
                            line_data = create_line_data(df_refer, 100)
                            chart = line_chart_ci(line_data, "Total referral rate by month", "Rate", [0, ymax])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_refer, 100)
                            chart = bar_chart_ci(bar_data, "Total referral rate annually", "Rate", [0, ymax])
                            st.altair_chart(chart)


                if selected_plot == "Emergency transfer":
                    st.markdown("<h3 style='text-align: left;'>Emergency transfers per 100 live births</h3>",
                                unsafe_allow_html=True)
                    tab1, tab2 = st.tabs(["For predicted pre-labor complications", "For actually occured complications"])

                    with tab1:
                        df_EM = prepare_chart_data(b_df, i_df, 'ER_trans_pred', 'Live Births Final', n_months, n_runs, 100)
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_EM, 100)
                            ymax = line_data['Upper_rate'].max()
                            chart = line_chart_ci(line_data, "Emergency transfer rate by month", "Rate", [0, ymax])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_EM, 100)
                            chart = bar_chart_ci(bar_data, "Emergency transfer rate annually", "Rate", [0, ymax])
                            st.altair_chart(chart)

                    with tab2:
                        df_ET = prepare_chart_data(b_df, i_df, 'ER_trans_actual', 'Live Births Final', n_months, n_runs, 100)
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_ET, 100)
                            ymax = line_data['Upper_rate'].max()
                            chart = line_chart_ci(line_data, "Emergency transfer rate by month", "Rate", [0, ymax])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_ET, 100)
                            chart = bar_chart_ci(bar_data, "Emergency transfer rate annually", "Rate", [0, ymax])
                            st.altair_chart(chart)

                if selected_plot == "DALYs averted":
                    st.markdown("<h3 style='text-align: left;'>Disability-adjusted life years (DALYs) averted</h3>",
                                unsafe_allow_html=True)
                    tab1, tab2, tab3 = st.tabs(["Maternal", "Neonatal", "All"])

                    with tab1:
                        col1, col2 = st.columns(2)
                        with col1:
                            df_DALY_avt = Acum_DALY_df('M_DALYs', 'Live Births Final', 100000)

                            ymin = df_DALY_avt['Rate'].min()
                            ymax = df_DALY_avt['Rate'].max()
                            chart = (
                                alt.Chart(df_DALY_avt, title="Accumulated DALYs Averted by Month")
                                .mark_line()
                                .encode(
                                    x=alt.X("Month:Q", title="Time since the start of intervention implementation (Months)"),
                                    y=alt.Y("Rate:Q", axis=alt.Axis(title="DALYs per 100,000 population"),
                                            scale=alt.Scale(domain=[ymin, ymax])),
                                    tooltip=["Month:N", "Rate:Q"]
                                ).properties(width=700, height=400).interactive()
                            )
                            chart = chart.configure_title(
                                anchor='middle'
                            )

                            st.altair_chart(chart)

                        with col2:
                            num = df_DALY_avt.loc[df_DALY_avt['Month'] == n_months - 1, 'Rate'].values[0]
                            st.markdown(
                                f"<p style='font-size:30px;'>The DALYs averted by intervention in total is ~ **{round(num)}** per 100,000 pregnant mothers.</p>",
                                unsafe_allow_html=True)

                    with tab2:
                        col1, col2 = st.columns(2)
                        with col1:

                            df_DALY_avt = Acum_DALY_df('N_DALYs', 'Live Births Final', 100000)

                            ymin = df_DALY_avt['Rate'].min()
                            ymax = df_DALY_avt['Rate'].max()
                            chart = (
                                alt.Chart(df_DALY_avt, title="Accumulated DALYs Averted by Month")
                                .mark_line()
                                .encode(
                                    x=alt.X("Month:Q", title="Time since the start of intervention implementation (Months)"),
                                    y=alt.Y("Rate:Q", axis=alt.Axis(title="DALYs per 100,000 population"),
                                            scale=alt.Scale(domain=[ymin, ymax])),
                                    tooltip=["Month:N", "Rate:Q"]
                                ).properties(width=700, height=400).interactive()
                            )
                            chart = chart.configure_title(
                                anchor='middle'
                            )

                            st.altair_chart(chart)

                        with col2:
                            num = df_DALY_avt.loc[df_DALY_avt['Month'] == n_months - 1, 'Rate'].values[0]
                            st.markdown(
                                f"<p style='font-size:30px;'>The DALYs averted by intervention in total is ~ **{round(num)}** per 100,000 live births.</p>",
                                unsafe_allow_html=True)

                    with tab3:
                        col1, col2 = st.columns(2)
                        with col1:
                            df_DALY_avt = Acum_DALY_df('DALYs', 'Live Births Final', 100000)

                            ymin = df_DALY_avt['Rate'].min()
                            ymax = df_DALY_avt['Rate'].max()

                            chart = (
                                alt.Chart(df_DALY_avt, title="Accumulated DALYs Averted by Month")
                                .mark_line()
                                .encode(
                                    x=alt.X("Month:Q", title="Time since the start of intervention implementation (Months)"),
                                    y=alt.Y("Rate:Q", axis=alt.Axis(title="DALYs per 100,000 population"),
                                            scale=alt.Scale(domain=[ymin, ymax])),
                                    tooltip=["Month:N", "Rate:Q"]
                                ).properties(width=700, height=400).interactive()
                            )
                            chart = chart.configure_title(
                                anchor='middle'
                            )

                            st.altair_chart(chart)

                        with col2:
                            num = df_DALY_avt.loc[df_DALY_avt['Month'] == n_months - 1, 'Rate'].values[0]
                            st.markdown(
                                f"<p style='font-size:30px;'>The DALYs averted by intervention in total is ~ **{round(num)}** per 100,000 dyads.</p>",
                                unsafe_allow_html=True)

                if selected_plot == "DALYs":
                    st.markdown("<h3 style='text-align: left;'>Disability-adjusted life years (DALYs)</h3>",
                                unsafe_allow_html=True)

                    tab1, tab2, tab3 = st.tabs(["Maternal", "Neonatal", "All"])

                    with tab1:
                        df_DALY = prepare_chart_data(b_df, i_df, 'M_DALYs', 'Live Births Final', n_months, n_runs, 100000)

                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_DALY, 100000)
                            ymax = line_data['Upper_rate'].max()
                            chart = line_chart_ci(line_data, "Maternal DALYs by month", "DALYs per 100,000 population", [0, ymax])
                            st.altair_chart(chart)

                        with col2:
                            bar_data = create_bar_data(df_DALY, 100000)
                            ymax = bar_data['Upper_rate'].max()
                            chart = bar_chart_ci(bar_data, "Maternal DALYs annually", "DALYs per 100,000 population", [0, ymax])
                            st.altair_chart(chart)

                    with tab2:
                        df_DALY = prepare_chart_data(b_df, i_df, 'N_DALYs', 'Live Births Final', n_months, n_runs, 100000)
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_DALY, 100000)
                            ymax = line_data['Upper_rate'].max()
                            chart = line_chart_ci(line_data, "Neonatal DALYs by month", "DALYs per 100,000 population", [0, ymax])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_DALY, 100000)
                            ymax = bar_data['Upper_rate'].max()
                            chart = bar_chart_ci(bar_data, "Neonatal DALYs annually", "DALYs per 100,000 population", [0, ymax])
                            st.altair_chart(chart)

                    with tab3:
                        df_DALY = prepare_chart_data(b_df, i_df, 'DALYs', 'Live Births Final', n_months, n_runs, 100000)
                        col1, col2 = st.columns(2)
                        with col1:
                            line_data = create_line_data(df_DALY, 100000)
                            ymax = line_data['Upper_rate'].max()
                            chart = line_chart_ci(line_data, "DALYs by month", "DALYs per 100,000 population", [0, ymax])
                            st.altair_chart(chart)
                        with col2:
                            bar_data = create_bar_data(df_DALY, 100000)
                            ymax = bar_data['Upper_rate'].max()
                            chart = bar_chart_ci(bar_data, "DALYs annually", "DALYs per 100,000 population", [0, ymax])
                            st.altair_chart(chart)

        elif st.session_state.b_df is None or st.session_state.i_df is None:
            st.warning("Please run the model first before generating plots.")

    # ==========================================================================
    # TABLE DOWNLOAD SESSION
    # ==========================================================================

    LOCATION_MAP = {0: "Home", 1: "L2/3", 2: "L4", 3: "L5"}

    INTERVENTION_VARIABLE_PRESETS = {
        "PROMPTS": {
            "description": "ANC, referrals, delivery locations, maternal complications & deaths",
            "columns": [
                "i_ANC", "i_free_referral", "i_self_referral",
                "i_loc", "i_loc_new_v2",
                "i_pph", "i_mat_sepsis", "i_eclampsia",
                "i_OL_final", "i_ruptured_uterus", "i_aph",
                "i_mat_death", "death_cause",
            ],
        },
        "MENTORS": {
            "description": "Maternal complications & mortality",
            "columns": [
                "i_pph", "i_mat_sepsis", "i_eclampsia",
                "i_OL_final", "i_ruptured_uterus", "i_aph",
                "i_mat_death", "death_cause",
            ],
        },
        "Referral": {
            "description": "Maternal mortality",
            "columns": ["i_mat_death", "death_cause"],
        },
        "Blood": {
            "description": "Maternal mortality",
            "columns": ["i_mat_death", "death_cause"],
        },
        "Custom": {
            "description": "Choose your own variables",
            "columns": [],
        },
    }

    ALL_EXPORTABLE_COLUMNS = [
        "i_ANC", "i_free_referral", "i_self_referral",
        "i_loc", "i_loc_new_v1", "i_loc_new_v2",
        "i_risk", "i_risk_pred", "i_class",
        "i_GA", "i_preterm", "i_preterm_pred",
        "i_elec_CS", "i_pocus",
        "i_anemia", "i_anemia_new", "i_iv_iron",
        "i_PL", "i_PL_new", "i_OL", "i_OL_final",
        "i_hypoxia", "i_mod",
        "i_transfer_pred", "i_transfer_actual",
        "i_oxytocin", "i_pph_bundle", "i_MgSO4", "i_antibiotics",
        "i_pph", "i_pph_new", "i_pph_severe_new",
        "i_mat_sepsis", "i_mat_sepsis_new",
        "i_eclampsia", "i_eclampsia_new",
        "i_ruptured_uterus", "i_aph",
        "i_severe_new", "i_comp_death_new",
        "i_stillbirth", "i_asphyxia", "i_RDS",
        "i_mat_death", "i_neo_death",
        "i_transfer", "travel_time_transfer", "death_cause",
        "M_DALY", "N_DALY", "DALY",
    ]

    META_COLUMNS = ["Month", "Run", "Scenario", "Delivery_Location"]


    def _build_export_df(ind_outcomes_df, selected_columns, stratify_col="i_loc_new_v2"):
        available = [c for c in selected_columns if c in ind_outcomes_df.columns]
        meta = [c for c in ["Month", "Run", "Scenario"] if c in ind_outcomes_df.columns]
        has_stratify = stratify_col in ind_outcomes_df.columns
        keep_cols = meta + ([stratify_col] if has_stratify else []) + available
        keep = list(dict.fromkeys(keep_cols))
        export = ind_outcomes_df[keep].copy()
        if has_stratify:
            loc_series = export[stratify_col]
            if isinstance(loc_series, pd.DataFrame):
                loc_series = loc_series.iloc[:, 0]
            export["Delivery_Location"] = loc_series.map(
                lambda x: LOCATION_MAP.get(int(x) if not isinstance(x, str) else -1, "Unknown")
            )
            if stratify_col not in available:
                export = export.drop(columns=[stratify_col])
        return export


    st.markdown("---")
    st.header("Table Download Session")
    st.caption(
        "Run the model with the current intervention settings and download a filtered CSV table. "
        "Uses multiple-run mode with shared random seeds."
    )

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        dl_scenario_name = st.text_input(
            "Scenario name",
            placeholder="e.g., PROMPTS_Baseline",
            key="dl_scenario_name",
        )
    with dl_col2:
        dl_n_runs = st.number_input(
            "Number of runs",
            min_value=1, max_value=300, value=10, step=1,
            key="dl_n_runs",
        )

    intervention_names = list(INTERVENTION_VARIABLE_PRESETS.keys())
    dl_template = st.selectbox(
        "Intervention template (selects export variables)",
        options=intervention_names,
        key="dl_run_template",
    )
    st.caption(INTERVENTION_VARIABLE_PRESETS[dl_template]["description"])

    preset_cols = INTERVENTION_VARIABLE_PRESETS[dl_template]["columns"]
    if dl_template == "Custom":
        dl_columns = st.multiselect(
            "Select variables to export",
            options=ALL_EXPORTABLE_COLUMNS,
            default=[],
            key="dl_run_custom_cols",
        )
    else:
        dl_columns = st.multiselect(
            "Variables (pre-filled — add or remove as needed)",
            options=ALL_EXPORTABLE_COLUMNS,
            default=preset_cols,
            key="dl_run_template_cols",
        )

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        run_clicked = st.button("Run & Generate Table", key="btn_run_dl")
    with btn_col2:
        clean_clicked = st.button("Clean Up Table", key="btn_clean_dl")

    if clean_clicked:
        st.session_state.dl_table = None
        st.rerun()

    if run_clicked:
        if not dl_scenario_name:
            st.warning("Please enter a scenario name.")
        elif not dl_columns:
            st.warning("Select at least one variable to export.")
        else:
            n_months = MODEL["n_months"] if MODEL["n_months"] > 0 else 36
            int_period = MODEL["int_period"] if MODEL["int_period"] > 0 else 36

            run_seeds_matrix = make_shifted_run_seeds(dl_n_runs, n_months)

            dl_status = st.empty()
            dl_progress = st.progress(0)
            temp_ind = []

            for run_index in range(dl_n_runs):
                monthly_seeds = run_seeds_matrix[run_index]
                param_rng = np.random.default_rng(monthly_seeds[0])

                sc_param = get_parameters(rng=param_rng, county=selected_county)
                sc_param = calculate_derived_parameters(sc_param)
                sc_param.update({"E": copy.deepcopy(i_E), "S": copy.deepcopy(i_S), "HSS": copy.deepcopy(i_HSS)})
                sync_param_momish_from_hss(sc_param, i_HSS, i_flags)

                _, ind_outcomes, _ = run_model_dash(
                    sc_param, copy.deepcopy(i_flags), n_months, int_period, base_seed=monthly_seeds,
                )
                ind_outcomes["Run"] = run_index + 1
                ind_outcomes["Scenario"] = dl_scenario_name
                temp_ind.append(ind_outcomes)

                dl_progress.progress((run_index + 1) / dl_n_runs)
                dl_status.text(f"Running... {run_index + 1}/{dl_n_runs}")

            combined = pd.concat(temp_ind, ignore_index=True)
            export = _build_export_df(combined, dl_columns)

            col_order = META_COLUMNS + [
                c for c in dl_columns if c in export.columns and c not in META_COLUMNS
            ]
            col_order = [c for c in col_order if c in export.columns]
            export = export[col_order]

            st.session_state.dl_table = export
            dl_progress.progress(1.0)
            dl_status.text(f"Done! {dl_n_runs} runs, {len(export):,} rows")

    if st.session_state.dl_table is not None:
        export = st.session_state.dl_table
        st.success(f"Table ready: {len(export):,} rows, {export.shape[1]} columns")

        csv_data = export.to_csv(index=False)
        st.download_button(
            label=f"Download CSV ({len(export):,} rows)",
            data=csv_data,
            file_name=f"{dl_scenario_name or 'scenario'}_outcomes.csv",
            mime="text/csv",
            key="btn_download_csv",
        )

        with st.expander("Preview", expanded=False):
            st.dataframe(export.head(100), use_container_width=True, hide_index=True)
