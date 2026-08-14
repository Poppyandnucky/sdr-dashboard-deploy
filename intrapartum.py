import numpy as np
import random
import streamlit as st
import time
import pandas as pd
from LB_effect import pulse_effect
from global_func import (P_Prolonged, P_Prolonged_vectorized, P_Sepsis_vectorized, P_Sepsis, P_NEC_vectorized, P_NEC,
                         P_IVH_vectorized, P_IVH, comps_riskstatus_vs_lowrisk, \
    sensors_accuracy_vectorized, sensors_accuracy, fetal_sensor_calculator, P_intervention, \
                         intrapartum_prediction, comp_OL_type, comp_severe, emergency_transfer_comps, preterm_complication, SI_reduction)


# Fixed order matching the 'pulse_indicator_targets' list in parameters.py / array_constants sheet.
PULSE_INDICATOR_ORDER = ["p_pph", "p_aph", "p_OL", "p_ruptured_uterus", "p_eclampsia", "p_mat_sepsis"]


def select_pulse_target(indicator_values, indicator_targets, indicator_threshold):
    targeted_indicator = None
    max_gap_ratio = 0.0

    for indicator_name, current_value in indicator_values.items():
        if indicator_name not in indicator_targets:
            continue

        target_value = float(indicator_targets[indicator_name])
        if target_value == 0:
            continue

        gap_ratio = max((float(current_value) - target_value) / abs(target_value), 0)
        if gap_ratio > max_gap_ratio:
            max_gap_ratio = gap_ratio
            targeted_indicator = indicator_name

    if max_gap_ratio < indicator_threshold:
        return None

    return targeted_indicator


def reduce_binary_indicator_to_prevalence(
    indicator_array,
    current_prevalence,
    target_prevalence,
    indicator_name,
    targeted_indicator,
    flags,
    param,
    rng,
):
    new_prevalence = pulse_effect(
        indicator_name,
        targeted_indicator,
        current_prevalence,
        target_prevalence,
        flags,
        param,
        clip_min=0,
        clip_max=1,
    )

    if new_prevalence >= current_prevalence:
        return indicator_array

    indicator_array = np.asarray(indicator_array).copy()
    current_count = int(np.sum(indicator_array == 1))
    desired_count = int(round(new_prevalence * len(indicator_array)))
    n_to_remove = max(current_count - desired_count, 0)

    positive_indices = np.where(indicator_array == 1)[0]
    if n_to_remove > 0 and len(positive_indices) > 0:
        remove_indices = rng.choice(
            positive_indices,
            size=min(n_to_remove, len(positive_indices)),
            replace=False,
        )
        indicator_array[remove_indices] = 0

    return indicator_array


def apply_fqa_effect(P, flags, param):
    if not flags.get("flag_fqa", 0):
        return P

    fqa_knowledge_improve = float(param.get("fqa_knowledge_improve", 0.043))
    fqa_implementation_index = float(param.get("fqa_implementation_index", 0.0))
    P["knowledge"] = np.clip(P["knowledge"] + fqa_implementation_index * fqa_knowledge_improve, 0, 1)

    return P

def initialize_intra_params(individual_outcomes, track, flags, param, i, rng):
    i_GA = individual_outcomes['i_GA'].values
    num_mothers = len(i_GA)
    n_FT = np.count_nonzero(i_GA >= 37)

    # global parameters
    P = {}  # dict to restore probabilities
    n = {}  # dict to restore counts
    E = {}  # dict to restore effects
    S = {}  # dict to restore supplies and capacities

    # maternal complications
    P["FT_all"] = n_FT / num_mothers
    P["severe"] = param["severe"]                                                               # probability of severe complications by risk level
    P["OL"] = param["OL"]                                                                       # probability of obstructed labor if not prolonged vs prolonged
    P["hypoxia"] = param["p_hypoxia"] / P["FT_all"]                                             # probability of hypoxia for full-term live births

    P["OL_by_risk"] = np.array([param["OL_lowrisk"], param["OL_highrisk"]])
    P["hypoxia_by_risk"] = np.array(comps_riskstatus_vs_lowrisk(P["hypoxia"], param['p_highrisk'], param["RR_comp_highrisk_vs_lowrisk"]))
    P["ruptured_by_risk"] = np.array([param["ruptured_uterus_lowrisk"], param["ruptured_uterus_highrisk"]])
    P["aph_by_risk"] = np.array([param["aph_lowrisk"], param["aph_highrisk"]])
    P["eclampsia_by_risk_anemia"] = np.vstack([param["eclampsia_lowrisk_anemia"], param["eclampsia_highrisk_anemia"]])

    # maternal complications with anemia
    P["pph_OL_anemia"] = param["pph_OL_anemia"]
    P["mat_sepsis_OL_anemia"] = param["mat_sepsis_OL_anemia"]
    P["pph_elective_CS_anemia"] = param["pph_elective_CS_anemia"]
    P["mat_sepsis_elective_CS_anemia"] = param["mat_sepsis_elective_CS_anemia"]
    P["pph_emergency_CS_anemia"] = param["pph_emergency_CS_anemia"]
    P["mat_sepsis_emergency_CS_anemia"] = param["mat_sepsis_emergency_CS_anemia"]
    P["pph_other_anemia"] = param["pph_other_anemia"]
    P["mat_sepsis_other_anemia"] = param["mat_sepsis_other_anemia"]

    # neonatal complications by delivery mode
    P["RDS_T"] = param["RDS_T"]
    P["stillbirth_OL"] = param["p_stillbirth_OL"]            # probability of stillbirth by OL
    P["stillbirth_hypoxia"] = param["p_stillbirth_hypoxia"]  # probability of stillbirths by hypoxia
    P["asphyxia_OL"] = param["p_asphyxia_OL"]                # probability of asphyxia by OL
    P["neo_sepsis_OL"] = param["p_neo_sepsis_OL"]            # probability of neonatal sepsis by OL

    #Sensitivity and specificity of traditional monitoring by location
    E["sen_comp_trad"] = np.array([0.5, param["sen_comp_trad"] * 0.8, param["sen_comp_trad"], param[
        "sen_comp_trad"]])
    E["spec_comp_trad"] = np.array([0.5, param["spec_comp_trad"] * 0.8, param["spec_comp_trad"], param[
        "spec_comp_trad"]])

    # Facility capacity
    S["Fac_capacity"] = track['Facility_Capacity_Track'][i, 0]
    CS_Capacity = track['CS_Capacity_Track'][i, 0]
    if flags['flag_capacity']:
        S["CS_capacity"] = np.array([0, 0, CS_Capacity, CS_Capacity])
    else:
        S["CS_capacity"] = param["p_cs_capacity"]

    P["CS_AVD_ratio"] = np.array([param["CS_AVD_ratio"], 1 - param["CS_AVD_ratio"]])

    # Emergency transfer intervention
    P["transfer_rate_severe"] = np.zeros((4, 5))
    P["transfer_rate_notsevere"] = np.zeros((4, 5))
    P["transfer_rate_preterm"] = np.zeros((4, 5))

    p_transfer_capacity_severe = param['t_l23_l45_severe']
    p_transfer_capacity_nonsevere = param['t_l23_l45_notsevere']
    p_transfer_capacity_preterm = param['t_l23_l45_preterm']
    t_l4_l4_severe = param['t_l4_l4_severe']
    t_l4_l5_severe = param['t_l4_l5_severe']
    if flags.get("flag_transfer_capacity", 0):
        transfer_capacity_target = param["HSS"]["transfer_capacity_target"]
        p_transfer_capacity_severe = max(transfer_capacity_target, param["t_l23_l45_severe"])
        p_transfer_capacity_nonsevere = max(transfer_capacity_target, param["t_l23_l45_notsevere"])
        p_transfer_capacity_preterm = max(transfer_capacity_target, param["t_l23_l45_preterm"])
        t_l4_l4_severe = 0
        t_l4_l5_severe = 0

    f_transfer_rates_severe = np.array([
        [0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, p_transfer_capacity_severe / 2, p_transfer_capacity_severe / 2],
        [0.00, 0.00, t_l4_l4_severe, t_l4_l5_severe],
        [0.00, 0.00, 0.00, 0.00]
    ]) / 100

    f_transfer_rates_notsevere = np.array([
        [0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, p_transfer_capacity_nonsevere / 2, p_transfer_capacity_nonsevere / 2],
        [0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00]
    ]) / 100

    f_transfer_rates_preterm = np.array([
        [0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, p_transfer_capacity_preterm / 2, p_transfer_capacity_preterm / 2],
        [0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00]
    ]) / 100

    for k_L in range(4):
        P["transfer_rate_severe"][k_L, 0] = 1 - f_transfer_rates_severe[k_L].sum()
        P["transfer_rate_severe"][k_L, 1:] = f_transfer_rates_severe[k_L]
        P["transfer_rate_notsevere"][k_L, 0] = 1 - f_transfer_rates_notsevere[k_L].sum()
        P["transfer_rate_notsevere"][k_L, 1:] = f_transfer_rates_notsevere[k_L]
        P["transfer_rate_preterm"][k_L, 0] = 1 - f_transfer_rates_preterm[k_L].sum()
        P["transfer_rate_preterm"][k_L, 1:] = f_transfer_rates_preterm[k_L]

    # Intrapartum sensor intervention
    sensors = fetal_sensor_calculator(track, param, i, flags, rng)
    dopplers_ratio = np.array(
        [sensors['dopplers_l23_ratio'], sensors['dopplers_l4_ratio'], sensors['dopplers_l5_ratio']])
    CTGs_ratio = np.array([sensors['CTGs_l23_ratio'], sensors['CTGs_l4_ratio'], sensors['CTGs_l5_ratio']])
    S["dopplers"] = np.array([0, dopplers_ratio[0], dopplers_ratio[1], dopplers_ratio[2]])
    S["dopplers"] = np.clip(S["dopplers"], 0, 1)
    S["CTGs"] = np.array([0, CTGs_ratio[0], CTGs_ratio[1], CTGs_ratio[2]])
    S["CTGs"] = np.clip(S["CTGs"], 0, 1)
    flag_AI = flags['flag_sensor_ai']
    if flag_AI:
        E["sen_prolonged_IS"] = param["E"]["sens_sensor"]
        E["spec_prolonged_IS"] = param["E"]["spec_sensor"]
        E["sen_ol_IS"] = param["E"]["sens_sensor"]
        E["spec_ol_IS"] = param["E"]["spec_sensor"]
        E["sen_hypoxia_IS"] = param["E"]["sens_sensor"]
        E["spec_hypoxia_IS"] = param["E"]["spec_sensor"]
    else:
        E["sen_prolonged_IS"] = param["sen_prolonged_IS"]
        E["spec_prolonged_IS"] = param["spec_prolonged_IS"]
        E["sen_ol_IS"] = param["sen_ol_IS"]
        E["spec_ol_IS"] = param["spec_ol_IS"]
        E["sen_hypoxia_IS"] = param["sen_hypoxia_IS"]
        E["spec_hypoxia_IS"] = param["spec_hypoxia_IS"]

        # Single interventions
    P["knowledge"] = np.array([
        0,
        param['base_knowledge_L23'],
        param['base_knowledge_L45'],
        param['base_knowledge_L45']
    ])

    if flags['flag_performance']:
        P["knowledge"][2] = param["HSS"]["knowledge"]
        P["knowledge"][3] = param["HSS"]["knowledge"]

    # MENTORS intervention (HSS dict + optional top-level copies from sync_param_momish_from_hss)
    if flags.get("flag_MENTOR"):

        mentors_coverage = param["mentors_implementation_index"]
        mentors_knowledge_target = float(param.get("mentors_knowledge_target", 1.0))

        P["knowledge"][1:4] = np.clip(
            P["knowledge"][1:4]
            + mentors_coverage * (mentors_knowledge_target - P["knowledge"][1:4]),
            0,
            1,
        )

    P = apply_fqa_effect(P, flags, param)

    E["int_pph"] = param['E_pph_bundle']
    E["stillbirth_CS"] = param["E_stillbirth_CS"]
    E["oxytocin"] = param["E_oxytocin"]
    P["pph_bundle"] = P_intervention('flag_pph_bundle', "pph_bundle", 'S_pph_bundle', flags, param, S, P)
    P["oxytocin"] = P_intervention('flag_oxytocin', "oxytocin", "S_oxytocin", flags, param, S, P)
    E["int_eclampsia"] = param['E_MgSO4']
    E["int_sepsis"] = param['E_antibiotics']
    P["MgSO4"] = P_intervention('flag_MgSO4', "MgSO4", 'S_MgSO4', flags, param, S, P)
    P["antibiotics"] = P_intervention('flag_antibiotics', "antibiotics", 'S_antibiotics', flags, param, S, P)

    # Initialize counters
    MC = {}  # dict to restore maternal complications
    NC = {}  # dict to restore neonatal complications
    M = {}  # dict to restore maternal outcomes

    keys_MC = ["PL", "hypoxia", "OL", "mat_sepsis", "pph", "eclampsia", "ruptured_uterus", "aph", "severe_comps", "pph_severe", "comps_death",
               "anemia"]
    keys_NC = ["stillbirth", "asphyxia", "neo_sepsis", "RDS", "IVH", "NEC"]
    keys_M = ["CS", "CS_unnessary", "AVD", "SVD", "ER_trans_pred", "ER_trans_actual",
              "LB_L_new", "ANC_L_new", "Highrisk_L_new", "Elective_CS", "Emergency_CS",
              "iv_iron", "pph_bundle", "PT", "MgSO4", "antibiotics", "LB_L_initial"]
    for key in keys_MC:
        MC[key] = np.zeros(4)
    for key in keys_NC:
        NC[key] = np.zeros(4)
    for key in keys_M:
        M[key] = np.zeros(4)

    M["Elective_CS_risk"] = np.zeros(2)  # Initialize elective CS by risk level
    M["GA"] = np.zeros((4, len(param['GA_sequence'])))  # Initialize GA distribution by facility levels
    return P, n, E, S, MC, NC, M


def intrapartum_effect_vectorized(track, flags, param, i, individual_outcomes, rng):
    ##-------------------Parameter initialization-------------------##
    #Extract individual outcomes
    i_loc = individual_outcomes['i_loc'].values
    i_ANC = individual_outcomes['i_ANC'].values
    i_risk = individual_outcomes['i_risk'].values
    i_GA = individual_outcomes['i_GA'].values
    i_elec_CS = individual_outcomes['i_elec_CS'].values
    i_jGA = individual_outcomes['i_jGA'].values
    i_anemia_new = individual_outcomes['i_anemia_new'].values
    i_iv_iron = individual_outcomes['i_iv_iron'].values
    i_highrisk = i_risk.copy()
    i_preterm = (i_GA < 37)
    i_FT = (i_GA >= 37)
    i_mod = np.where(i_elec_CS, "ELCS", "SVD")
    num_mothers = i_loc.shape[0]
    binary_outcomes = np.zeros((num_mothers, 13), dtype=int)
    (i_PL, i_OL, i_hypoxia, \
     i_pph, i_mat_sepsis, i_stillbirth, i_asphyxia, i_neo_sepsis, \
     i_RDS, i_IVH, i_NEC, i_transfer_pred, i_transfer_actual) = binary_outcomes.T  # Transpose for easy indexing
    # Initialize other parameters
    P, n, E, S, MC, NC, M = initialize_intra_params(individual_outcomes, track, flags, param, i, rng)

    # Complications not related to intrapartum monitoring - can be preterm or full-term
    i_eclampsia = (rng.random(num_mothers) < P["eclampsia_by_risk_anemia"][i_highrisk, i_anemia_new]).astype(int)
    i_ruptured_uterus = (rng.random(num_mothers) < P["ruptured_by_risk"][i_highrisk]).astype(int)
    i_aph = (rng.random(num_mothers) < P["aph_by_risk"][i_highrisk]).astype(int)

    ####----------------------------Intrapartum Monitoring ---------------------------------####
    i_intra_monitor = (i_FT & (i_elec_CS == 0)).astype(bool) #mothers need intrapartum monitoring
    index_intra_monitor = np.where(i_intra_monitor)[0]

    #**1) Initialize pre-labor complications
    P_PL = P_Prolonged_vectorized(i_GA, param)
    P_OL = P["OL_by_risk"][i_highrisk]
    P_hypoxia = P["hypoxia_by_risk"][i_highrisk]

    PL_true = (rng.random(num_mothers) < P_PL).astype(int)
    i_PL[index_intra_monitor] = PL_true[index_intra_monitor]
    index_ol_mask = np.where(i_intra_monitor & (i_PL == 0))[0]

    OL_true = (rng.random(num_mothers) < P_OL).astype(int)
    i_OL[index_ol_mask] = OL_true[index_ol_mask]

    index_hypoxia_mask = np.where(i_intra_monitor & (i_PL == 0) & (i_OL == 0))[0]
    hypoxia_true = (rng.random(num_mothers) < P_hypoxia).astype(int)
    i_hypoxia[index_hypoxia_mask] = hypoxia_true[index_hypoxia_mask]

    #**2) Fetal monitoring
    # extract sensitivity and specificity for each complication
    sen_PL, spec_PL, sen_OL, spec_OL, sen_hypoxia, spec_hypoxia, i_sensors = sensors_accuracy_vectorized(num_mothers, S, E, i_highrisk, i_loc, rng)

    # calculate TP, FP, TN, FN - only facilities with intrapartum monitoring
    facility_mask = (i_loc > 0)
    monitoring_mask = (facility_mask & i_intra_monitor).astype(bool)   #mothers can really get intrapartum monitoring in facilities

    i_PL_pred = intrapartum_prediction(num_mothers, monitoring_mask, i_PL, sen_PL, spec_PL, rng)
    i_OL_pred = intrapartum_prediction(num_mothers, monitoring_mask, i_OL, sen_OL, spec_OL, rng)
    i_hypoxia_pred = intrapartum_prediction(num_mothers, monitoring_mask, i_hypoxia, sen_hypoxia, spec_hypoxia, rng)

    #**3) Single interventions to reduce prolonged labor
    mother_need_treat = i_intra_monitor & (i_PL == 1) & (i_PL_pred == 1)
    index_treat_mask = np.where(mother_need_treat)[0]
    P_oxytocin = E["oxytocin"] * (P["oxytocin"][i_loc])  # Get effectiveness per facility
    PL_treated =  (rng.random(num_mothers) < P_oxytocin).astype(int)
    PL_removed = 1 - PL_treated
    i_PL_new = i_PL.copy()
    i_PL_pred_new = i_PL_pred.copy()
    i_PL_new[index_treat_mask] = PL_removed[index_treat_mask]
    i_PL_pred_new[index_treat_mask] = PL_removed[index_treat_mask]
    i_oxytocin = np.zeros(num_mothers, dtype=int)
    oxytocin_provided = (rng.random(num_mothers) < P["oxytocin"][i_loc]).astype(int)
    i_oxytocin[index_treat_mask] = oxytocin_provided[index_treat_mask]

    # 4) Initial delivery mode decided by monitoring
    emergency_mask = monitoring_mask & ((i_PL_pred_new == 1) | (i_OL_pred == 1) | (i_hypoxia_pred == 1))
    CS_capacity_mask = rng.random(num_mothers) < S["CS_capacity"][i_loc]                        # Check if CS is available
    emergency_cs_mask = emergency_mask & CS_capacity_mask                                               # emergency with CS capacity
    EmCS_or_AVD = rng.choice(["EmCS", "AVD"], size=num_mothers, p=P["CS_AVD_ratio"])
    i_mod[emergency_cs_mask] = EmCS_or_AVD[emergency_cs_mask]                                            # Assign `EmCS` or `AVD` for Mothers Who Need Emergency Delivery

    # 5) Emergency transfer for predicted complications - in L2/3
    i_loc_new_v1 = i_loc.copy()

    transfer_mask = emergency_mask & (i_mod == "SVD") & (i_loc == 1) # Define Transfer Mask (SVD Mothers with Predicted Complications)
    p_transfer_capacity = S["CS_capacity"][2] + S["CS_capacity"][3]
    with_cs_capacity = (rng.random(num_mothers) < p_transfer_capacity).astype(int)
    transfer_mask_2 = with_cs_capacity & transfer_mask
    n_transfer = np.sum(transfer_mask_2)

    #**Check facility capacity
    num_L4_L5 = np.count_nonzero(i_loc >= 2)                # Number of mothers in L4/L5
    max_capacity = S["Fac_capacity"].astype(int)            # Maximum Facility Capacity
    available_slots = max(0, max_capacity - num_L4_L5)      # Compute Available Capacity
    # Filter out mothers who need transfer
    shuffled_all = rng.permutation(num_mothers)             # shuffle all mother indices
    transfer_indices = np.where(transfer_mask_2)[0]         # identify mothers who need transfer
    shuffled_transfer = shuffled_all[np.isin(shuffled_all, transfer_indices)]  # filter only those who need transfer from the shuffled list
    # Ensure we only select from the shuffled transfer indices
    mask_can_transfer = np.zeros(num_mothers, dtype=bool)  # Always initialize mask (used or not)
    if available_slots > 0 and n_transfer > 0:             # Proceed only if allowed to transfer
        num_can_transfer = min(n_transfer, available_slots)
        selected_indices = shuffled_transfer[:num_can_transfer]
        mask_can_transfer[selected_indices] = True

    # Apply transfer to selected agents
    p_transfer_capacity_relative = np.array([S["CS_capacity"][2], S["CS_capacity"][3]]) / p_transfer_capacity
    l4_or_l5_all = rng.choice(np.array([2, 3]), size=num_mothers, p=p_transfer_capacity_relative)
    i_loc_new_v1[mask_can_transfer] = l4_or_l5_all[mask_can_transfer]
    i_mod[mask_can_transfer] = "EmCS"
    i_transfer_pred[mask_can_transfer] = 1

    ####----------------------------Home Births---------------------------------####
    home_mask = (i_loc_new_v1 == 0)
    i_mod[home_mask] = "SVD"        # All home births use "SVD"

    ####----------------------------Post-delivery complications---------------------------------####
    PL_lead_OL = ((i_PL_new == 1) & (i_mod == "SVD") & (rng.random(num_mothers) < P["OL"][1])).astype(int) #obstructed labor by prolonged labor
    OL_lead_OL = ((i_OL == 1) & (i_mod == "SVD")).astype(int)                                              #obstructed labor not caused by prolonged labor
    i_OL_final = ((PL_lead_OL == 1) | (OL_lead_OL == 1)).astype(int)                                       #sum of both

    # 1) **OL Complications: PPH, maternal sepsis, stillbirths, asphyxia, neonatal sepsis**
    i_OL_final_mask = (i_OL_final == 1)

    p_pph_ol = P["pph_OL_anemia"][i_anemia_new]
    p_sepsis_ol = P["mat_sepsis_OL_anemia"][i_anemia_new]
    p_stillbirth_ol = P["stillbirth_OL"]
    p_asphyxia_ol = P["asphyxia_OL"]
    p_neo_sepsis_ol = P["neo_sepsis_OL"]

    i_pph = comp_OL_type(num_mothers, i_pph, p_pph_ol, i_OL_final_mask, rng)
    i_mat_sepsis = comp_OL_type(num_mothers, i_mat_sepsis, p_sepsis_ol, i_OL_final_mask, rng)

    i_stillbirth = comp_OL_type(num_mothers, i_stillbirth, p_stillbirth_ol, i_OL_final_mask, rng)
    non_stillbirth_mask = (i_OL_final == 1) & (i_stillbirth == 0)
    i_asphyxia = comp_OL_type(num_mothers, i_asphyxia, p_asphyxia_ol, non_stillbirth_mask, rng)
    i_neo_sepsis = comp_OL_type(num_mothers, i_neo_sepsis, p_neo_sepsis_ol, non_stillbirth_mask, rng)

    # 2) **Emergency CS Complications: PPH and maternal sepsis**
    EmCS_mask = (i_mod == "EmCS")
    p_pph_emcs = P["pph_emergency_CS_anemia"][i_anemia_new]
    p_sepsis_emcs = P["mat_sepsis_emergency_CS_anemia"][i_anemia_new]
    i_pph = comp_OL_type(num_mothers, i_pph, p_pph_emcs, EmCS_mask, rng)
    i_mat_sepsis = comp_OL_type(num_mothers, i_mat_sepsis, p_sepsis_emcs, EmCS_mask, rng)

    # 3) **Hypoxia Complications: stillbirths**
    CS_effect = np.ones(num_mothers, dtype = int)
    hypoxia_cs_mask = (i_mod != "SVD")
    CS_effect[hypoxia_cs_mask] = 1 - E["stillbirth_CS"]
    p_stillbirth_hypoxia = P["stillbirth_hypoxia"] * CS_effect
    i_hypoxia_mask = (i_hypoxia == 1)
    i_stillbirth = comp_OL_type(num_mothers, i_stillbirth, p_stillbirth_hypoxia, i_hypoxia_mask, rng)

    # 4) **Elective CS Complications: PPH and maternal sepsis**
    i_elec_CS_mask = (i_elec_CS == 1)
    p_pph_elcs = P["pph_elective_CS_anemia"][i_anemia_new]
    p_sepsis_elcs = P["mat_sepsis_elective_CS_anemia"][i_anemia_new]
    i_pph = comp_OL_type(num_mothers, i_pph, p_pph_elcs, i_elec_CS_mask, rng)
    i_mat_sepsis = comp_OL_type(num_mothers, i_mat_sepsis, p_sepsis_elcs, i_elec_CS_mask, rng)

    # 5) **Other Cases: (SVD|AVD, ~ OL): PPH and maternal sepsis**
    other_mask = (~i_OL_final_mask) & np.isin(i_mod, ["SVD", "AVD"])
    p_pph_other = P["pph_other_anemia"][i_anemia_new]
    p_sepsis_other = P["mat_sepsis_other_anemia"][i_anemia_new]
    i_pph = comp_OL_type(num_mothers, i_pph, p_pph_other, other_mask, rng)
    i_mat_sepsis = comp_OL_type(num_mothers, i_mat_sepsis, p_sepsis_other, other_mask, rng)

    # PULSE acts after complications are observed and before downstream transfer/severity logic.
    targeted_indicator = None
    if flags.get("flag_pulse", 0):
        indicator_targets = dict(zip(PULSE_INDICATOR_ORDER, param.get("pulse_indicator_targets", [])))
        indicator_values = {
            "p_pph": np.mean(i_pph),
            "p_aph": np.mean(i_aph),
            "p_OL": np.mean(i_OL_final),
            "p_ruptured_uterus": np.mean(i_ruptured_uterus),
            "p_eclampsia": np.mean(i_eclampsia),
            "p_mat_sepsis": np.mean(i_mat_sepsis),
        }
        targeted_indicator = select_pulse_target(
            indicator_values,
            indicator_targets,
            float(param.get("pulse_indicator_threshold", 0.013)),
        )

        if targeted_indicator is not None:
            target_prevalence = float(indicator_targets[targeted_indicator])
            updated_indicator = reduce_binary_indicator_to_prevalence(
                {
                    "p_pph": i_pph,
                    "p_aph": i_aph,
                    "p_OL": i_OL_final,
                    "p_ruptured_uterus": i_ruptured_uterus,
                    "p_eclampsia": i_eclampsia,
                    "p_mat_sepsis": i_mat_sepsis,
                }[targeted_indicator],
                indicator_values[targeted_indicator],
                target_prevalence,
                targeted_indicator,
                targeted_indicator,
                flags,
                param,
                rng,
            )

            if targeted_indicator == "p_pph":
                i_pph = updated_indicator
            elif targeted_indicator == "p_aph":
                i_aph = updated_indicator
            elif targeted_indicator == "p_OL":
                i_OL_final = updated_indicator
            elif targeted_indicator == "p_ruptured_uterus":
                i_ruptured_uterus = updated_indicator
            elif targeted_indicator == "p_eclampsia":
                i_eclampsia = updated_indicator
            elif targeted_indicator == "p_mat_sepsis":
                i_mat_sepsis = updated_indicator

    ####----------------------------Emergency Transfer for complications---------------------------------####
    # ---- Step 1: Pre-transfer complications ----
    i_comp_death_bf = ((i_pph == 1) | (i_mat_sepsis == 1) | (i_eclampsia == 1) | (i_ruptured_uterus == 1) | (i_OL_final == 1) | (i_aph == 1)).astype(int)  #complications before transfer
    p_severe_risk = P["severe"][i_highrisk]
    i_pph_severe = comp_severe(num_mothers, i_pph, p_severe_risk, rng)
    i_sepsis_severe = comp_severe(num_mothers, i_mat_sepsis, p_severe_risk, rng)
    i_eclampsia_severe = comp_severe(num_mothers, i_eclampsia, p_severe_risk, rng)
    i_ol_severe = comp_severe(num_mothers, i_OL_final, p_severe_risk, rng)
    i_ruptured_uterus_severe = comp_severe(num_mothers, i_ruptured_uterus, p_severe_risk, rng)
    i_aph_severe = comp_severe(num_mothers, i_aph, p_severe_risk, rng)

    # ---- Step 2: Emergency Transfer ----
    i_severe_bf = ((i_pph_severe == 1) | (i_sepsis_severe == 1) | (i_eclampsia_severe == 1) | (i_ruptured_uterus_severe == 1) | (i_ol_severe == 1) | (i_aph_severe)).astype(int)
    i_notsevere_bf = (i_comp_death_bf == 1) & (i_severe_bf == 0)
    severe_mask = (i_severe_bf == 1)                        # Transfer Condition 1 - severe
    notsevere_mask = (i_notsevere_bf == 1)                  # Transfer Condition 3 - not severe
    preterm_mask = i_preterm & (~i_comp_death_bf)             # Transfer Condition 3 - preterm
    max_capacity = S["Fac_capacity"].astype(int)            # Maximum Facility Capacity

    i_loc_new_v2, i_transfer_actual = emergency_transfer_comps(i_transfer_actual, num_mothers, i_loc_new_v1, max_capacity, severe_mask, 1, P["transfer_rate_severe"], rng)       # severe transfers from l23 to l45
    i_loc_new_v3, i_transfer_actual = emergency_transfer_comps(i_transfer_actual, num_mothers, i_loc_new_v2, max_capacity, severe_mask, 2, P["transfer_rate_severe"], rng)       # severe transfers from l4 to l45
    i_loc_new_v4, i_transfer_actual = emergency_transfer_comps(i_transfer_actual, num_mothers, i_loc_new_v3, max_capacity, preterm_mask, 1, P["transfer_rate_preterm"], rng)     # preterm transfers from l23 to l45
    i_loc_new_final, i_transfer_actual = emergency_transfer_comps(i_transfer_actual, num_mothers, i_loc_new_v4, max_capacity, notsevere_mask, 1, P["transfer_rate_notsevere"], rng) #not severe transfers from l23 to l45

    # ---- Step 3: Preterm-related complications and treatments ----
    preterm_mask2 = (i_preterm == 1)
    p_treat = param["S_preterm_treat"][i_loc_new_final] * P["knowledge"][i_loc_new_final]
    i_T = (rng.random(num_mothers) < p_treat).astype(int)
    p_RDS = P["RDS_T"][i_T, i_jGA]
    p_IVH = P_IVH_vectorized(i_GA, i_T, param)
    p_NEC = P_NEC_vectorized(i_GA, i_T, param)
    p_neo_sepsis = P_Sepsis_vectorized(i_GA, i_T, param)

    i_RDS = preterm_complication(num_mothers, preterm_mask2, i_RDS, p_RDS, rng)
    i_IVH = preterm_complication(num_mothers, preterm_mask2, i_IVH, p_IVH, rng)
    i_NEC = preterm_complication(num_mothers, preterm_mask2, i_NEC, p_NEC, rng)
    i_neo_sepsis = preterm_complication(num_mothers, preterm_mask2, i_neo_sepsis, p_neo_sepsis, rng)

    # ---- Step 4: Single Interventions for reducing postpartum complications ----
    #pph bundle
    i_pph_new, i_pph_severe_new, i_pph_bundle = SI_reduction(num_mothers, i_loc_new_final, i_pph, i_pph_severe, P["pph_bundle"], E["int_pph"], rng)
    #MGSO4
    i_eclampsia_new, i_eclampsia_severe_new, i_MgSO4 = SI_reduction(num_mothers, i_loc_new_final, i_eclampsia, i_eclampsia_severe, P["MgSO4"], E["int_eclampsia"], rng)
    #antibiotics
    i_mat_sepsis_new, i_sepsis_severe_new, i_antibiotics = SI_reduction(num_mothers, i_loc_new_final, i_mat_sepsis, i_sepsis_severe, P["antibiotics"], E["int_sepsis"], rng)

    # if i == 4:
    #     st.text(i_eclampsia_new[20:40])
    #     st.text(np.sum(i_eclampsia_new))

    i_comp_death_new = ((i_pph_new == 1) | (i_mat_sepsis_new == 1) | (i_eclampsia_new == 1) | (i_ruptured_uterus == 1) | (i_OL_final == 1) | (i_aph == 1)).astype(int)
    i_severe_new = ((i_pph_severe_new == 1) | (i_sepsis_severe_new == 1) | (i_eclampsia_severe_new == 1) | (i_ruptured_uterus_severe == 1) | (i_ol_severe == 1) | (i_aph_severe == 1)).astype(int)

    i_unnecessary_cs = ((i_mod == "EmCS") & ~(i_PL_new | i_OL | i_hypoxia)).astype(int)

    #Update outcomes
    M, MC, NC, track = update_outcomes_vectorized(
        M, MC, NC, track, i_loc_new_final, i_jGA, i_mod, i_highrisk,
        i_PL_new, i_OL, i_hypoxia, i_transfer_pred, i_transfer_actual,
        i_iv_iron, i_pph_bundle, i_MgSO4, i_antibiotics, i_preterm, i_anemia_new, i_severe_new, i_pph_severe_new,
        i_eclampsia_new, i_ruptured_uterus, i_aph, i_OL_final, i_pph_new, i_mat_sepsis_new,
        i_comp_death_new, i_stillbirth, i_asphyxia, i_RDS, i_IVH,
        i_NEC, i_neo_sepsis, i_ANC, i, i_loc, i_unnecessary_cs
    )

    individual_outcomes = update_outcomes_vectorized_individual(i_loc_new_v1, i_loc_new_final, i_mod, i_PL,
                                          i_PL_new, i_OL, i_hypoxia, i_transfer_pred, i_transfer_actual,
                                          i_iv_iron, i_oxytocin, i_pph_bundle, i_MgSO4, i_antibiotics, i_preterm, i_severe_new, i_pph_severe_new,
                                          i_eclampsia, i_eclampsia_new, i_ruptured_uterus, i_aph, i_OL_final, i_pph, i_pph_new, i_mat_sepsis, i_mat_sepsis_new,
                                          i_comp_death_new, i_stillbirth, i_asphyxia, i_RDS, i_IVH,
                                          i_NEC, i_neo_sepsis, individual_outcomes,
                                          i_PL_pred, i_OL_pred, i_hypoxia_pred, i_sensors,
                                          i_sepsis_severe_new, i_eclampsia_severe_new, i_ruptured_uterus_severe, i_ol_severe, i_aph_severe, i_loc, i_unnecessary_cs)

    return MC, M, NC, individual_outcomes


def update_outcomes_vectorized(M, MC, NC, track, i_loc_new_final, i_jGA, i_mod, i_highrisk,
                    i_PL_new, i_OL, i_hypoxia, i_transfer_pred, i_transfer_actual,
                    i_iv_iron, i_pph_bundle, i_MgSO4, i_antibiotics, i_preterm, i_anemia_new, i_severe_new, i_pph_severe_new,
                    i_eclampsia_new, i_ruptured_uterus, i_aph, i_OL_final, i_pph_new, i_mat_sepsis_new,
                    i_comp_death_new, i_stillbirth, i_asphyxia, i_RDS, i_IVH,
                    i_NEC, i_neo_sepsis, i_ANC, i, i_loc, i_unnecessary_cs):
    # Batch updates for M (Delivery and maternal outcomes)
    np.add.at(M["GA"], (i_loc_new_final, i_jGA), 1)
    np.add.at(M["CS"], i_loc_new_final, np.isin(i_mod, ["EmCS", "ELCS"]))
    np.add.at(M["Emergency_CS"], i_loc_new_final, i_mod == "EmCS")
    np.add.at(M["CS_unnessary"], i_loc_new_final, i_unnecessary_cs)
    np.add.at(M["Elective_CS_risk"], i_highrisk, i_mod == "ELCS")
    np.add.at(M["Elective_CS"], i_loc_new_final, i_mod == "ELCS")
    np.add.at(M["AVD"], i_loc_new_final, i_mod == "AVD")
    np.add.at(M["SVD"], i_loc_new_final, i_mod == "SVD")
    np.add.at(M["ER_trans_pred"], i_loc_new_final, i_transfer_pred)
    np.add.at(M["ER_trans_actual"], i_loc_new_final, i_transfer_actual)
    np.add.at(M["iv_iron"], i_loc_new_final, i_iv_iron)
    np.add.at(M["pph_bundle"], i_loc_new_final, i_pph_bundle)
    np.add.at(M["MgSO4"], i_loc_new_final, i_MgSO4)
    np.add.at(M["antibiotics"], i_loc_new_final, i_antibiotics)
    np.add.at(M["PT"], i_loc_new_final, i_preterm)

    # Fully vectorized batch updates for MC (Maternal complications)
    np.add.at(MC["anemia"], i_loc_new_final, i_anemia_new)
    np.add.at(MC["severe_comps"], i_loc_new_final, i_severe_new)
    np.add.at(MC["pph_severe"], i_loc_new_final, i_pph_severe_new)
    np.add.at(MC["eclampsia"], i_loc_new_final, i_eclampsia_new)
    np.add.at(MC["ruptured_uterus"], i_loc_new_final, i_ruptured_uterus)
    np.add.at(MC["aph"], i_loc_new_final, i_aph)
    np.add.at(MC["PL"], i_loc_new_final, i_PL_new)
    np.add.at(MC["hypoxia"], i_loc_new_final, i_hypoxia)
    np.add.at(MC["OL"], i_loc_new_final, i_OL_final)
    np.add.at(MC["pph"], i_loc_new_final, i_pph_new)
    np.add.at(MC["mat_sepsis"], i_loc_new_final, i_mat_sepsis_new)
    np.add.at(MC["comps_death"], i_loc_new_final, i_comp_death_new)

    # Fully vectorized batch updates for NC (Neonatal complications)
    np.add.at(NC["stillbirth"], i_loc_new_final, i_stillbirth)
    np.add.at(NC["asphyxia"], i_loc_new_final, i_asphyxia)
    np.add.at(NC["RDS"], i_loc_new_final, i_RDS)
    np.add.at(NC["IVH"], i_loc_new_final, i_IVH)
    np.add.at(NC["NEC"], i_loc_new_final, i_NEC)
    np.add.at(NC["neo_sepsis"], i_loc_new_final, i_neo_sepsis)

    # Fully vectorized updates for Birth and Risk Tracking
    np.add.at(M["LB_L_initial"], i_loc, 1)
    np.add.at(M["LB_L_new"], i_loc_new_final, 1)
    np.add.at(M["ANC_L_new"], i_loc_new_final, i_ANC)
    np.add.at(M["Highrisk_L_new"], i_loc_new_final, i_highrisk)

    # Direct assignment for track tracking (remains unchanged)
    track["LB_Track"][i, :] = M["LB_L_new"]
    track["ANC_Track"][i, :] = M["ANC_L_new"]
    track["HighRisk_Track"][i, :] = M["Highrisk_L_new"]

    return M, MC, NC, track

def update_outcomes_vectorized_individual(i_loc_new_v1, i_loc_new_final, i_mod, i_PL,
                    i_PL_new, i_OL, i_hypoxia, i_transfer_pred, i_transfer_actual,
                    i_iv_iron, i_oxytocin, i_pph_bundle, i_MgSO4, i_antibiotics, i_preterm, i_severe_new, i_pph_severe_new,
                    i_eclampsia, i_eclampsia_new, i_ruptured_uterus, i_aph, i_OL_final, i_pph, i_pph_new, i_mat_sepsis, i_mat_sepsis_new,
                    i_comp_death_new, i_stillbirth, i_asphyxia, i_RDS, i_IVH,
                    i_NEC, i_neo_sepsis, individual_outcomes,
                    i_PL_pred, i_OL_pred, i_hypoxia_pred, i_sensors,
                    i_sepsis_severe_new, i_eclampsia_severe_new, i_ruptured_uterus_severe, i_ol_severe, i_aph_severe, i_loc, i_unnecessary_cs):
    individual_outcomes["i_loc"] = i_loc.astype(int)
    individual_outcomes["i_loc_new_v1"] = i_loc_new_v1.astype(int)
    individual_outcomes["i_loc_new_v2"] = i_loc_new_final.astype(int)
    individual_outcomes["i_mod"] = i_mod.astype(str)
    individual_outcomes["i_PL"] = i_PL.astype(int)
    individual_outcomes["i_PL_new"] = i_PL_new.astype(int)
    individual_outcomes["i_OL"] = i_OL.astype(int)
    individual_outcomes["i_OL_final"] = i_OL_final.astype(int)
    individual_outcomes["i_hypoxia"] = i_hypoxia.astype(int)
    individual_outcomes["i_transfer_pred"] = i_transfer_pred.astype(int)
    individual_outcomes["i_transfer_actual"] = i_transfer_actual.astype(int)
    individual_outcomes["i_iv_iron"] = i_iv_iron.astype(int)
    individual_outcomes["i_oxytocin"] = i_oxytocin.astype(int)
    individual_outcomes["i_pph_bundle"] = i_pph_bundle.astype(int)
    individual_outcomes["i_MgSO4"] = i_MgSO4.astype(int)
    individual_outcomes["i_antibiotics"] = i_antibiotics.astype(int)
    individual_outcomes["i_preterm"] = i_preterm.astype(int)
    individual_outcomes["i_severe_new"] = i_severe_new.astype(int)
    individual_outcomes["i_pph_severe_new"] = i_pph_severe_new.astype(int)
    individual_outcomes["i_eclampsia"] = i_eclampsia.astype(int)
    individual_outcomes["i_eclampsia_new"] = i_eclampsia_new.astype(int)
    individual_outcomes["i_ruptured_uterus"] = i_ruptured_uterus.astype(int)
    individual_outcomes["i_aph"] = i_aph.astype(int)
    individual_outcomes["i_pph"] = i_pph.astype(int)
    individual_outcomes["i_pph_new"] = i_pph_new.astype(int)
    individual_outcomes["i_mat_sepsis"] = i_mat_sepsis.astype(int)
    individual_outcomes["i_mat_sepsis_new"] = i_mat_sepsis_new.astype(int)
    individual_outcomes["i_comp_death_new"] = i_comp_death_new.astype(int)
    individual_outcomes["i_stillbirth"] = i_stillbirth.astype(int)
    individual_outcomes["i_asphyxia"] = i_asphyxia.astype(int)
    individual_outcomes["i_RDS"] = i_RDS.astype(int)
    individual_outcomes["i_IVH"] = i_IVH.astype(int)
    individual_outcomes["i_NEC"] = i_NEC.astype(int)
    individual_outcomes["i_neo_sepsis"] = i_neo_sepsis.astype(int)
    individual_outcomes["i_PL_pred"] = i_PL_pred.astype(int)
    individual_outcomes["i_OL_pred"] = i_OL_pred.astype(int)
    individual_outcomes["i_hypoxia_pred"] = i_hypoxia_pred.astype(int)
    individual_outcomes["i_sensors"] = i_sensors.astype(int)
    individual_outcomes["i_sepsis_severe"] = i_sepsis_severe_new.astype(int)
    individual_outcomes["i_eclampsia_severe"] = i_eclampsia_severe_new.astype(int)
    individual_outcomes["i_ruptured_uterus_severe"] = i_ruptured_uterus_severe.astype(int)
    individual_outcomes["i_ol_severe"] = i_ol_severe.astype(int)
    individual_outcomes["i_aph_severe"] = i_aph_severe.astype(int)
    individual_outcomes["i_unnecessary_cs"] = i_unnecessary_cs.astype(int)

    return individual_outcomes
