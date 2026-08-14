import random
import numpy as np
from scipy.optimize import fsolve, least_squares
import math
import streamlit as st
from scipy.stats import truncnorm


def get_P_l45(p_anc, slider_params):
    idx = np.where(slider_params['p_l45_anc_slider'][:, 0] == p_anc)  # Find the row index where P_ANC matches
    return slider_params['p_l45_anc_slider'][idx, 1][0][0] if idx[0].size > 0 else None  # Return P_l45 or None if not found

def odds_prob(oddsratio, p_comp, p_expose):
    def equations(vars):
        x, y = vars
        eq1 = x / (1 - x) / (y / (1 - y)) - oddsratio
        eq2 = p_comp - p_expose * x - (1 - p_expose) * y  #x = P(comp|exposed), y = P(comp|not exposed)
        return [eq1, eq2]

    initial_guess = [0.5, 0.5]  # Initial guess for x and y
    solution = fsolve(equations, initial_guess)
    solution[0] = round(np.clip(solution[0], 0, 1), 2)
    solution[1] = round(np.clip(solution[1], 0, 1), 2)
    return solution #solution[0] = P(comp|exposed), solution[1] = P(comp|not exposed)

def GA_assign_kenya(param, n, P):
    n["GA"] = param["GA_distribution"]
    P["GA"] = n["GA"] / np.sum(n["GA"])
    PT_mask = np.array([1] * 10 + [0] * 8, dtype=bool)
    FT_mask = ~PT_mask
    P["GA"][PT_mask] = P["GA"][PT_mask] * param["PT_scale"]            # scale P[GA|PT] to match kenya level
    P_mult = (1 - np.sum(P["GA"][PT_mask])) / np.sum(P["GA"][FT_mask])
    P["GA"][FT_mask] = P["GA"][FT_mask] * P_mult
    return P["GA"]

def GA_by_ANC(param, OR, P):
    OR["ANC"] = param["OR_preterm_ANC"]
    PT_mask = np.array([1] * 10 + [0] * 8, dtype=bool)
    FT_mask = ~PT_mask
    Preterm_rate = np.sum(P["GA"][PT_mask])  # overall preterm rate
    preterm_anc_noanc = odds_prob(OR["ANC"], Preterm_rate, param['p_ANC_base'])
    P["GA_anc"] = P["GA"].copy()
    P_scale_anc = preterm_anc_noanc[0] / Preterm_rate
    P["GA_anc"][PT_mask] *= P_scale_anc
    P_mult = (1 - preterm_anc_noanc[0]) / np.sum(P["GA_anc"][FT_mask])
    P["GA_anc"][FT_mask] *= P_mult

    P["GA_noanc"] = P["GA"].copy()
    P_scale_noanc = preterm_anc_noanc[1] / Preterm_rate
    P["GA_noanc"][PT_mask] *= P_scale_noanc
    P_mult = (1 - preterm_anc_noanc[1]) / np.sum(P["GA_noanc"][FT_mask])
    P["GA_noanc"][FT_mask] *= P_mult
    return P["GA_anc"], P["GA_noanc"]

#Functions in ANC phase
def risk_stratification(i_risk, i_ANC, num_mothers, sen_risk, spec_risk, rng):
    i_risk_pred = np.zeros(num_mothers, dtype=int)
    true_high_anc_mask = (i_risk == 1) & (i_ANC == 1)
    pred_as_high = (rng.random(num_mothers) < sen_risk).astype(int)     #rng.binomial(1, sen_risk, num_mothers).astype(int)
    true_low_anc_mask = (i_risk == 0) & (i_ANC == 1)
    pred_as_low = 1 - (rng.random(num_mothers) < spec_risk).astype(int) #rng.binomial(1, spec_risk, num_mothers).astype(int)
    i_risk_pred[true_high_anc_mask] = pred_as_high[true_high_anc_mask]
    i_risk_pred[true_low_anc_mask] = pred_as_low[true_low_anc_mask]
    return i_risk_pred

def move_function(num_mothers, l4_l5, i_class, i_loc, i_loc_new, i_free_referral, i_self_referral, Referral_Capacity, flags, num_move, loc_index, rng):
    flag_refer_voucher = flags["flag_refer_voucher"]

    loc_mask = (i_loc == loc_index)
    eligible_indices = np.where(loc_mask)[0]

    # shuffle all mothers, filter to those eligible for relocation
    shuffled_all = rng.permutation(num_mothers)
    shuffled_eligible = shuffled_all[np.isin(shuffled_all, eligible_indices)]

    # Select top `num_move` eligible mothers to move
    index_move = shuffled_eligible[:num_move]
    mask_move = np.zeros(num_mothers, dtype=bool)
    mask_move[index_move] = True

    # Whether has free referrals by rescue network
    p_free_refer = Referral_Capacity if flag_refer_voucher else 0
    free_refer_draws = (rng.random(num_mothers) < p_free_refer).astype(int)
    move_free_refer_mask = mask_move & (free_refer_draws == 1)
    i_free_referral[move_free_refer_mask] = 1
    i_loc_new[move_free_refer_mask] = l4_l5[move_free_refer_mask]

    # if not, self referrals using their own transportation
    move_self_refer_mask = mask_move & (free_refer_draws == 0) & (
                i_class == 1)  # only high SES mothers can self refer
    i_self_referral[move_self_refer_mask] = 1
    i_loc_new[move_self_refer_mask] = l4_l5[move_self_refer_mask]

    return i_loc_new, i_free_referral, i_self_referral

#Functions in Intrapartum phase
def intrapartum_prediction(num_mothers, monitoring_mask, comp_type, sen_type, spec_type, rng):
    i_comp_pred = np.zeros(num_mothers, dtype=int)

    true_comp_monitor_mask = (comp_type == 1) & monitoring_mask
    pred_as_comp = (rng.random(num_mothers) < sen_type).astype(int)
    pred_as_comp[~monitoring_mask] = 0

    true_nocomp_monitor_mask = (comp_type == 0) & monitoring_mask
    pred_as_nocomp = 1 - (rng.random(num_mothers) < spec_type).astype(int)
    pred_as_nocomp[~monitoring_mask] = 0

    i_comp_pred[true_comp_monitor_mask] = pred_as_comp[true_comp_monitor_mask]
    i_comp_pred[true_nocomp_monitor_mask] = pred_as_nocomp[true_nocomp_monitor_mask]
    return i_comp_pred

def comp_OL_type(num_mothers, comp_type, p_comp_ol, mask, rng):
    comp_ol = (rng.random(num_mothers) < p_comp_ol).astype(int) #rng.binomial(1, p_comp_ol).astype(int)
    comp_type[mask] = comp_ol[mask]
    return comp_type

def comp_severe(num_mothers, comp_type, p_severe_risk, rng):
    i_comp_severe = np.zeros(num_mothers, dtype=int)
    true_severe = (rng.random(num_mothers) < p_severe_risk).astype(int) #rng.binomial(1, p_severe_risk).astype(int)
    i_comp_severe[(comp_type == 1)] = true_severe[(comp_type == 1)]
    return i_comp_severe

def P_IVH(GA, flag_T, param):
    b = 10.05928
    c = 0.13321
    d = 0.94642
    e = 25.86781
    OR = param['OR_IVH_treat'] #0.38

    if GA < 37:
        P = c + (d - c) / (1 + np.exp(b * (np.log(GA) - np.log(e))))
        if flag_T:
            P = OR / (OR + (1 / P - 1))
    else:
        P = 0
    return P

def P_IVH_vectorized(GA, flag_T, param):
    """ Vectorized function to compute probability of IVH """

    b = 10.05928
    c = 0.13321
    d = 0.94642
    e = 25.86781
    OR = param['OR_IVH_treat']  # 0.38

    # **Step 1: Compute IVH Probability for All GA Values**
    P = np.where(GA < 37, c + (d - c) / (1 + np.exp(b * (np.log(GA) - np.log(e)))), 0)

    # **Step 2: Adjust Probability for Treatment (Only for flag_T == 1)**
    P_safe = np.clip(P, 1e-6, 1 - 1e-6)
    P_Treated = OR / (OR + (1 / P_safe - 1))
    #P_Treated = np.where(P > 0, OR / (OR + (1 / P - 1)), 0)
    P = np.where(flag_T, P_Treated, P)  # Apply treatment adjustment only where flag_T == 1

    return P

def P_NEC(GA, flag_T, param):
    b = 19.6727
    c = 0.00318
    d = 0.10013
    e = 29.43533
    OR = param['OR_NEC_treat'] #0.28

    if GA < 37:
        P = c + (d - c) / (1 + np.exp(b * (np.log(GA) - np.log(e))))
        if flag_T:
            P = OR / (OR + (1 / P - 1))
    else:
        P = 0
    return P

def P_NEC_vectorized(GA, flag_T, param):
    """ Vectorized function to compute probability of NEC """

    b = 19.6727
    c = 0.00318
    d = 0.10013
    e = 29.43533
    OR = param['OR_NEC_treat']  # 0.28

    # **Step 1: Compute NEC Probability for All GA Values**
    P = np.where(GA < 37, c + (d - c) / (1 + np.exp(b * (np.log(GA) - np.log(e)))), 0)

    # **Step 2: Adjust Probability for Treatment (Only for flag_T == 1)**
    P_safe = np.clip(P, 1e-6, 1 - 1e-6)
    P_Treated = OR / (OR + (1 / P_safe - 1))
    #P_Treated = np.where(P > 0, OR / (OR + (1 / P - 1)), 0)
    P = np.where(flag_T, P_Treated, P)  # Apply treatment adjustment only where flag_T == 1

    return P

def P_Sepsis(GA, flag_T, param):
    RR = param['RR_Sepsis_treat'] #0.24

    if GA < 37:
        P = np.maximum(np.minimum(1, 20.5046 * np.exp(-0.271732 * GA)), 0)
        if flag_T:
            P = P * RR
    else:
        P = 0
    return P

def P_Sepsis_vectorized(GA, flag_T, param):
    """ Vectorized function to compute probability of Sepsis """

    RR = param['RR_Sepsis_treat']  # 0.24

    # **Step 1: Compute Sepsis Probability for All GA Values**
    P = np.where(GA < 37, np.clip(20.5046 * np.exp(-0.271732 * GA), 0, 1), 0)

    # **Step 2: Adjust Probability for Treatment (Only for flag_T == 1)**
    P = np.where(flag_T, P * RR, P)

    return P

def P_RDS(param):
    # Step 1: Clip to prevent division by zero or one
    P_safe = np.clip(param["p_RDS_noT"], 1e-6, 1 - 1e-6)

    # Step 2: Calculate treated probability using odds ratio formula
    P_Treated = param["OR_RDS_treat"] / (param["OR_RDS_treat"] + (1 / P_safe - 1))

    # Step 3: If original value is 0, return 0 (you already wrote this, good!)
    P_RDS_T = np.where(param["p_RDS_noT"] == 0, 0, P_Treated)

    # Step 4: Stack untreated and treated probabilities
    P = np.vstack([param["p_RDS_noT"], P_RDS_T])
    return P

def comps_riskstatus(P_C, P_HR, RR):
    # Complication rate for high-risk pregnancies
    P_C_HR = RR * P_C
    # Complication rate for low-risk pregnancies
    P_C_LR = (P_C * (1 - RR * P_HR)) / (1 - P_HR)
    return P_C_HR, P_C_LR

def comps_riskstatus_vs_lowrisk(P_C, P_HR, RR):
    denom = P_HR * RR + (1 - P_HR)
    P_C_LR = P_C / denom
    P_C_HR = RR * P_C_LR
    return P_C_HR, P_C_LR

def comp2_comp1_anemia(p_comp2_comp1, OR_comp2_anemia):
    P_comp2_comp1_noanemia = p_comp2_comp1
    P_comp2_comp1_anemia = (OR_comp2_anemia * p_comp2_comp1) / ((1 - p_comp2_comp1) + (OR_comp2_anemia * p_comp2_comp1))

    comp2_matrix = np.array([P_comp2_comp1_noanemia, P_comp2_comp1_anemia])
    return comp2_matrix

def P_Prolonged(GA):
    # Define the data from the table
    gestational_age_weeks = [37, 38, 39, 40, 41, 42]
    prolonged_labor_rates = [0.073, 0.079, 0.092, 0.112, 0.149, 0.197]  # Converting percentages to probabilities
    if GA < 37:
        return 0
    elif GA >= 42:
        return prolonged_labor_rates[-1]
    else:
        index = gestational_age_weeks.index(GA)
        return prolonged_labor_rates[index]

def P_Prolonged_vectorized(GA_array, param):
    # Step 1: Define gestational age categories
    gestational_age_weeks = np.array([37, 38, 39, 40, 41, 42])
    prolonged_labor_rates = param['p_PL_GA'] #np.array([0.073, 0.079, 0.092, 0.112, 0.149, 0.197])

    # Step 2: Create output array with same shape as GA_array
    P_PL = np.zeros_like(GA_array, dtype=float)                 # Default is 0 for GA < 37

    # Step 3: Apply vectorized conditions
    mask_42_plus = GA_array >= 42                               # Mothers with GA >= 42
    mask_37_to_41 = (GA_array >= 37) & (GA_array < 42)          # GA between 37-41

    # Step 4: Assign prolonged labor probabilities
    P_PL[mask_42_plus] = prolonged_labor_rates[-1]              # Assign highest rate for GA >= 42
    valid_GA = GA_array[mask_37_to_41]                          # Extract valid GAs
    indices = np.searchsorted(gestational_age_weeks, valid_GA)  # Find indices
    P_PL[mask_37_to_41] = prolonged_labor_rates[indices]        # Assign corresponding values

    return P_PL

def emergency_transfer_comps(i_transfer_actual, num_mothers, i_loc_last, max_capacity, comp_mask, i_loc_index, p_transfer_capacity, rng):
    i_loc_new = i_loc_last.copy()
    need_transfer = comp_mask & (i_loc_last == i_loc_index)                     # mothers need transfer in this level
    # **-transfer capacity
    p_t_to_l45 = p_transfer_capacity[i_loc_index, 3] + p_transfer_capacity[i_loc_index, 4]        # extract the probability of transfering to l4/5 by rescue network
    with_transport = rng.random(num_mothers) < p_t_to_l45                       # binary index for transport
    mask_to_l45_w_t = with_transport & need_transfer                            # mothers with transport
    n_to_l45_w_t = np.sum(mask_to_l45_w_t)                                      # count the number
    # **-facility capacity
    num_L4_L5 = np.count_nonzero(i_loc_last >= 2)                               # Number of mothers in L4/L5
    available_slots = max(0, max_capacity - num_L4_L5)                          # Compute Available Capacity

    shuffled_all = rng.permutation(num_mothers)                                 # shuffle all mother indices
    transfer_indices = np.where(mask_to_l45_w_t)[0]                             # identify mothers who need transfer in this round
    shuffled_transfer = shuffled_all[np.isin(shuffled_all, transfer_indices)]   # filter only those who need transfer from the shuffled list
    mask_can_transfer = np.zeros(num_mothers, dtype=bool)                       # Initialize mask for can transfer

    if available_slots > 0 and n_to_l45_w_t > 0:
        num_can_transfer = min(n_to_l45_w_t, available_slots)
        selected_indices = shuffled_transfer[:num_can_transfer]
        mask_can_transfer[selected_indices] = True

    if p_t_to_l45 > 0:
        p_t_to_l45_relative = np.array([p_transfer_capacity[i_loc_index, 3], p_transfer_capacity[i_loc_index, 4]]) / p_t_to_l45
    else:
        p_t_to_l45_relative = np.array([0.5, 0.5])
    l4_or_l5 = rng.choice([2, 3], size=num_mothers, p=p_t_to_l45_relative)
    i_loc_new[mask_can_transfer] = l4_or_l5[mask_can_transfer]   # Update the location for those who can transfer
    i_transfer_actual[mask_can_transfer] = 1                        # Mark these mothers as transferred (actual transfer)
    return i_loc_new, i_transfer_actual

def preterm_complication(num_mothers, preterm_mask2, comp_type, p_comp, rng):
    True_comp = (rng.random(num_mothers) < p_comp).astype(int)
    comp_type[preterm_mask2] = True_comp[preterm_mask2]
    return comp_type

def SI_reduction(num_mothers, i_loc_new_final, comp_type, comp_type_severe, int_prob, int_efficacy, rng):
    mother_with_comp = (comp_type == 1)
    comp_type_new = comp_type.copy()
    comp_type_severe_new = comp_type_severe.copy()

    i_int = np.zeros(num_mothers, dtype=int)
    # Determine whether intervention was provided, for all mothers
    p_int = int_prob[i_loc_new_final]
    int_provided = (rng.random(num_mothers) < p_int).astype(int) #rng.binomial(1, p_int)
    i_int[mother_with_comp] = int_provided[mother_with_comp]
    # Calculate effectiveness of intervention
    effect_int = p_int * int_efficacy
    # Pre-draw randoms for all mothers and apply effect where applicable
    comp_treated = (rng.random(num_mothers) < effect_int).astype(int) #rng.binomial(1, effect_int).astype(int)
    comp_removed = 1 - comp_treated
    # Update the complication flags only for affected mothers
    comp_type_new[mother_with_comp] = comp_removed[mother_with_comp]
    comp_type_severe_mask = (comp_type_severe == 1)
    comp_type_severe_new[mother_with_comp & comp_type_severe_mask] = comp_removed[mother_with_comp & comp_type_severe_mask]
    return comp_type_new, comp_type_severe_new, i_int

def sensors_accuracy(S, E, i_highrisk, k_L):
    S["sensors"] = S["CTGs"] if i_highrisk else S["dopplers"]
    sen_PL = E["sen_prolonged_IS"] * S["sensors"][k_L] + (1 - S["sensors"][k_L]) * E["sen_comp_trad"][k_L]
    spec_PL = E["spec_prolonged_IS"] * S["sensors"][k_L] + (1 - S["sensors"][k_L]) * E["spec_comp_trad"][k_L]
    sen_OL = E["sen_ol_IS"] * S["sensors"][k_L] + (1 - S["sensors"][k_L]) * E["sen_comp_trad"][k_L]
    spec_OL = E["spec_ol_IS"] * S["sensors"][k_L] + (1 - S["sensors"][k_L]) * E["spec_comp_trad"][k_L]
    sen_hypoxia = E["sen_hypoxia_IS"] * S["sensors"][k_L] + (1 - S["sensors"][k_L]) * E["sen_comp_trad"][k_L]
    spec_hypoxia = E["spec_hypoxia_IS"] * S["sensors"][k_L] + (1 - S["sensors"][k_L]) * E["spec_comp_trad"][k_L]
    return sen_PL, spec_PL, sen_OL, spec_OL, sen_hypoxia, spec_hypoxia

def sensors_accuracy_vectorized(num_mothers, S, E, i_highrisk, i_loc_new, rng):
    # Step 1: Vectorized Sensor Selection
    #S["CTGs"] refers to the supply level of CTGs, S["dopplers"] refers to the supply level of dopplers
    p_sensors = np.where(i_highrisk, S["CTGs"][i_loc_new], S["dopplers"][i_loc_new])
    sensors_binary = (rng.random(num_mothers) < p_sensors).astype(int)

    # Step 2: Compute Sensitivity & Specificity for Each Condition
    #IS refers to intra-partum sensors, trad refers to traditional monitoring approaches such as partogram
    sen_PL = E["sen_prolonged_IS"] * sensors_binary + (1 - sensors_binary) * E["sen_comp_trad"][i_loc_new]
    spec_PL = E["spec_prolonged_IS"] * sensors_binary + (1 - sensors_binary) * E["spec_comp_trad"][i_loc_new]

    sen_OL = E["sen_ol_IS"] * sensors_binary + (1 - sensors_binary) * E["sen_comp_trad"][i_loc_new]
    spec_OL = E["spec_ol_IS"] * sensors_binary + (1 - sensors_binary) * E["spec_comp_trad"][i_loc_new]

    sen_hypoxia = E["sen_hypoxia_IS"] * sensors_binary + (1 - sensors_binary) * E["sen_comp_trad"][i_loc_new]
    spec_hypoxia = E["spec_hypoxia_IS"] * sensors_binary + (1 - sensors_binary) * E["spec_comp_trad"][i_loc_new]
    return sen_PL, spec_PL, sen_OL, spec_OL, sen_hypoxia, spec_hypoxia, sensors_binary

def labor_calculator(n_lb, n_cs, param, flags):
    labor = {}

    # Calculate L4 and L5 Live Births and averages
    L23_LBs = n_lb[1]
    L4_LBs = n_lb[2]
    L5_LBs = n_lb[3]
    def average_births_per_facility(births, facility_count):
        return births / facility_count if np.isfinite(facility_count) and facility_count > 0 else 0.0

    Avg_L23_LBs = average_births_per_facility(L23_LBs, param['num_L2/3'])
    Avg_L4_LBs = average_births_per_facility(L4_LBs, param['num_L4'])
    Avg_L5_LBs = average_births_per_facility(L5_LBs, param['num_L5'])

    # Surgical staff calculation
    surgical_l23 = (param['surgical_needed_below_thres'] * param['num_L2/3'] if Avg_L23_LBs < param['Ave_LBs_thres']
                    else param['surgical_needed_perLB_above_thres'] * L23_LBs)
    surgical_l4 = (param['surgical_needed_below_thres'] * param['num_L4'] if Avg_L4_LBs < param['Ave_LBs_thres']
                   else param['surgical_needed_perLB_above_thres'] * L4_LBs)
    surgical_l5 = (param['surgical_needed_below_thres'] * param['num_L5'] if Avg_L5_LBs < param['Ave_LBs_thres']
                   else param['surgical_needed_perLB_above_thres'] * L5_LBs)
    surgical = np.array([surgical_l23, surgical_l4, surgical_l5])

    # Nurse staff calculation
    nurse_l23 = L23_LBs * param['nurse_needed_perLB']
    nurse_l4 = L4_LBs * param['nurse_needed_perLB']
    nurse_l5 = L5_LBs * param['nurse_needed_perLB']
    nurse = np.array([nurse_l23, nurse_l4, nurse_l5])

    # Anesthetist staff calculation
    L23_CS = n_cs[1]
    L4_CS = n_cs[2]
    L5_CS = n_cs[3]
    anesthetist_l23 = L23_CS * param['anesthetist_needed_perCS']
    anesthetist_l4 = L4_CS * param['anesthetist_needed_perCS']
    anesthetist_l5 = L5_CS * param['anesthetist_needed_perCS']
    anesthetist = np.array([anesthetist_l23, anesthetist_l4, anesthetist_l5])

    # Adjust based on flags
    if not flags['flag_labor']:
        actual_surgical = param['base_surgical']
        actual_nurse = param['base_nurse']
        actual_anesthetist = param['base_anesthetist']
    else:
        labor_ratio = param["HSS"]["labor_ratio"]
        actual_surgical = np.array([param['base_surgical'][0], \
                                    max(surgical_l4 * labor_ratio, param['base_surgical'][1]), \
                                    max(surgical_l5 * labor_ratio, param['base_surgical'][2])
                                    ])
        actual_nurse = np.array([param['base_nurse'][0], \
                                 max(nurse_l4 * labor_ratio, param['base_nurse'][1]), \
                                 max(nurse_l5 * labor_ratio, param['base_nurse'][2])
                                 ])
        actual_anesthetist = np.array([param['base_anesthetist'][0], anesthetist_l4 * labor_ratio, anesthetist_l5 * labor_ratio])

    # Round up to the nearest whole number and update the labor dictionary
    staff_keys = ['surgical', 'nurse', 'anesthetist']
    actual_staff = [actual_surgical, actual_nurse, actual_anesthetist]
    estimated_staff = [surgical, nurse, anesthetist]

    #math.ceil for each item within each item of actual_staff
    labor.update({f"actual_{key}": [math.ceil(val) for val in actual_staff] for key, actual_staff in zip(staff_keys, actual_staff)})
    labor.update({key: [math.ceil(val) for val in estimated_staff] for key, estimated_staff in zip(staff_keys, estimated_staff)})
    return labor

# Define a single function to compute the scaled quality-adjusted density index
def compute_scaled_density_index(d_surgical, d_nurses, surgical_weight, scaled_factor):
    surgical_weight = surgical_weight
    scaled_factor = scaled_factor
    # Apply weight to surgical staff
    d_surgical_weighted = d_surgical * surgical_weight

    # Compute the harmonic mean-based density index with weighted surgical staff
    numerator = 2 * d_surgical_weighted * d_nurses
    denominator = d_surgical_weighted + d_nurses
    quality_adjusted = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=(d_surgical_weighted > 0) & (d_nurses > 0) & (denominator != 0),
    )

    scaled_index = quality_adjusted * scaled_factor
    #scaled factor is used to match the baseline density of skilled healthcare worker to Kenya level = 174.09
    return scaled_index

def p_maternal_death(worker_density):
    #Calculate the probability of maternal death for mother given the density of skilled healthcare workers.
    a = 797.88  # Scaling factor
    b = 0.0171  # Decay rate
    c = 160.15  # Asymptote (minimum achievable MMR)

    mmr = a * np.exp(-b * worker_density) + c  # Compute MMR
    probability = mmr / 100000  # Convert MMR to individual probability
    return probability

def p_neonatal_death(worker_density):
    ##Calculate the probability of neonatal death given the density of skilled healthcare workers.
    # Parameters from fitted exponential decay function
    a = 190.58  # Scaling factor
    b = 0.00434  # Decay rate
    c = 81.84  # Asymptote (minimum achievable NMR)
    # Compute Neonatal Mortality Rate (NMR)
    nmr = a * np.exp(-b * worker_density) + c
    # Convert NMR (deaths per 10,000 live births) to probability per individual birth
    probability = nmr / 10000  # Since NMR is per 10,000 live births

    return probability

def baseline_p_death(track, M, param, flags, i, n):
    labor = labor_calculator(track['LB_Track'][i, :], M["CS"], param, flags)
    actual_surgical, actual_nurse, actual_anesthetist = labor['actual_surgical'], labor['actual_nurse'], labor[
        'actual_anesthetist']
    n_pop_wt_lb = param['n_population'] * n["LB_L"][1:] / np.sum(n["LB_L"])
    def worker_density(actual_staff):
        actual_staff = np.asarray(actual_staff, dtype=float)
        density = np.divide(
            actual_staff,
            n_pop_wt_lb,
            out=np.zeros_like(actual_staff),
            where=n_pop_wt_lb != 0,
        )
        return np.round(density * 100000)

    density_skilled_surgical = worker_density(actual_surgical)
    density_skilled_nurse = worker_density(actual_nurse)
    # Compute the scaled quality-adjusted density index
    scaled_density_index = compute_scaled_density_index(density_skilled_surgical, density_skilled_nurse,
                                                        param['surgical_weight'], param['scaled_factor_density'])

    # Compute the probability of maternal death at baseline and by facility level
    p_mat_death_l23 = p_maternal_death(scaled_density_index[0])
    p_mat_death_l4 = p_maternal_death(scaled_density_index[1])
    p_mat_death_l5 = p_maternal_death(scaled_density_index[2])
    p_mat_death_baseline = np.array([0, p_mat_death_l23, p_mat_death_l4, p_mat_death_l5])

    # Compute the probability of neonatal death at baseline and by facility level
    p_neo_death_l23 = p_neonatal_death(scaled_density_index[0])
    p_neo_death_l4 = p_neonatal_death(scaled_density_index[1])
    p_neo_death_l5 = p_neonatal_death(scaled_density_index[2])
    p_neo_death_baseline = np.array([0, p_neo_death_l23, p_neo_death_l4, p_neo_death_l5])

    return p_mat_death_baseline, p_neo_death_baseline

def DALY_calculator(df, param, i):
    # p_severe = df.loc[i, 'severe_comps'] / (df.loc[i, "Comps after transfer"])
    # p_mild = 1 - p_severe
    p_nodeath_comp = (df.loc[i, "Comps after transfer"] - df.loc[i, 'Deaths']) / df.loc[i, "Comps after transfer"] # avoid overcounting

    DW = param['DW']

    M_DALYs = (df.loc[i, 'Anemia'] * DW['anemia'] +
                # df.loc[i, 'pph'] * DW['low pph'] * p_mild +
                # df.loc[i, 'pph'] * DW['high pph'] * p_severe +
               (df.loc[i, 'pph'] - df.loc[i, 'severe_pph']) * DW['low pph'] * p_nodeath_comp +
               df.loc[i, 'severe_pph'] * DW['high pph'] * p_nodeath_comp +
                df.loc[i, 'mat_sepsis'] * DW['maternal sepsis'] * p_nodeath_comp +
                df.loc[i, 'eclampsia'] * DW['eclampsia'] * p_nodeath_comp +
                df.loc[i, 'OL'] * DW['obstructed labor'] * p_nodeath_comp +
                df.loc[i, 'Deaths'] * DW['maternal death']) * (param['Mother_life_expectancy'] - param['Childbearing_age'])

    N_DALYs = ((df.loc[i, 'RDS'] + df.loc[i, 'IVH'] + df.loc[i, 'NEC']) * DW['preterm comp'] +
                df.loc[i, 'neo_sepsis'] * DW['neonatal sepsis'] +
                df.loc[i, 'asphyxia'] * DW['asphyxia'] +
                df.loc[i, 'Neonatal Deaths'] * DW['neonatal death']) * param['Neonate_life_expectancy']

    return M_DALYs, N_DALYs

def DALY_calculator_vectorized(individual_outcomes, param):
    DW = param['DW']

    M_DALYs = np.zeros(4, dtype=float)
    N_DALYs = np.zeros(4, dtype=float)

    #extract individual outcomes
    i_loc = individual_outcomes["i_loc_new_v2"]
    i_anemia = individual_outcomes["i_anemia_new"]
    i_pph_severe = individual_outcomes["i_pph_severe_new"]
    i_pph = individual_outcomes["i_pph_new"]
    i_pph_notsevere = ((i_pph == 1) & (i_pph_severe == 0)).astype(int)
    i_mat_sepsis = individual_outcomes["i_mat_sepsis_new"]
    i_eclampsia = individual_outcomes["i_eclampsia_new"]
    i_OL = individual_outcomes["i_OL_final"]
    i_MD = individual_outcomes["i_mat_death"]

    i_RDS = individual_outcomes["i_RDS"]
    i_IVH = individual_outcomes["i_IVH"]
    i_NEC = individual_outcomes["i_NEC"]
    i_neo_sepsis = individual_outcomes["i_neo_sepsis"]
    i_asphyxia = individual_outcomes["i_asphyxia"]
    i_stillbirth = individual_outcomes["i_stillbirth"]
    i_ND = individual_outcomes["i_neo_death"]

    #DALY calculation
    num_mothers = i_anemia.shape[0]
    M_DALY = np.zeros(num_mothers, dtype=float)  #maternal DALY
    N_DALY = np.zeros(num_mothers, dtype=float)  #neonatal DALY
    Mcomps_mask = ((i_anemia == 1) | (i_pph_severe == 1) | (i_pph_notsevere == 1) | (i_mat_sepsis == 1) | (i_eclampsia == 1) | (i_OL == 1)) & (i_MD == 0)
    MD_mask = (i_MD == 1)
    Ncomps_mask = ((i_RDS == 1) | (i_IVH == 1) | (i_NEC == 1) | (i_neo_sepsis == 1) | (i_asphyxia == 1)) & (i_stillbirth == 0) & (i_ND == 0)
    ND_mask = (i_ND == 1) & (i_stillbirth == 0)
    #update DALY for mothers
    DALY_Mcomps = (i_anemia * DW['anemia'] + i_pph_notsevere * DW['low pph'] + i_pph_severe * DW['high pph'] + \
                  i_mat_sepsis * DW['maternal sepsis'] + i_eclampsia * DW['eclampsia'] + i_OL * DW['obstructed labor']) * \
                  (param['Mother_life_expectancy'] - param['Childbearing_age'])
    DALY_MD = (i_MD * DW['maternal death']) * (param['Mother_life_expectancy'] - param['Childbearing_age'])
    M_DALY[Mcomps_mask] = DALY_Mcomps[Mcomps_mask]
    M_DALY[MD_mask] = DALY_MD[MD_mask]
    #update DALY for babies
    DALY_Ncomps = (i_RDS * DW['preterm comp'] + i_IVH * DW['preterm comp'] + i_NEC * DW['preterm comp'] + \
                i_neo_sepsis * DW['neonatal sepsis'] + i_asphyxia * DW['asphyxia']) * param['Neonate_life_expectancy']
    DALY_ND = (i_ND * DW['neonatal death']) * param['Neonate_life_expectancy']
    N_DALY[Ncomps_mask] = DALY_Ncomps[Ncomps_mask]
    N_DALY[ND_mask] = DALY_ND[ND_mask]

    #aggregate M_DALYs and N_DALYs by facility level
    np.add.at(M_DALYs, i_loc, M_DALY)
    np.add.at(N_DALYs, i_loc, N_DALY)

    return M_DALYs, N_DALYs, M_DALY, N_DALY

def fetal_sensor_calculator(track, param, i, flags, rng):
    fetal_sensor = {}

    # Calculate high-risk and low-risk pregancies in each facility level
    # (a facility level with 0 facilities in this county has no per-facility rate to compute)
    highrisk_perday_perl23 = math.ceil(track['HighRisk_Track'][i][1] / 30 / param['num_L2/3']) if param['num_L2/3'] > 0 else 0
    highrisk_perday_perl4 = math.ceil(track['HighRisk_Track'][i][2] / 30 / param['num_L4']) if param['num_L4'] > 0 else 0
    highrisk_perday_perl5 = math.ceil(track['HighRisk_Track'][i][3] / 30 / param['num_L5']) if param['num_L5'] > 0 else 0
    lowrisk_perday_perl23 = math.ceil((track['LB_Track'][i][1] - track['HighRisk_Track'][i][1]) / 30 / param['num_L2/3']) if param['num_L2/3'] > 0 else 0
    lowrisk_perday_perl4 = math.ceil((track['LB_Track'][i][2] - track['HighRisk_Track'][i][2]) / 30 / param['num_L4']) if param['num_L4'] > 0 else 0
    lowrisk_perday_perl5 = math.ceil((track['LB_Track'][i][3] - track['HighRisk_Track'][i][3]) / 30 / param['num_L5']) if param['num_L5'] > 0 else 0
    

    # Calculate the number of fetal dopplers needed
    def doppler_usage_time(lowrisks_perday):
        total_doppler_usage_time = 0.0
        if lowrisks_perday > 0:
            for K_LB in range(lowrisks_perday):
                duration_1st_stage = param['1st_stage_time_normal'][0] + param['1st_stage_time_normal'][1] * rng.normal()
                duration_2nd_stage = param['2nd_stage_time_normal'][0] + param['2nd_stage_time_normal'][1] * rng.normal()
                doppler_usage_time = (duration_1st_stage / param['check_interval_1st_stage'] + duration_2nd_stage / param['check_interval_2nd_stage']) * param['check_time_doppler']
                total_doppler_usage_time += doppler_usage_time
        return total_doppler_usage_time

    total_doppler_usage_time_perl23 = doppler_usage_time(lowrisk_perday_perl23)
    total_doppler_usage_time_perl4 = doppler_usage_time(lowrisk_perday_perl4)
    total_doppler_usage_time_perl5 = doppler_usage_time(lowrisk_perday_perl5)

    num_dopplers_l23 = math.ceil(total_doppler_usage_time_perl23 / param['usage_time_sensor_perday']) * param['num_L2/3']
    num_dopplers_l4 = math.ceil(total_doppler_usage_time_perl4 / param['usage_time_sensor_perday']) * param['num_L4']
    num_dopplers_l5 = math.ceil(total_doppler_usage_time_perl5 / param['usage_time_sensor_perday']) * param['num_L5']

    # Calculate the number of CTGs needed
    def CTG_usage_time(highrisks_perday):
        total_CTG_usage_time = 0.0
        if highrisks_perday > 0:
            for K_LB in range(highrisks_perday):
                duration_1st_stage = param['1st_stage_time_abnormal'][0] + param['1st_stage_time_abnormal'][1] * rng.normal()
                duration_2nd_stage = param['2nd_stage_time_abnormal'][0] + param['2nd_stage_time_abnormal'][1] * rng.normal()
                CTG_usage_time = duration_1st_stage + duration_2nd_stage
                total_CTG_usage_time += CTG_usage_time
        return total_CTG_usage_time

    total_CTG_usage_time_perl23 = CTG_usage_time(highrisk_perday_perl23)
    total_CTG_usage_time_perl4 = CTG_usage_time(highrisk_perday_perl4)
    total_CTG_usage_time_perl5 = CTG_usage_time(highrisk_perday_perl5)

    num_CTGs_l23 = math.ceil(total_CTG_usage_time_perl23 / param['usage_time_sensor_perday']) * param['num_L2/3']
    num_CTGs_l4 = math.ceil(total_CTG_usage_time_perl4 / param['usage_time_sensor_perday']) * param['num_L4']
    num_CTGs_l5 = math.ceil(total_CTG_usage_time_perl5 / param['usage_time_sensor_perday']) * param['num_L5']

    # Adjust based on flags
    if (not flags['flag_intrasensor']) and (not flags['flag_equipment']):
        actual_dopplers_l23, actual_dopplers_l4, actual_dopplers_l5 = param['num_dopplers_L2/3'], param['num_dopplers_L4'], param['num_dopplers_L5']
        actual_CTGs_l23, actual_CTGs_l4, actual_CTGs_l5 = param['num_CTGs_L2/3'], param['num_CTGs_L4'], param['num_CTGs_L5']

    if flags['flag_equipment']:
        actual_dopplers_l23, actual_dopplers_l4, actual_dopplers_l5 = \
            param['num_dopplers_L2/3'], \
            max(param["HSS"]["sensor_ratio"] * num_dopplers_l4, param['num_dopplers_L4']), \
            max(param["HSS"]["sensor_ratio"] * num_dopplers_l5, param['num_dopplers_L5'])

        actual_CTGs_l23, actual_CTGs_l4, actual_CTGs_l5 = \
                param['num_CTGs_L2/3'], \
                max(param["HSS"]["sensor_ratio"] * num_CTGs_l4, param['num_CTGs_L4']), \
                max(param["HSS"]["sensor_ratio"] * num_CTGs_l5, param['num_CTGs_L5'])

    if flags['flag_intrasensor']:
        actual_dopplers_l23, actual_dopplers_l4, actual_dopplers_l5 = num_dopplers_l23, num_dopplers_l4, num_dopplers_l5
        actual_CTGs_l23, actual_CTGs_l4, actual_CTGs_l5 = num_CTGs_l23, num_CTGs_l4, num_CTGs_l5

    # Update the fetal sensor dictionary
    sensor_keys = ['dopplers_l23', 'dopplers_l4', 'dopplers_l5', 'CTGs_l23', 'CTGs_l4', 'CTGs_l5']
    actual_sensors = [actual_dopplers_l23, actual_dopplers_l4, actual_dopplers_l5, actual_CTGs_l23, actual_CTGs_l4, actual_CTGs_l5]
    estimated_sensors = [num_dopplers_l23, num_dopplers_l4, num_dopplers_l5, num_CTGs_l23, num_CTGs_l4, num_CTGs_l5]
    #if estimated_sensor == 0, set ratio to 0, else calculate ratio
    sensors_ratio = [0 if est_sensor == 0 else act_sensor / est_sensor for act_sensor, est_sensor in zip(actual_sensors, estimated_sensors)]

    fetal_sensor.update({f"actual_{key}": math.ceil(val) for key, val in zip(sensor_keys, actual_sensors)})
    fetal_sensor.update({key: math.ceil(val) for key, val in zip(sensor_keys, estimated_sensors)})
    fetal_sensor.update({f"{key}_ratio": val for key, val in zip(sensor_keys, sensors_ratio)})

    return fetal_sensor

def P_intervention(int_flag, int_name, int_param, flags, param, S, P):
    S[int_name] = param[int_param]
    if flags[int_flag]:
        S[int_name][2] = param["S"][int_name]
        S[int_name][3] = param["S"][int_name]
    P[int_name] = S[int_name] * P["knowledge"]
    return P[int_name]

##Functions related to Dashboard###
def reset_flags():
    flags = {
        'flag_SDR': 0,
        'flag_PROMPTS': 0,
        'flag_pulse': 0,
        'flag_fqa': 0,
        #SDR Demand
        'flag_CHV': 0,
        'flag_pushback': 0,
        'flag_ANC': 0,
        'flag_LB': 0,
        #SDR Supply
        'flag_performance': 0,
        'flag_MENTOR': 0,
        'flag_capacity': 0,
        'flag_labor': 0,
        'flag_refer_voucher': 0,
        'flag_transfer_capacity': 0,
        'flag_transfer_delay': 0,
        'flag_equipment': 0,
        #Treatment
        'flag_pph_bundle': 0,
        'flag_iv_iron': 0,
        'flag_MgSO4': 0,
        'flag_antibiotics': 0,
        'flag_oxytocin': 0,
        #Diagnosis
        'flag_us': 0,
        'flag_sdr': 0,
        'flag_intrasensor': 0,
        'flag_sensor_ai': 0,
    }
    return flags

def reset_E(): 
    E = {}
    E["sens_us"] = 0.95
    E["spec_us"] = 0.95
    E["sens_sensor"] = 0.95
    E["spec_sensor"] = 0.95
    return E

def reset_HSS(slider_params):
    HSS = {}
    HSS["referadded"] = 0
    HSS["transadded"] = 0
    HSS["capacity_added"] = 0
    HSS["knowledge"] = slider_params['base_knowledge_L45_slider'] #knowledge in L4/5
    HSS["supply_level"] = 0
    HSS["P_L45"] = slider_params['base_p_45_slider']
    HSS["P_ANC"] = slider_params['p_ANC_base_slider']
    HSS["P_refer"] = 0
    HSS["transfer_capacity_target"] = 0.0
    HSS["labor_ratio"] = 0
    HSS["sensor_ratio"] = 0
    HSS['CHV_memory'] = "Always Forget"  # Default memory model for CHVs
    HSS['tau_decay'] = 6
    HSS['adoption_prompts'] = 0.0
    HSS['chv_engagement'] = 0.0
    HSS['prompts_effect'] = 0.0
    HSS['pulse_coverage'] = 0.0
    HSS['pulse_effectiveness'] = 0.0
    HSS['fqa_pulse_modifier_level'] = "Medium"
    HSS['fqa_pulse_modifier'] = 0.2
    HSS['pulse_implementation_boost_level'] = "Medium"
    HSS['pulse_implementation_boost'] = 0.5
    return HSS

def reset_S(slider_params):
    S = {}
    S["pph_bundle"] = slider_params['S_pph_bundle_slider'][3]
    S["iv_iron"] = slider_params['S_iv_iron_slider']
    S["MgSO4"] = slider_params['S_MgSO4_slider'][3]
    S["antibiotics"] = slider_params['S_antibiotics_slider'][3]
    S["oxytocin"] = slider_params['S_oxytocin_slider'][3]
    S["US"] = 0
    S["Sensors"] = np.array([0, 0, 0, 0])
    return S

def sample_from_ci(value, lower, upper, n=None, kind='proportion', size=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if kind == 'proportion':
        if n is None:
            se = (upper - lower) / (2 * 1.96)
            a, b = (0 - value) / se, (1 - value) / se
            samples = truncnorm.rvs(a, b, loc=value, scale=se, size=size, random_state=rng)
        else:
            alpha = value * n + 1
            beta = (1 - value) * n + 1
            samples = rng.beta(alpha, beta, size=size)

    elif kind in ['RR', 'OR']:
        mu = np.log(value)
        se = (np.log(upper) - np.log(lower)) / (2 * 1.96)
        samples = np.exp(rng.normal(loc=mu, scale=se, size=size))

    elif kind == 'mean':
        se = (upper - lower) / (2 * 1.96)
        samples = rng.normal(loc=value, scale=se, size=size)
    else:
        raise ValueError("Unsupported 'kind'. Choose from 'proportion', 'RR', 'OR', or 'mean'.")
    return samples


def generate_negative_experience_heard(
    rng, num_mothers, n_CHV, mothers_per_CHV,
    track, i, tau_decay, p_CHV_soften_spread,
    memory_model
):
    """
    Generate negative_experience_heard for mothers based on CHV network and memory decay.

    Parameters:
    - rng: random generator
    - num_mothers: number of mothers
    - n_CHV: number of CHVs
    - mothers_per_CHV: number of mothers each CHV can link to (average)
    - track: tracking dictionary containing CHV_negative_Track and CHV_memory_Track
    - i: current month index
    - tau_decay: memory decay parameter controlling CHV's memory strength
    - p_CHV_soften_spread: baseline probability of spreading information
    - memory_model: "logistic", "always_remember", "always_forget"

    Returns:
    - negative_experience_heard: (0/1) array for each mother
    - CHV_IDs: array mapping each mother to their CHV (-1 if no CHV)
    - updated CHV_negative_experience, CHV_memory_age
    """

    # Initialize CHV IDs
    # Clamped to num_mothers: county-wide CHV counts can imply more linked
    # mothers than are actually in a given month's cohort (e.g. smaller
    # counties/months), which would otherwise desync the length of
    # linked_mother_indices (bounded by num_mothers) from CHV_assignments.
    n_CHV = int(n_CHV)
    total_CHV_linked_mothers = min(round(n_CHV * mothers_per_CHV), num_mothers)
    CHV_IDs = np.full(num_mothers, -1, dtype=int)  # -1 means no CHV

    # Randomly assign CHV IDs
    perm_mothers = rng.permutation(num_mothers)
    linked_mother_indices = perm_mothers[:total_CHV_linked_mothers]

    repeats = int(np.ceil(total_CHV_linked_mothers / n_CHV)) if n_CHV > 0 else 0
    CHV_assignments = np.tile(np.arange(n_CHV), int(np.ceil(mothers_per_CHV)))
    rng.shuffle(CHV_assignments)
    CHV_assignments = CHV_assignments[:total_CHV_linked_mothers]

    CHV_IDs[linked_mother_indices] = CHV_assignments

    # Load previous CHV experience and memory
    CHV_negative_experience = track['CHV_negative_Track'][i-1, :].copy()
    CHV_memory_age = track['CHV_memory_Track'][i-1, :].copy() + 1

    # --- Memory strength by model ---
    if memory_model == "Logistic Decay":
        x = np.clip(CHV_memory_age, 0, tau_decay + 1)
        memory_strength = 1 / (1 + np.exp((x - tau_decay / 2) / (0.1 * tau_decay)))
    elif memory_model == "Always Remember":
        memory_strength = np.ones(n_CHV)
    elif memory_model == "Always Forget":
        memory_strength = np.zeros(n_CHV)

    # --- Spread information to mothers ---
    negative_experience_heard = np.zeros(num_mothers, dtype=int)

    # For mothers linked to a CHV with negative experience
    linked_CHV_valid_mask = (CHV_IDs >= 0) & (CHV_negative_experience[CHV_IDs] == 1)

    # For these mothers: calculate their final spread probability
    memory_strength_per_mother = memory_strength[CHV_IDs]  # map CHV memory strength to mothers
    memory_strength_per_mother[CHV_IDs == -1] = 0  # For unlinked mothers

    # Overall probability = baseline soften spread × memory strength
    final_spread_probability = p_CHV_soften_spread * memory_strength_per_mother

    # Draw random numbers for ALL mothers
    spread_random_draw = rng.random(num_mothers)

    # Spread if random draw < probability and CHV has negative experience
    spread_success_mask = linked_CHV_valid_mask & (spread_random_draw < final_spread_probability)
    negative_experience_heard[spread_success_mask] = 1

    return negative_experience_heard, CHV_IDs, CHV_negative_experience, CHV_memory_age
