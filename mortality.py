import numpy as np
from LB_effect import clip01
from global_func import baseline_p_death, P_intervention
import streamlit as st


def apply_transfer_delay_intervention(delay_probs, coverage, shift_2plus, shift_1_2):
    """Shift covered transfers down one delay category without changing transfer count."""
    p_lt1, p_1_2, p_2plus = np.asarray(delay_probs, dtype=float)
    coverage = np.clip(float(coverage), 0.0, 1.0)
    shift_2plus = np.clip(float(shift_2plus), 0.0, 1.0)
    shift_1_2 = np.clip(float(shift_1_2), 0.0, 1.0)

    move_2plus_to_1_2 = coverage * shift_2plus * p_2plus
    move_1_2_to_lt1 = coverage * shift_1_2 * p_1_2
    adjusted = np.array([
        p_lt1 + move_1_2_to_lt1,
        p_1_2 - move_1_2_to_lt1 + move_2plus_to_1_2,
        p_2plus - move_2plus_to_1_2,
    ])
    return adjusted / adjusted.sum()


def assign_transfer_delay_categories(param, flags, rng, num_mothers, transfer_mask, i_loc):
    travel_time_transfer = np.full(num_mothers, -1, dtype=int)  # -1: no transfer, 0: <1h, 1: 1-2h, 2: 2+h
    from_l23_mask = transfer_mask & (i_loc == 1)
    from_l45_mask = transfer_mask & ((i_loc == 2) | (i_loc == 3))

    delay_probs_l23 = np.asarray(param["transfer_delay_probs_l23"], dtype=float)
    delay_probs_l45 = np.asarray(param["transfer_delay_probs_l45"], dtype=float)
    if flags.get("flag_transfer_delay", 0):
        coverage = param.get("referral_implementation_index", 0.0)
        delay_probs_l23 = apply_transfer_delay_intervention(
            delay_probs_l23,
            coverage,
            param.get("transfer_delay_shift_2plus", 0.60),
            param.get("transfer_delay_shift_1_2", 0.40),
        )
        delay_probs_l45 = apply_transfer_delay_intervention(
            delay_probs_l45,
            coverage,
            param.get("transfer_delay_shift_2plus", 0.60),
            param.get("transfer_delay_shift_1_2", 0.40),
        )
    delay_probs_l23 = delay_probs_l23 / delay_probs_l23.sum()
    delay_probs_l45 = delay_probs_l45 / delay_probs_l45.sum()

    # Fixed size=num_mothers (not size=mask.sum()) so this always consumes the
    # same number of random draws regardless of how many mothers are actually
    # eligible -- otherwise an intervention that changes transfer eligibility
    # (including the delay-shift adjustment above) shifts the rng state for
    # every draw later in the month, for everyone, not just the mothers whose
    # eligibility or delay category changed.
    l23_delay_draws = rng.choice(
        [0, 1, 2],
        size=num_mothers,
        p=delay_probs_l23,
    )
    l45_delay_draws = rng.choice(
        [0, 1, 2],
        size=num_mothers,
        p=delay_probs_l45,
    )

    travel_time_transfer[from_l23_mask] = l23_delay_draws[from_l23_mask]
    travel_time_transfer[from_l45_mask] = l45_delay_draws[from_l45_mask]

    return travel_time_transfer


def apply_transfer_delay_rr(p_death, transfer_update_mask, travel_time_transfer, param):
    transfer_delay_rr = np.ones_like(p_death, dtype=float)
    rr_scale = float(param.get("transfer_delay_rr_scale", 1.0))

    delay_lt1_mask = transfer_update_mask & (travel_time_transfer == 0)
    delay_1_2_mask = transfer_update_mask & (travel_time_transfer == 1)
    delay_2plus_mask = transfer_update_mask & (travel_time_transfer == 2)

    transfer_delay_rr[delay_lt1_mask] = rr_scale
    transfer_delay_rr[delay_1_2_mask] = rr_scale * param["RR_transfer_delay_1_2"]
    transfer_delay_rr[delay_2plus_mask] = rr_scale * param["RR_transfer_delay_2plus"]

    p_death[transfer_update_mask] *= transfer_delay_rr[transfer_update_mask]
    return p_death

def initialize_MM_params_vectorized(track, param, flags, i, MC, M, NC):
    MC = MC
    M = M
    NC = NC
    P = {}  # dict to restore probabilities
    n = {}  # dict to restore counts
    E = {}   # dict to restore effects
    OR = {}  # dict to restore odds ratios
    W = {}  # dict to restore weights
    MD = {}  # dict to restore maternal deaths
    ND = {}  # dict to restore neonatal deaths

    n["LB_L"] = track['LB_Track'][i, :].astype(int)  # number of live births by facility level

    # % Death
    P["D_RDS"] = param["D_RDS"]
    P["D_IVH"] = param["D_IVH"]
    P["D_NEC"] = param["D_NEC"]
    P["D_Sepsis"] = param["D_Sepsis"]
    P["D_asphyxia"] = param["D_asphyxia"]
    p_mat_death_baseline, p_neo_death_baseline = baseline_p_death(track, M, param, flags, i, n)
    P["mat_death_labor"] = np.array(p_mat_death_baseline)
    P["neo_death_labor"] = np.array(p_neo_death_baseline)

    # The effect of transfer, severity of complications, planned c-section, emergency c-section
    P['MM_home'] = param['p_MM_home']  # baseline probability of maternal death at home
    P['NM_home'] = param['p_NM_home']  # baseline probability of neonatal death at home
    W["weight_facility_mat"] = param['weight_facility_mat']  # weight of facility in contributing to maternal death
    W["weight_facility_neo"] = param['weight_facility_neo']  # weight of facility in contributing to neonatal death
    OR["MM_CSvsSVD"] = param["OR_MM_CSvsSVD"]  # odds ratio of maternal death for CS vs SVD
    OR["MM_EmCSvsELCS"] = param["OR_MM_EmCSvsELCS"]  # odds ratio of maternal death for emergency CS vs elective CS
    OR['MM_transfer'] = param['OR_MM_transfer']  # odds ratio of maternal death for transfer

    # % Initialize counters
    keys_MD = ["death"]
    keys_ND = ["death"]

    for key in keys_MD:
        MD[key] = np.zeros(4)
    for key in keys_ND:
        ND[key] = np.zeros(4)
    return P, n, MC, M, NC, E, OR, MD, ND, W

def f_MM_vectorized(track, param, flags, i, MC, M, NC, individual_outcomes, rng):


    # Initialize parameters and counters
    P, n, MC, M, NC, E, OR, MD, ND, W = initialize_MM_params_vectorized(track, param, flags, i, MC, M, NC)

    # Extract individual-level data
    i_loc = individual_outcomes["i_loc"].values                  # original delivery location index before emergency transfers (0-3)
    i_loc_new_v2 = individual_outcomes["i_loc_new_v2"].values  # final delivery location index (0-3)
    i_mod = individual_outcomes["i_mod"].values                # mode of delivery (SVD, EmCS, ELCS)
    i_transfer_pred = individual_outcomes["i_transfer_pred"].values     # whether first-time emergency transfer status (0/1)
    i_transfer_actual = individual_outcomes["i_transfer_actual"].values # whether second-time emergency transfer status (0/1)
    i_severe_new = individual_outcomes["i_severe_new"].values           # whether severe complications (0/1)
    i_RDS = individual_outcomes["i_RDS"].values                         # whether neonatal RDS (0/1)
    i_IVH = individual_outcomes["i_IVH"].values                         # whether neonatal IVH (0/1)
    i_NEC = individual_outcomes["i_NEC"].values                         # whether neonatal NEC (0/1)
    i_neo_sepsis = individual_outcomes["i_neo_sepsis"].values           # whether neonatal sepsis (0/1)
    i_asphyxia = individual_outcomes["i_asphyxia"].values               # whether neonatal asphyxia (0/1)
    i_pph_severe = individual_outcomes["i_pph_severe_new"].values       # whether severe PPH (0/1)
    i_sepsis_severe = individual_outcomes["i_sepsis_severe"].values     # whether severe maternal sepsis (0/1)
    i_eclampsia_severe = individual_outcomes["i_eclampsia_severe"].values # whether severe eclampsia (0/1)
    i_ol_severe = individual_outcomes["i_ol_severe"].values               # whether severe obstructed labor (0/1)
    i_aph_severe = individual_outcomes["i_aph_severe"].values             # whether severe APH (0/1)
    num_mothers = i_loc_new_v2.shape[0]                                 # total number of mothers in this iteration

    i_transfer = ((i_transfer_pred == 1) | (i_transfer_actual == 1))    # combine transfer status - whether emergency transfer (0/1)
    i_CS = np.isin(i_mod, ["EmCS", "ELCS"]).astype(int)    # whether cesarean section (0/1)

    severe_mask = (i_severe_new == 1) # mask for severe complications
    home_mask = (i_loc_new_v2 == 0)   # mask for home deliveries
    facility_mask = (~home_mask)      # mask for facility deliveries
    CS_mask = (i_CS == 1)             # mask for cesarean sections
    EmCS_mask = (i_mod == "EmCS")     # mask for emergency cesarean sections
    transfer_mask = (i_transfer == 1)  # mask for emergency transfers
    neo_coms_mask = ((i_RDS == 1) | (i_IVH == 1) | (i_NEC == 1) | (i_neo_sepsis == 1) | (i_asphyxia == 1)) # mask for neonatal complications

    #Maternal Deaths - different weight version
    death_cause = np.full(num_mothers, "none", dtype=object)

    #  tracking: attenuate PPH/APH death weights (dashboard: flag_blood, blood_adoption)
    flag_blood_tracking = float(
        flags.get("flag_blood_tracking", flags.get("flag_blood", 0))
    )

    blood_adoption = float(param.get("HSS", {}).get("blood_adoption", 0.0))

    # Blood tracking is a facility-based intervention: no effect for home
    # deliveries (blood_mult == 1.0 there), full effect at facilities.
    # Per-mother array rather than a mask multiply on comp_risks -- multiplying
    # comp_risks by a 0/1 home mask would zero out the entire PPH/APH severity
    # term for home mothers (removing their PPH/APH mortality risk entirely),
    # not just neutralize the blood effect.
    blood_mult = np.where(home_mask, 1.0, 1.0 - flag_blood_tracking * blood_adoption)

    # Complication risk weights (constant across locations)
    comp_names = ["pph", "sepsis", "eclampsia", "ol", "other", "aph"]
    comp_risks = np.stack([
        i_pph_severe * param["MM_weight_pph"] * blood_mult,
        i_sepsis_severe * param["MM_weight_sepsis"],
        i_eclampsia_severe * param["MM_weight_eclampsia"],
        i_ol_severe * param["MM_weight_ol"],
        np.ones(num_mothers, dtype=int) * param["p_MM_others"],
        i_aph_severe * param["MM_weight_aph"] * blood_mult,
    ], axis=1)

    # Get max risk index and value for each mother
    max_risk_indices = np.argmax(comp_risks, axis=1)
    base_comp_risks = comp_risks[np.arange(num_mothers), max_risk_indices]

    # Location-based modifier
    location_modifier = np.zeros(num_mothers, dtype=float)
    location_modifier[home_mask] = P["MM_home"]
    location_modifier[facility_mask] = W['weight_facility_mat'] * P["mat_death_labor"][i_loc_new_v2[facility_mask]]

    p_death = base_comp_risks * location_modifier       # Initial death probability based on max complication risk and location

    # **Step 2: Apply Odds Ratios (OR) for CS & EmCS**
    cs_update_mask = severe_mask & CS_mask             # Mask for severe cases with cesarean section
    p_death[cs_update_mask] = (
            OR["MM_CSvsSVD"] * p_death[cs_update_mask] /
            ((1 - p_death[cs_update_mask]) + (OR["MM_CSvsSVD"] * p_death[cs_update_mask]))
    )  # Update death probability for CS cases

    # **Step 3: Apply OR for Emergency CS (EmCS)**
    emcs_update_mask = severe_mask & EmCS_mask        # Mask for severe cases with emergency cesarean section
    p_death[emcs_update_mask] = (
            OR["MM_EmCSvsELCS"] * p_death[emcs_update_mask] / (
                (1 - p_death[emcs_update_mask]) + (OR["MM_EmCSvsELCS"] * p_death[emcs_update_mask]))
    )  # Update death probability for EmCS cases

    # **Step 4: Apply transfer delay RRs**
    transfer_update_mask = severe_mask & transfer_mask  # Mask for severe cases with emergency transfer
    travel_time_transfer = assign_transfer_delay_categories(param, flags, rng, num_mothers, transfer_mask, i_loc)
    p_death = apply_transfer_delay_rr(p_death, transfer_update_mask, travel_time_transfer, param)

    # **Step 3: Clip Death Probabilities & Assign Maternal Deaths**
    p_death = np.clip(p_death, 0, 1)  # Ensure probabilities stay within [0,1]
    i_mort = (rng.random(num_mothers) < p_death).astype(int) # Assign deaths

    # Assign death cause
    death_cause[i_mort == 1] = np.array(comp_names)[max_risk_indices[i_mort == 1]]

    # Neonatal Deaths -- need recalibration
    # **Step 1: Initialize Neonatal Death Probability & Outcome**
    p_neo_death = np.zeros(num_mothers, dtype=float)

    # **Step 2: Assign baseline probability based on location**
    p_neo_death[home_mask & neo_coms_mask] = P["NM_home"]        # Home deliveries with complications
    p_neo_facility_death = W['weight_facility_neo'] * (P["neo_death_labor"][i_loc_new_v2])  # Facility-based death probability
    p_neo_death[facility_mask & neo_coms_mask] = p_neo_facility_death[facility_mask & neo_coms_mask] # Facility deliveries with complications

    # **Step 5: Apply Transfer Effect**
    p_neo_death[transfer_mask] = param['OR_NM_transfer'] * p_neo_death[transfer_mask] / (
            (1 - p_neo_death[transfer_mask]) + (param['OR_NM_transfer'] * p_neo_death[transfer_mask])
    ) # Update neonatal death probability for transfer cases

    # **Step 6: Assign Neonatal Deaths**
    i_ND = (rng.random(num_mothers) < np.clip(p_neo_death, 0, 1)).astype(int)  # Clip & sample ND

    # Update counters
    i_mort = i_mort.astype(int)
    i_ND = i_ND.astype(int)
    np.add.at(MD["death"], i_loc_new_v2, i_mort)  # Update maternal deaths per facility
    np.add.at(ND["death"], i_loc_new_v2, i_ND)  # Update neonatal deaths per facility

    #update individual outcomes
    individual_outcomes["i_mat_death"] = i_mort.astype(int)
    individual_outcomes["i_neo_death"] = i_ND.astype(int)
    individual_outcomes["i_transfer"] = i_transfer.astype(int)
    individual_outcomes["travel_time_transfer"] = travel_time_transfer.astype(int)
    individual_outcomes["death_cause"] = death_cause.astype(str)

    return MC, MD, NC, ND, M, individual_outcomes
