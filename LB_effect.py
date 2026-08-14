def clip01(x):
    return float(np.clip(x, 0.0, 1.0))

def odds_update(p, OR):
    p = clip01(p)
    OR = float(OR)
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    odds = p / (1.0 - p)
    odds_new = odds * OR
    return clip01(odds_new / (1.0 + odds_new))

def pulse_effect(
    indicator_of_interest,
    targeted_indicator,
    current_value,
    target_value,
    flags,
    param,
    clip_min=None,
    clip_max=None,
):
    if targeted_indicator != indicator_of_interest or not flags.get("flag_pulse", 0):
        return current_value

    pulse_influence_strength = float(param["pulse_influence_strength"])
    fqa_pulse_modifier = float(param["fqa_pulse_modifier"])
    # pulse_coverage = float(param.get("HSS", {}).get("pulse_coverage", 1.0))
    pulse_implementation_index = float(param["pulse_implementation_index"])
    # flag_fqa = float(flags.get("flag_fqa", 0))
    # Pre-boost snapshot (sync_param_momish_from_hss), not the live
    # fqa_implementation_index -- otherwise PULSE's own strength would be
    # inflated by its own boost to FQA's implementation index, a circular
    # amplification the fqa_pulse_modifier term was never calibrated for.
    fqa_implementation_index = float(param.get("fqa_pulse_amplifier_index", param["fqa_implementation_index"]))
    pulse_influence_strength_effective = np.clip(
        pulse_implementation_index * pulse_influence_strength * (1 + fqa_implementation_index * fqa_pulse_modifier),
        0,
        1,
    )

    indicator_gap = max(current_value - target_value, 0)
    new_value = current_value - pulse_influence_strength_effective * indicator_gap

    if clip_min is not None or clip_max is not None:
        low = -np.inf if clip_min is None else clip_min
        high = np.inf if clip_max is None else clip_max
        new_value = np.clip(new_value, low, high)

    return new_value

import random
import numpy as np
import math
import streamlit as st
import pandas as pd
from global_func import risk_stratification, move_function, generate_negative_experience_heard

def f_LB_effect_vectorized(param, i, track, flags, int_period, rng):
    LB_base = track['LB_Track'][0]
    #ANC phase - affect LB
    individual_outcomes, negative_experience_heard, CHV_IDs, CHV_negative_experience, CHV_memory_age = f_ANC_LB_effect_vectorized(track, LB_base, param, flags, i, int_period, rng)

    #SDR intervention - shifting of live births to L4/5
    individual_outcomes = shifting_live_births_vectorized(individual_outcomes, param, i, track, flags, int_period, rng, negative_experience_heard, CHV_IDs, CHV_negative_experience, CHV_memory_age)

    # Initialize counters as NumPy arrays (avoid dictionaries)
    n_LB_new = np.zeros(4, dtype=int)
    n_ANC_new = np.zeros(4, dtype=int)
    n_highrisk_new = np.zeros(4, dtype=int)
    n_free_referrals = np.zeros(2, dtype=int)
    n_self_referrals = np.zeros(2, dtype=int)

    #Extract individual outcomes
    i_loc = individual_outcomes['i_loc'].values
    i_risk = individual_outcomes['i_risk'].values
    i_ANC = individual_outcomes['i_ANC'].values
    i_free_referral = individual_outcomes['i_free_referral'].values
    i_self_referral = individual_outcomes['i_self_referral'].values
    i_class = individual_outcomes['i_class'].values

    #Update counters
    np.add.at(n_LB_new, i_loc, 1)
    np.add.at(n_highrisk_new, i_loc, i_risk)
    np.add.at(n_ANC_new, i_loc, i_ANC)
    np.add.at(n_free_referrals, i_class, i_free_referral)
    np.add.at(n_self_referrals, i_class, i_self_referral)

    #update whole county outcomes at time i
    track['LB_Track'][i] = n_LB_new
    track['HighRisk_Track'][i] = n_highrisk_new
    track['ANC_Track'][i] = n_ANC_new

    # st.text(track['LB_Track'][i] / np.sum(track['LB_Track'][i]))

    return track, n_free_referrals, n_self_referrals, individual_outcomes

#Optimzed version
def f_ANC_LB_effect_vectorized(track, LB_base, param, flags, i, int_period, rng):
    ##-------------------Parameter initialization-------------------##
    Capacity = track['Facility_Capacity_Track'][i, 0]
    E_Preterm_LMP = np.zeros(3)
    E_Preterm_LMP[0], E_Preterm_LMP[1] = param['E_Preterm_LMP'][0], param['E_Preterm_LMP'][1]
    E_Preterm_LMP[2] = 1 - (E_Preterm_LMP[0] + E_Preterm_LMP[1])
    E_Postterm_LMP = np.zeros(3)
    E_Postterm_LMP[0], E_Postterm_LMP[1] = param['E_Postterm_LMP'][0], param['E_Postterm_LMP'][1]
    E_Postterm_LMP[2] = 1 - (E_Postterm_LMP[0] + E_Postterm_LMP[1])
    num_mothers = np.sum(LB_base.astype(int))

    ##-------------------Intervention Set Up----------------------##
    flag_ANC = flags["flag_ANC"]
    flag_POCUS = flags["flag_us"]
    flag_PROMPTS = flags["flag_PROMPTS"]

    # ANC intervention
    P_ANC_base = param['p_ANC_base']
    P_ANC_target = param["HSS"]["P_ANC"]
    if flag_ANC:
        P_ANC = P_ANC_base + (P_ANC_target - P_ANC_base) / (int_period - 1) * i if i < int_period else P_ANC_target
    else:
        P_ANC = P_ANC_base

    # ------------------- PROMPTS BLOCK A: Increase 4+ANC via RR on the covered share ------------------- #
    if flag_PROMPTS:
        # implementation_index = share of mothers reached by PROMPTS (population coverage).
        # That share gets the full scenario RR on 4+ ANC; the rest keep their baseline
        # probability -- the index is a coverage weight, not a scaling factor on the RR.
        prompts_implementation_index = clip01(param.get("prompts_implementation_index", 0.0))
        prompts_rr_anc4p = float(param.get("prompts_rr_anc4p", 1.0))

        P_ANC_treated = clip01(P_ANC * prompts_rr_anc4p)
        P_ANC = clip01((1 - prompts_implementation_index) * P_ANC + prompts_implementation_index * P_ANC_treated)

        engagement_level = prompts_implementation_index
    else:
        engagement_level = 0.0

    # POCUS intervention
    i_pocus = np.zeros(num_mothers, dtype=int)
    if flag_POCUS:
        sen_risk = param['E']["sens_us"]
        spec_risk = param['E']["spec_us"]
    else:
        sen_risk = param["sen_risk_trad"]
        spec_risk = param["spec_risk_trad"]
        if flags.get("flag_PROMPTS", 0):
            sen_target = param.get("sen_risk_trad_target", sen_risk)
            spec_target = param.get("spec_risk_trad_target", spec_risk)
            sen_risk = clip01(sen_risk + (sen_target - sen_risk) * engagement_level)
            spec_risk = clip01(spec_risk + (spec_target - spec_risk) * engagement_level)

    if not flag_POCUS:
        p_elec_CS_highrisk = param["p_elec_CS|highrisk"]                  # probability of elective CS given high risk
    else:
        p_elec_CS_highrisk = param["p_elec_CS|highrisk_us"]

    ##-------------------Mothers' Initial Conditions----------------------##
    i_class = (rng.random(num_mothers) < param["class"]).astype(int)       # 1 = high SES, 0 = low SES
    i_risk = (rng.random(num_mothers) < param["p_highrisk"]).astype(int)            # 1 = high risk, 0 = low risk
    i_ANC = (rng.random(num_mothers) < P_ANC).astype(int)                  # 1 = received ANC, 0 = no ANC

    ##-------------------Gestational Age Assignment - based on ANC status----------------------###
    GA_anc_cumsum = np.cumsum(param["GA_anc"])
    GA_noanc_cumsum = np.cumsum(param["GA_noanc"])
    rand_vals = rng.random(num_mothers)
    i_jGA = np.zeros(num_mothers, dtype=int)
    i_jGA[i_ANC == 1] = np.searchsorted(GA_anc_cumsum, rand_vals[i_ANC == 1])
    i_jGA[i_ANC == 0] = np.searchsorted(GA_noanc_cumsum, rand_vals[i_ANC == 0])
    i_GA = param['GA_sequence'][i_jGA]
    i_term_status = np.select([i_GA < 37, i_GA >= 43], [0, 2], default=1) #preterm atterm or postterm

    ##-------------------Risk Stratification in ANC----------------------##
    i_risk_pred = risk_stratification(i_risk, i_ANC, num_mothers, sen_risk, spec_risk, rng)

    ##-------------------Gestational Age Estimation in ANC----------------------##
    # Pre-generate all randomness - ensure both scenarios consume the same amount of seeds
    ga_noise = rng.normal(0, 1, size=num_mothers)  # always draw this
    rand_preterm = rng.random(num_mothers)  # always draw this
    rand_postterm = rng.random(num_mothers)  # always draw this
    if flag_POCUS:
        i_GA_approx = i_GA + param["E_GA_US"][0] + param["E_GA_US"][1] * ga_noise
        i_preterm_pred = (i_GA_approx < 37) & (i_ANC == 1)
        i_postterm_pred = (i_GA_approx >= 43) & (i_ANC == 1)
        i_pocus[i_ANC == 1] = 1
    else:
        i_preterm_pred = (rand_preterm < E_Preterm_LMP[i_term_status]) & (i_ANC == 1)
        i_postterm_pred = (rand_postterm < E_Postterm_LMP[i_term_status]) & (i_ANC == 1)

    # st.text(param["HSS"]['CHV_memory'])

    ##-------------------Delivery Location Selection----------------------##
    negative_experience_heard, CHV_IDs, CHV_negative_experience, CHV_memory_age = generate_negative_experience_heard(
        rng=rng,
        num_mothers=num_mothers,
        n_CHV=param['n_CHV'],
        mothers_per_CHV=param['mothers_per_CHV_permonth'],
        track=track,
        i=i,
        tau_decay=param["HSS"]['tau_decay'],
        p_CHV_soften_spread=param['p_CHV_soften_spread'],
        memory_model=param["HSS"]['CHV_memory']
    )

    P_home_noANC = param["home_noANC"]
    P_L45_noANC = (1 - P_home_noANC) * param["l45_fac"]
    P_L23_noANC = 1 - P_home_noANC - P_L45_noANC
    P_home_lowrisk = param["home_lowrisk"]
    P_close_to_L23 = param["close_to_L23"]
    P_close_to_L45 = 1 - P_close_to_L23
    P_L23_lowrisk = (1 - P_home_lowrisk) * P_close_to_L23
    P_L45_lowrisk = (1 - P_home_lowrisk) * P_close_to_L45
    P_L23_highrisk = param["L23_highrisk"]
    P_L45_highrisk = 1 - P_L23_highrisk
    prob_matrix_noANC = np.array([P_home_noANC, P_L23_noANC, P_L45_noANC])
    prob_matrix_lowrisk = np.array([P_home_lowrisk, P_L23_lowrisk, P_L45_lowrisk])
    prob_matrix_highrisk = np.array([0, P_L23_highrisk, P_L45_highrisk])

    def adjust_probabilities(prob_matrix, RR_l45):
        """Internal function to adjust a probability matrix by Risk Ratio for L4/5"""
        p_home = prob_matrix[0]
        p_l23 = prob_matrix[1]
        p_l45 = prob_matrix[2]

        # Reduce L4/5 probability
        p_l45_new = p_l45 * RR_l45
        loss = p_l45 - p_l45_new
        redistribution_factor = p_home + p_l23
        p_home_new = p_home + loss * (p_home / redistribution_factor)
        p_l23_new = p_l23 + loss * (p_l23 / redistribution_factor)

        adjusted_matrix = np.array([p_home_new, p_l23_new, p_l45_new])
        return adjusted_matrix / adjusted_matrix.sum()

    RR_l45_poorQOC = param['RR_l45_poorQOC']

    # Adjusted probability matrices

    # distribution of mothers to delivery location - no ANC 
    # distrubition of mothers to delivery location - low-risk
    # distrubition of mothers to delivery location - high-risk
    prob_matrix_noANC_adj = adjust_probabilities(prob_matrix_noANC, RR_l45_poorQOC)
    prob_matrix_lowrisk_adj = adjust_probabilities(prob_matrix_lowrisk, RR_l45_poorQOC)
    prob_matrix_highrisk_adj = adjust_probabilities(prob_matrix_highrisk, RR_l45_poorQOC)

    #define the masks for mothers with different ANC and prediction status
    pred_highrisk_mask = ((i_risk_pred == 1) | (i_preterm_pred == 1) | (i_postterm_pred == 1)).astype(bool)
    noanc_mask = (i_ANC == 0).astype(bool)
    anc_highrisk_mask = ((i_ANC == 1) & (pred_highrisk_mask)).astype(bool)
    anc_lowrisk_mask = ((i_ANC == 1) & (~pred_highrisk_mask)).astype(bool)

    # Define experience masks
    positive_mask = (negative_experience_heard == 0)
    negative_mask = (negative_experience_heard == 1)

    # Initialize
    i_loc = np.zeros(num_mothers, dtype=int)

    # No ANC
    i_loc_noanc_pos = rng.choice(3, p=prob_matrix_noANC, size=num_mothers)
    i_loc_noanc_neg = rng.choice(3, p=prob_matrix_noANC_adj, size=num_mothers)

    # ANC Lowrisk
    i_loc_anc_lowrisk_pos = rng.choice(3, p=prob_matrix_lowrisk, size=num_mothers)
    i_loc_anc_lowrisk_neg = rng.choice(3, p=prob_matrix_lowrisk_adj, size=num_mothers)

    # ANC Highrisk
    i_loc_anc_highrisk_pos = rng.choice(3, p=prob_matrix_highrisk, size=num_mothers)
    i_loc_anc_highrisk_neg = rng.choice(3, p=prob_matrix_highrisk_adj, size=num_mothers)

    # Assign locations based on ANC and risk status
    i_loc[noanc_mask & positive_mask] = i_loc_noanc_pos[noanc_mask & positive_mask]
    i_loc[noanc_mask & negative_mask] = i_loc_noanc_neg[noanc_mask & negative_mask]
    i_loc[anc_lowrisk_mask & positive_mask] = i_loc_anc_lowrisk_pos[anc_lowrisk_mask & positive_mask]
    i_loc[anc_lowrisk_mask & negative_mask] = i_loc_anc_lowrisk_neg[anc_lowrisk_mask & negative_mask]
    i_loc[anc_highrisk_mask & positive_mask] = i_loc_anc_highrisk_pos[anc_highrisk_mask & positive_mask]
    i_loc[anc_highrisk_mask & negative_mask] = i_loc_anc_highrisk_neg[anc_highrisk_mask & negative_mask]

    # #draw location assignments for all mothers using each strategy
    i_loc_noanc_all = rng.choice(3, p=prob_matrix_noANC, size=num_mothers)
    i_loc_anc_highrisk_all = rng.choice(3, p=prob_matrix_highrisk, size=num_mothers)
    i_loc_anc_lowrisk_all = rng.choice(3, p=prob_matrix_lowrisk, size=num_mothers)
    #apply results only to relevant mothers
    i_loc = np.zeros(num_mothers, dtype=int)
    i_loc[noanc_mask] = i_loc_noanc_all[noanc_mask]
    i_loc[anc_highrisk_mask] = i_loc_anc_highrisk_all[anc_highrisk_mask]
    i_loc[anc_lowrisk_mask] = i_loc_anc_lowrisk_all[anc_lowrisk_mask]


    # # # ## PROMPTS re-allocation ## 
    # if flag_PROMPTS:
    #     # - p_move_home_base: max fraction of home-deliverers that PROMPTS can shift
    #     # - p_to_L45: conditional probability that a shifted mother goes to L4/5 (else L2/
    #     # 3)
    #     p_move_home_base = float(param.get("p_move_home_base", 0.30))
    #     p_move_home = clip01(p_move_home_base * engagement_level)
    #     p_to_L23 = 0.1
    #     p_to_L45 = 1 - p_to_L23

    #     home_mask = (i_loc == 0)
    #     move_mask = home_mask & (rng.random(num_mothers) < p_move_home)
    #     i_loc[move_mask] = rng.choice(
    #         [1, 2],
    #         size=int(np.sum(move_mask)),
    #         p=[p_to_L23, p_to_L45]
    #     )
    

    #Reallocate mothers if overcapacity
    n_l45 = np.count_nonzero(i_loc == 2)
    exceed_lb = int(max(n_l45 - Capacity, 0))
    shuffled_all = rng.permutation(num_mothers)   # shuffle all mother indices
    l45_indices = np.where(i_loc == 2)[0]         # identify L4/5 mothers
    shuffled_l45 = shuffled_all[np.isin(shuffled_all, l45_indices)]  # filter only those in L4/5 from the shuffled list
    relocate_indices = shuffled_l45[:exceed_lb]                      # select only the top `exceed_lb` mothers to relocate
    mask_relocate_l23 = np.zeros(num_mothers, dtype=bool)            # apply relocation
    mask_relocate_l23[relocate_indices] = True
    i_loc[mask_relocate_l23] = 1
    
    ##-------------------Elective C-section Decision----------------------##
    elcs_mask1 = (i_loc == 2) & (i_risk_pred == 1) & (i_preterm_pred == 0) & (i_ANC == 1)
    elcs_mask2 = (i_loc == 2) & (i_preterm_pred == 1) & (i_ANC == 1)
    i_elcs_case1 = (rng.random(num_mothers) < p_elec_CS_highrisk).astype(int)
    i_elcs_case2 = (rng.random(num_mothers) < param["p_elec_CS|preterm"]).astype(int)
    i_elec_CS = np.zeros(num_mothers, dtype=int)
    i_elec_CS[elcs_mask1] = i_elcs_case1[elcs_mask1]
    i_elec_CS[elcs_mask2] = i_elcs_case2[elcs_mask2]

    ##-------------------Move some live births from l4 to l5----------------------##
    l4_mask = (i_loc == 2)
    l4_to_l5_mask = (rng.random(num_mothers) <  param['p_l5_l45']).astype(bool)
    l4_to_l5 = l4_mask & l4_to_l5_mask
    i_loc[l4_to_l5] = 3
    i_loc = i_loc.astype(int)
    i_self_referral = (i_loc >= 2).astype(int)
    i_free_referral = np.zeros(num_mothers, dtype=int)

    ##------------------ANC-related single interventions------------------------##
    P_knowledge = np.array(
        [0, param['base_knowledge_L23'], param['base_knowledge_L45'], param['base_knowledge_L45']])
    if flags['flag_performance']:
        P_knowledge[2] = param["HSS"]["knowledge"]
        P_knowledge[3] = param["HSS"]["knowledge"]
    # MENTORS: use the parameter-level implementation index from SDR Parameters.
    if flags.get("flag_MENTOR"):
        mentors_coverage = clip01(float(param.get("mentors_implementation_index", 0.0)))
        mentors_knowledge_target = float(param.get("mentors_knowledge_target", 1.0))
        P_knowledge[1:4] = np.clip(
            P_knowledge[1:4]
            + mentors_coverage * (mentors_knowledge_target - P_knowledge[1:4]),
            0.0,
            1.0,
        )
    P_close_to_L23 = param["close_to_L23"]
    P_close_to_L45 = 1 - P_close_to_L23
    P_iv_iron = param["S"]["iv_iron"] * (P_knowledge[1] * P_close_to_L23 + P_knowledge[3] * P_close_to_L45)

    i_anemia = (rng.random(num_mothers) < param['p_anemia_anc'][i_ANC]).astype(int) #rng.binomial(1, param['p_anemia_anc'][i_ANC])         # Anemia status for each mother
    i_anemia_new = i_anemia.copy()
    eligible_iv_iron = (i_ANC == 1) & (i_anemia == 1)                                   # Boolean mask for eligible mothers
    i_iv_iron = np.zeros(num_mothers, dtype=int)
    iv_iron_provided = (rng.random(num_mothers) < P_iv_iron).astype(int)  #rng.binomial(1, P["iv_iron"], size=num_mothers)
    i_iv_iron[eligible_iv_iron] = iv_iron_provided[eligible_iv_iron]     # Ensures only eligible mothers can receive it
    anemia_cured = (i_iv_iron == 1) & (rng.random(num_mothers) < (1 - param['E_iv_iron']))  # Boolean mask for cured cases
    i_anemia_new[anemia_cured] = 0                                                          # Cure anemia for affected mothers

    ##-------------------Update indvidual outcomes----------------------##
    i_preterm = (i_GA < 37).astype(int)
    individual_outcomes = pd.DataFrame({
        'i_loc': i_loc.astype(int),
        'i_ANC': i_ANC.astype(int),
        'i_class': i_class.astype(int),
        'i_risk': i_risk.astype(int),
        'i_GA': i_GA.astype(int),
        'i_preterm': i_preterm.astype(int),
        'i_elec_CS': i_elec_CS.astype(int),
        'i_jGA': i_jGA.astype(int),
        'i_risk_pred': i_risk_pred.astype(int),
        'i_preterm_pred': i_preterm_pred.astype(int),
        'i_pocus': i_pocus.astype(int),
        'i_self_referral': i_self_referral.astype(int),
        'i_free_referral': i_free_referral.astype(int),
        'i_anemia': i_anemia.astype(int),
        'i_anemia_new': i_anemia_new.astype(int),
        'i_iv_iron': i_iv_iron.astype(int),
    })
    return individual_outcomes, negative_experience_heard, CHV_IDs, CHV_negative_experience, CHV_memory_age

#Updated - add CHV effect (need revisions for CHV effect)
def shifting_live_births_vectorized(individual_outcomes, param, i, track, flags, int_period, rng, negative_experience_heard, CHV_IDs, CHV_negative_experience, CHV_memory_age):
    if not flags["flag_LB"]:
        # Dummy RNG usage to consume same number of random draws
        num_mothers = individual_outcomes.shape[0]
        _ = rng.permutation(num_mothers)
        _ = rng.random(num_mothers)
        _ = rng.permutation(num_mothers)
        _ = rng.choice([2, 3], size=num_mothers, p=[0.5, 0.5])
        _ = rng.random(num_mothers)

        individual_outcomes['i_neg_exp_heard'] = negative_experience_heard.astype(int)
        individual_outcomes['i_neg_exp_owned'] = np.zeros(num_mothers, dtype=int)

        return individual_outcomes
    else:
        ##-------------------Parameter initialization-------------------##
        i_loc = individual_outcomes['i_loc'].values
        i_class = individual_outcomes['i_class'].values
        i_self_referral = individual_outcomes['i_self_referral'].values.copy()
        i_free_referral = individual_outcomes['i_free_referral'].values.copy()
        num_mothers = i_loc.shape[0]
        LB_base = track["LB_Track"][0]
        Facility_Capacity = track['Facility_Capacity_Track'][i, 0]
        Referral_Capacity = track['Referral_Capacity_Track'][i, 0]
        p_l5_l45 = param['p_l5_l45']
        p_lb_l23to45 = param['p_lb_l23_45']
        tau_decay = param["HSS"]['tau_decay']
        RR_l45_poorQOC = param['RR_l45_poorQOC']

        ##-------------------Shifting of live births-------------------##
        # Step 1: Compute expected L45 target
        p_l45_base = (LB_base[2] + LB_base[3]) / num_mothers
        p_l45_pre_base = p_l45_base - p_lb_l23to45
        l45_target_growth = (param["HSS"]["P_L45"] - p_l45_pre_base)  # targeted live birth growth at l4/5 before transfer

        if i < int_period:
            p_l45_exp_actual = (p_l45_pre_base + (l45_target_growth / (int_period - 1) * i))
        else:
            p_l45_exp_actual = (p_l45_pre_base + l45_target_growth)

        # st.text("p_l45_exp_actual: " + str(p_l45_exp_actual))
        # p_negative = np.mean(negative_experience_heard)  # Probability of negative experience heard
        # st.text("p_negative: " + str(p_negative))

        # Step 2: Find mothers need to shift to L4/5
        def select_mothers_for_shift(i_loc, num_mothers, p_l45_exp_actual, Facility_Capacity,
                                        rng, negative_experience_heard, RR_l45_poorQOC):
            """
            Updated version:
            - No separate permutation.
            - Fully global random numbers control.
            """

            # Step 1: Identify mothers at home or L2/3
            home_l23_mask = (i_loc == 0) | (i_loc == 1)

            # Step 2: Compute total expected movers
            num_l45_exp = np.ceil(p_l45_exp_actual * num_mothers).astype(int)
            num_l45_bf_shift = np.count_nonzero((i_loc == 2) | (i_loc == 3))
            num_l45_exp_new = max(num_l45_exp - num_l45_bf_shift, 0)

            # Step 3: Random numbers for everyone
            move_random_draw = rng.random(num_mothers)

            # Step 4: Among eligible (home/L2/3) mothers, pick based on random numbers (smallest first)
            eligible_indices = np.where(home_l23_mask)[0]
            eligible_random = move_random_draw[eligible_indices]

            sorted_eligible_indices = eligible_indices[np.argsort(eligible_random)]
            selected_indices = sorted_eligible_indices[:min(num_l45_exp_new, len(sorted_eligible_indices))]

            mothers_need_shift = np.zeros(num_mothers, dtype=int)
            mothers_need_shift[selected_indices] = 1

            # Step 5: Apply negative experience adjustment
            p_move = np.ones(num_mothers)
            p_move[negative_experience_heard == 1] *= RR_l45_poorQOC

            # Step 6: Random draw again for move intention
            final_shift_mask = (move_random_draw < p_move) & (mothers_need_shift == 1)

            # Step 7: Facility capacity constraint
            movers_intended_indices = np.where(final_shift_mask)[0]
            move_random_draw_intended = move_random_draw[movers_intended_indices]

            sorted_movers_indices = movers_intended_indices[np.argsort(move_random_draw_intended)]

            num_intended = len(movers_intended_indices)
            effective_capacity = -pulse_effect(
                "Facility_Capacity",
                "Facility_Capacity",
                -Facility_Capacity,
                -num_l45_exp,
                flags,
                param,
            )
            num_l45_max = int(np.floor(effective_capacity - num_l45_bf_shift))
            num_l45_max = max(num_l45_max, 0)
            num_select_final = min(num_l45_max, num_intended)

            allowed_indices = sorted_movers_indices[:num_select_final]
            constrained_indices = sorted_movers_indices[num_select_final:]

            mothers_allowed_shift = np.zeros(num_mothers, dtype=int)
            mothers_constrained_shift = np.zeros(num_mothers, dtype=int)
            mothers_allowed_shift[allowed_indices] = 1
            mothers_constrained_shift[constrained_indices] = 1

            negative_experience_owned = np.zeros(num_mothers, dtype=int)
            negative_experience_owned[mothers_constrained_shift == 1] = 1

            return mothers_need_shift, final_shift_mask, mothers_allowed_shift, negative_experience_owned

        mothers_need_shift, final_shift_mask, mothers_allowed_shift, negative_experience_owned = select_mothers_for_shift(i_loc, num_mothers, p_l45_exp_actual, Facility_Capacity,
                                     rng, negative_experience_heard, RR_l45_poorQOC)

        # p_reduced_intention = np.sum(final_shift_mask.astype(int)) / np.sum(mothers_need_shift.astype(int))
        # p_allowed = np.sum(mothers_allowed_shift.astype(int)) / np.sum(final_shift_mask.astype(int))
        # st.text(f'month {i}')
        # st.text(f'p_intended: {p_reduced_intention}')
        # st.text(f'p_allowed: {p_allowed}')

        # Step 3: Shift the mothers who are allowed to shift
        i_loc_new = i_loc.copy()
        p_l4_l5 = np.array([(1 - p_l5_l45), p_l5_l45])
        l4_l5 = rng.choice([2, 3], p=p_l4_l5, size=num_mothers)

        def move_function(num_mothers, l4_l5, i_class, i_loc_new,
                          i_free_referral, i_self_referral, Referral_Capacity, flags,
                          mothers_allowed_shift, rng, negative_experience_owned):
            """
                Move allowed mothers to L4/5. Handle referral constraints.
            """
            flag_refer_voucher = flags["flag_refer_voucher"]
            allowed_mask = (mothers_allowed_shift == 1)

            p_free_refer = Referral_Capacity if flag_refer_voucher else 0

            # Random draw for free referral
            free_refer_draws = rng.random(num_mothers)
            move_free_refer_mask = allowed_mask & (free_refer_draws < p_free_refer)

            i_free_referral[move_free_refer_mask] = 1
            i_loc_new[move_free_refer_mask] = l4_l5[move_free_refer_mask]

            # Self-referral for high SES
            no_free_referral_mask = allowed_mask & (~move_free_refer_mask)
            self_referral_possible_mask = no_free_referral_mask & (i_class == 1)

            i_self_referral[self_referral_possible_mask] = 1
            i_loc_new[self_referral_possible_mask] = l4_l5[self_referral_possible_mask]

            # Mothers who fail to move because of no transport
            failed_referral_mask = no_free_referral_mask & (i_class != 1)
            negative_experience_owned[failed_referral_mask] = 1  # Mark these mothers as having negative experience

            return i_loc_new, i_free_referral, i_self_referral, negative_experience_owned

        i_loc_new, i_free_referral, i_self_referral, negative_experience_owned = move_function(num_mothers, l4_l5, i_class, i_loc_new,
                          i_free_referral, i_self_referral, Referral_Capacity, flags,
                          mothers_allowed_shift, rng, negative_experience_owned)

        ##-------------------Apply negative feedback to CHVs-------------------##
        # Extract previous CHV negative experience and memory age
        CHV_negative_experience_prev = track['CHV_negative_Track'][i-1, :].copy()    # Retrieve previous CHV negative experience
        CHV_memory_age_prev = track['CHV_memory_Track'][i - 1, :].copy()
        CHV_negative_experience = CHV_negative_experience_prev.copy()
        CHV_memory_age = CHV_memory_age_prev.copy()

        # Mothers who had new negative ANC_i
        mothers_with_negative = (negative_experience_owned == 1)
        linked_CHVs = CHV_IDs[mothers_with_negative]
        linked_CHVs = linked_CHVs[linked_CHVs >= 0]   # Filter valid CHVs only
        CHV_negative_experience[np.unique(linked_CHVs)] = 1                     # Update CHV negative experience

        # Detect CHVs who newly received negative experience this month
        new_infections = (CHV_negative_experience == 1) & (CHV_negative_experience_prev == 0)
        CHV_memory_age[new_infections] = 0 # Reset memory for newly infected CHVs
        already_infected = (CHV_negative_experience_prev == 1)
        CHV_memory_age[already_infected] += 1

        ##-------------------Update parameters-------------------##
        track['CHV_negative_Track'][i, :] = CHV_negative_experience.copy()
        track['CHV_memory_Track'][i, :] = CHV_memory_age.copy()
        individual_outcomes['i_loc'] = i_loc_new.astype(int)
        individual_outcomes['i_free_referral'] = i_free_referral.astype(int)
        individual_outcomes['i_self_referral'] = i_self_referral.astype(int)
        individual_outcomes['i_neg_exp_heard'] = negative_experience_heard.astype(int)
        individual_outcomes['i_neg_exp_owned'] = negative_experience_owned.astype(int)

        return individual_outcomes
