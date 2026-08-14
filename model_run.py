import numpy as np
import pandas as pd
import random
import time
import streamlit as st
from parameter_loader import reset_inputs
from LB_effect import f_LB_effect_vectorized
from mortality import f_MM_vectorized
from global_func import labor_calculator, fetal_sensor_calculator, DALY_calculator_vectorized
from intrapartum import intrapartum_effect_vectorized


def run_model_dash(param, flags, n_months, int_period, base_seed = None):
    #Initialize Model Parameters
    track = reset_inputs(param, n_months)

    # Create outcome df
    columns = ['Live Births Initial', 'Live Births Final', 'Fac non-CS', 'ANC','L4/5 LBs',
               'Free_referrals', 'Self_referrals', 'Normal_referrals',
                'Mothers with pph_bundle', 'Mothers with iv_iron','Mothers with MgSO4', 'Mothers with antibiotics',
                'Preterm', 'RDS', 'IVH', 'neo_sepsis', 'NEC', 'Neonatal Deaths', 'asphyxia',
                'High risk', 'PL', 'hypoxia', 'OL', 'mat_sepsis', 'pph', 'stillbirths', "eclampsia", 'ruptured_uterus', 'aph', 'severe_comps', 'severe_pph',
                'CS', 'CS_unnessary', 'Elective CS', 'Emergency CS', 'Elective CS risk status', 'Risk status',
                'AVD', 'SVD', 'Anemia', "ER_trans_actual", "ER_trans_pred", "Emergency transfers",
                "Comps after transfer", 'Deaths', 'M_DALYs', 'N_DALYs', 'DALYs',
                #only facility level indicators
                'Facility_capacity_actual', 'Facility_capacity_ideal', 'Capacity Ratio',
                'Surgical_actual', 'Nurse_actual', 'Anesthetist_actual',
                'Surgical_ratio', 'Nurse_ratio', 'Anesthetist_ratio',
                'Surgical_needed', 'Nurse_needed', 'Anesthetist_needed',
                'Doppler_Actual', 'Doppler_Needed', 'Doppler_Ratio',
                'CTG_Actual', 'CTG_Needed', 'CTG_Ratio', 'Month',
        ]
    df = pd.DataFrame(columns=columns)
    df_individual_temp = []
    df_facility_temp = []

    #Run simulation for each time period
    for i in range(n_months):
        if i >= 0:
            if base_seed is not None:
                rng = np.random.default_rng(base_seed[i])

            #Updating features due to intervention changes
            track = update_capacity(track, param, i, flags, int_period)

            # ANC phase
            track, free_refers, self_refers, individual_outcomes = f_LB_effect_vectorized(param, i, track, flags, int_period, rng)

            # Intrapartum phase
            MC, M, NC, individual_outcomes = intrapartum_effect_vectorized(track, flags, param, i, individual_outcomes, rng)

            # Calculate maternal and neonatal health outcomes
            MC, MD, NC, ND, M, individual_df = f_MM_vectorized(track, param, flags, i, MC, M, NC, individual_outcomes, rng)
            individual_df["Month"] = i

            #create an empty dataframe with three rows
            df_facility = pd.DataFrame()
            df_facility["Level"] = np.array([1,2,3])
            df_facility["Month"] = np.array([i,i,i])

            M["Free_referrals"] = free_refers
            M["Self_referrals"] = self_refers
            df.loc[i, 'Month'] = i
            df.loc[i, 'Live Births Initial'] = M["LB_L_initial"]
            df.loc[i, 'Live Births Final'] = track['LB_Track'][i]
            df.loc[i, 'Fac non-CS'] = np.array([0, track['LB_Track'][i][1] - M["CS"][1], track['LB_Track'][i][2] - M["CS"][2], track['LB_Track'][i][3] - M["CS"][3]])
            df.loc[i, 'ANC'] = track['ANC_Track'][i]
            df.loc[i, 'High risk'] = track['HighRisk_Track'][i, :]
            df.loc[i, 'Free_referrals'] = M["Free_referrals"]
            df.loc[i, 'Self_referrals'] = M["Self_referrals"]
            df.loc[i, 'Normal_referrals'] = M["Free_referrals"] + M["Self_referrals"]
            df.loc[i, 'Mothers with pph_bundle'] = M["pph_bundle"]
            df.loc[i, 'Mothers with iv_iron'] = M["iv_iron"]
            df.loc[i, 'Mothers with MgSO4'] = M["MgSO4"]
            df.loc[i, 'Mothers with antibiotics'] = M["antibiotics"]
            df.loc[i, 'Preterm'] = M["PT"]
            df.loc[i, 'RDS'] = NC["RDS"]
            df.loc[i, 'IVH'] = NC["IVH"]
            df.loc[i, 'neo_sepsis'] = NC["neo_sepsis"]
            df.loc[i, 'NEC'] = NC["NEC"]
            df.loc[i, 'Neonatal Deaths'] = ND["death"]
            df.loc[i, 'hypoxia'] = MC["hypoxia"]
            df.loc[i, 'asphyxia'] = NC["asphyxia"]
            df.loc[i, 'stillbirths'] = NC["stillbirth"]

            df.loc[i, 'PL'] = MC["PL"]
            df.loc[i, 'Anemia'] = MC["anemia"]
            df.loc[i, 'OL'] = MC["OL"]
            df.loc[i, 'mat_sepsis'] = MC['mat_sepsis']
            df.loc[i, 'pph'] = MC["pph"]
            df.loc[i, 'severe_pph'] = MC["pph_severe"]
            df.loc[i, 'eclampsia'] = MC["eclampsia"]
            df.loc[i, 'ruptured_uterus'] = MC["ruptured_uterus"]
            df.loc[i, 'aph'] = MC["aph"]
            df.loc[i, "Comps after transfer"] = MC["comps_death"]
            df.loc[i, 'severe_comps'] = MC["severe_comps"]
            df.loc[i, 'Deaths'] = MD["death"]
            df.loc[i, 'CS'] = M["CS"]
            df.loc[i, 'CS_unnessary'] = M["CS_unnessary"]
            df.loc[i, 'Elective CS'] = M["Elective_CS"]
            df.loc[i, 'Emergency CS'] = M["Emergency_CS"]
            df.loc[i, 'Elective CS risk status'] = M["Elective_CS_risk"]
            n_lowrisk = np.sum(track['LB_Track'][i]) - np.sum(track['HighRisk_Track'][i])
            n_highrisk = np.sum(track['HighRisk_Track'][i])
            df.loc[i, 'Risk status'] = np.array([n_lowrisk, n_highrisk])
            df.loc[i, 'AVD'] = M["AVD"]
            df.loc[i, 'SVD'] = M["SVD"]
            df.loc[i, "ER_trans_actual"] = M["ER_trans_actual"]
            df.loc[i, "ER_trans_pred"] = M["ER_trans_pred"]
            df.loc[i, "Emergency transfers"] = M["ER_trans_actual"] + M["ER_trans_pred"]
            #M_DALYs_old, N_DALYs_old = DALY_calculator(df, param, i)
            M_DALYs_new, N_DALYs_new, M_DALY_ind, N_DALY_ind = DALY_calculator_vectorized(individual_outcomes, param)

            df.loc[i, "M_DALYs"] = M_DALYs_new
            df.loc[i, "N_DALYs"] = N_DALYs_new
            df.loc[i, "DALYs"] = (df.loc[i, "M_DALYs"] + df.loc[i, "N_DALYs"])
            df.loc[i, 'L4/5 LBs'] = round(track['LB_Track'][i, 2] + track['LB_Track'][i, 3])

            #update daly
            individual_df["M_DALY"] = M_DALY_ind
            individual_df["N_DALY"] = N_DALY_ind
            individual_df["DALY"] = individual_df["M_DALY"] + individual_df["N_DALY"]
            df_individual_temp.append(individual_df)

            df.loc[i, 'Facility_capacity_actual'] = round(track['Facility_Capacity_Track'][i, 0])
            df.loc[i, 'Facility_capacity_ideal'] = round(df.loc[i, 'L4/5 LBs'])
            df.loc[i, 'Capacity Ratio'] = df.loc[i, 'L4/5 LBs'] / track['Facility_Capacity_Track'][i, 0]
            labor = labor_calculator(track['LB_Track'][i, :], M["CS"], param, flags)
            df.loc[i, 'Surgical_actual'] = np.array(labor['actual_surgical'][1:])
            df.loc[i, 'Nurse_actual'] = np.array(labor['actual_nurse'][1:])
            df.loc[i, 'Anesthetist_actual'] = np.array(labor['actual_anesthetist'][1:])
            df.loc[i, 'Surgical_needed'] = np.array(labor['surgical'][1:])
            df.loc[i, 'Nurse_needed'] = np.array(labor['nurse'][1:])
            df.loc[i, 'Anesthetist_needed'] = np.array(labor['anesthetist'][1:])
            def staffing_ratio(actual, needed):
                actual = np.asarray(actual, dtype=float)
                needed = np.asarray(needed, dtype=float)
                return np.divide(actual, needed, out=np.zeros_like(actual), where=needed != 0)

            df.loc[i, 'Surgical_ratio'] = staffing_ratio(
                df.loc[i, 'Surgical_actual'], df.loc[i, 'Surgical_needed']
            )
            df.loc[i, 'Nurse_ratio'] = staffing_ratio(
                df.loc[i, 'Nurse_actual'], df.loc[i, 'Nurse_needed']
            )
            df.loc[i, 'Anesthetist_ratio'] = staffing_ratio(
                df.loc[i, 'Anesthetist_actual'], df.loc[i, 'Anesthetist_needed']
            )           # calculate number of sensors and sensor ratio
            sensors = fetal_sensor_calculator(track, param, i, flags, rng)
            dopplers_ratio = np.array([sensors['dopplers_l23_ratio'], sensors['dopplers_l4_ratio'], sensors['dopplers_l5_ratio']])
            CTGs_ratio = np.array([sensors['CTGs_l23_ratio'], sensors['CTGs_l4_ratio'], sensors['CTGs_l5_ratio']])
            df.loc[i, 'Doppler_Actual'] = np.array([sensors['actual_dopplers_l23'], sensors['actual_dopplers_l4'], sensors['actual_dopplers_l5']])
            df.loc[i, 'Doppler_Needed'] = np.array([sensors['dopplers_l23'], sensors['dopplers_l4'], sensors['dopplers_l5']])
            df.loc[i, 'Doppler_Ratio'] = dopplers_ratio
            df.loc[i, 'CTG_Actual'] = np.array([sensors['actual_CTGs_l23'], sensors['actual_CTGs_l4'], sensors['actual_CTGs_l5']])
            df.loc[i, 'CTG_Needed'] = np.array([sensors['CTGs_l23'], sensors['CTGs_l4'], sensors['CTGs_l5']])
            df.loc[i, 'CTG_Ratio'] = CTGs_ratio

            #####only facility-level variables####
            df_facility['Facility_capacity_actual'] = np.array([0, df.loc[i, 'Facility_capacity_actual'], df.loc[i, 'Facility_capacity_actual']])
            df_facility['Facility_capacity_ideal'] = np.array([0, df.loc[i, 'Facility_capacity_ideal'], df.loc[i, 'Facility_capacity_ideal']])
            df_facility['Surgical_actual'] = np.array(labor['actual_surgical'])
            df_facility['Nurse_actual'] = np.array(labor['actual_nurse'])
            df_facility['Anesthetist_actual'] = np.array(labor['surgical'])
            df_facility['Surgical_needed'] = np.array(labor['surgical'])
            df_facility['Nurse_needed'] = np.array(labor['nurse'])
            df_facility['Anesthetist_needed'] = np.array(labor['anesthetist'])
            df_facility['Doppler_Actual'] = df.loc[i, 'Doppler_Actual']
            df_facility['Doppler_Needed'] = df.loc[i, 'Doppler_Needed']
            df_facility['CTG_Actual'] = df.loc[i, 'CTG_Actual']
            df_facility['CTG_Needed'] = df.loc[i, 'CTG_Needed']
            df_facility_temp.append(df_facility)

    df_individual = pd.concat(df_individual_temp, ignore_index=True)
    df_facility_all = pd.concat(df_facility_temp, ignore_index=True)

    return df, df_individual, df_facility_all

def update_capacity(track, param, i, flags, int_period):
    #Updating facility capacity
    if flags['flag_capacity']:
        if i < int_period:   #assume linearly increasing capacity
            Fac_Capacity = param["Capacity"] * (1 + param["HSS"]["capacity_added"] / (int_period-1) * i)
            CS_Capacity = param["p_cs_capacity"][3] * (1 + param["HSS"]["capacity_added"] / (int_period-1) * i)
        else:
            Fac_Capacity = param["Capacity"] * (1 + param["HSS"]["capacity_added"])
            CS_Capacity = param["p_cs_capacity"][3] * (1 + param["HSS"]["capacity_added"])
    else:
        Fac_Capacity = param["Capacity"]
        CS_Capacity = param["p_cs_capacity"][3]

    track['Facility_Capacity_Track'][i, 0] = Fac_Capacity
    track['CS_Capacity_Track'][i, 0] = CS_Capacity

    #Updating referral capacity
    if flags['flag_refer_voucher']:
        track['Referral_Capacity_Track'][i, 0] = param["HSS"]["P_refer"]
    else:
        track['Referral_Capacity_Track'][i, 0] = track['Referral_Capacity_Track'][i - 1, 0]

    return track






 
