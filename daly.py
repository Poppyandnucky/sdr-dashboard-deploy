"""DALY calculations for SDR dashboard model outputs."""

import numpy as np


DEFAULT_DW_DURATION = {
    "anemia": 0.08,
    "hemorrhage": 0.25,
    "maternal sepsis": 0.04,
    "eclampsia": 0.17,
    "obstructed labor": 0.04,
}


def maternal_DALY_calculator_vectorized(individual_outcomes, param):
    """Calculate maternal DALYs, YLLs, and YLDs by final delivery location."""
    dw = param["DW"]
    duration = param.get("DW_duration", DEFAULT_DW_DURATION)

    maternal_dalys = np.zeros(4, dtype=float)
    maternal_ylls = np.zeros(4, dtype=float)
    maternal_ylds = np.zeros(4, dtype=float)

    i_loc = individual_outcomes["i_loc_new_v2"]
    i_anemia = individual_outcomes["i_anemia_new"]
    i_pph_severe = individual_outcomes["i_pph_severe_new"]
    i_pph = individual_outcomes["i_pph_new"]
    i_pph_notsevere = ((i_pph == 1) & (i_pph_severe == 0)).astype(int)
    i_mat_sepsis = individual_outcomes["i_mat_sepsis_new"]
    i_eclampsia = individual_outcomes["i_eclampsia_new"]
    i_obstructed_labor = individual_outcomes["i_OL_final"]
    i_mat_death = individual_outcomes["i_mat_death"]

    num_mothers = i_anemia.shape[0]
    maternal_yll = np.zeros(num_mothers, dtype=float)
    maternal_yld = np.zeros(num_mothers, dtype=float)

    maternal_complication_mask = (
        (i_anemia == 1)
        | (i_pph_severe == 1)
        | (i_pph_notsevere == 1)
        | (i_mat_sepsis == 1)
        | (i_eclampsia == 1)
        | (i_obstructed_labor == 1)
    ) & (i_mat_death == 0)
    maternal_death_mask = i_mat_death == 1

    complication_yld = (
        i_anemia * dw["anemia"] * duration["anemia"]
        + i_pph_notsevere * dw["low pph"] * duration["hemorrhage"]
        + i_pph_severe * dw["high pph"] * duration["hemorrhage"]
        + i_mat_sepsis * dw["maternal sepsis"] * duration["maternal sepsis"]
        + i_eclampsia * dw["eclampsia"] * duration["eclampsia"]
        + i_obstructed_labor * dw["obstructed labor"] * duration["obstructed labor"]
    )
    maternal_yld[maternal_complication_mask] = complication_yld[maternal_complication_mask]

    remaining_life_expectancy = param["Mother_life_expectancy"] - param["Childbearing_age"]
    death_yll = i_mat_death * dw["maternal death"] * remaining_life_expectancy
    maternal_yll[maternal_death_mask] = death_yll[maternal_death_mask]

    maternal_daly = maternal_yll + maternal_yld

    np.add.at(maternal_dalys, i_loc, maternal_daly)
    np.add.at(maternal_ylls, i_loc, maternal_yll)
    np.add.at(maternal_ylds, i_loc, maternal_yld)

    return maternal_dalys, maternal_ylls, maternal_ylds, maternal_daly, maternal_yll, maternal_yld


def neonatal_DALY_calculator_vectorized(individual_outcomes, param):
    """Calculate neonatal DALYs by final delivery location using current deploy logic."""
    dw = param["DW"]

    neonatal_dalys = np.zeros(4, dtype=float)

    i_loc = individual_outcomes["i_loc_new_v2"]
    i_rds = individual_outcomes["i_RDS"]
    i_ivh = individual_outcomes["i_IVH"]
    i_nec = individual_outcomes["i_NEC"]
    i_neo_sepsis = individual_outcomes["i_neo_sepsis"]
    i_asphyxia = individual_outcomes["i_asphyxia"]
    i_stillbirth = individual_outcomes["i_stillbirth"]
    i_neo_death = individual_outcomes["i_neo_death"]

    num_mothers = i_rds.shape[0]
    neonatal_daly = np.zeros(num_mothers, dtype=float)
    neonatal_complication_mask = (
        (i_rds == 1)
        | (i_ivh == 1)
        | (i_nec == 1)
        | (i_neo_sepsis == 1)
        | (i_asphyxia == 1)
    ) & (i_stillbirth == 0) & (i_neo_death == 0)
    neonatal_death_mask = (i_neo_death == 1) & (i_stillbirth == 0)

    complication_daly = (
        i_rds * dw["preterm comp"]
        + i_ivh * dw["preterm comp"]
        + i_nec * dw["preterm comp"]
        + i_neo_sepsis * dw["neonatal sepsis"]
        + i_asphyxia * dw["asphyxia"]
    ) * param["Neonate_life_expectancy"]
    death_daly = i_neo_death * dw["neonatal death"] * param["Neonate_life_expectancy"]

    neonatal_daly[neonatal_complication_mask] = complication_daly[neonatal_complication_mask]
    neonatal_daly[neonatal_death_mask] = death_daly[neonatal_death_mask]

    np.add.at(neonatal_dalys, i_loc, neonatal_daly)
    return neonatal_dalys, neonatal_daly


def DALY_calculator_vectorized(individual_outcomes, param):
    """Calculate corrected maternal DALYs plus existing neonatal DALYs."""
    (
        maternal_dalys,
        maternal_ylls,
        maternal_ylds,
        maternal_daly,
        maternal_yll,
        maternal_yld,
    ) = maternal_DALY_calculator_vectorized(individual_outcomes, param)
    neonatal_dalys, neonatal_daly = neonatal_DALY_calculator_vectorized(individual_outcomes, param)

    return (
        maternal_dalys,
        maternal_ylls,
        maternal_ylds,
        neonatal_dalys,
        maternal_daly,
        maternal_yll,
        maternal_yld,
        neonatal_daly,
    )
