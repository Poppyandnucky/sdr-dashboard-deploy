"""Cost-effectiveness calculations for the Kakamega SDR dashboard."""
import math

import numpy as np
import pandas as pd


def get_cost_parameters():
    """Return Kakamega SDR costing assumptions aligned with Tingting's Appendix B logic.

    Values are stored in Kenyan shillings unless otherwise noted. Calculation
    functions convert unit costs to USD internally using ``USD_to_Ksh``.
    """
    return {
        "USD_to_Ksh": 129.5,
        "cost_discount_rate": 0.03,
        "time_horizon_years": 10,
        "default_implementation_years": 4,
        "default_maintenance_years": 6,
        "planning_cycle_years": 2,
        "dispatches_per_vehicle": 57,
        "useful_life_dict": {
            "Doppler": 7,
            "CTG": 7,
            "Infra": 20,
            "Equip": 5,
            "Taxi_Setup": 5,
            "Ambulance_Setup": 5,
        },
        "scenario_tiers": {
            "conservative": ["SDR-40", "SDR-45", "SDR-50", "Conservative"],
            "moderate": ["SDR-55", "SDR-60", "SDR-65", "SDR-70", "Moderate"],
            "aggressive": ["SDR-75", "SDR-80", "SDR-85", "SDR-90", "Aggressive"],
        },
        "capacity_expansion_by_scenario": {
            "SDR-40": 0.05,
            "SDR-45": 0.12,
            "SDR-50": 0.21,
            "SDR-55": 0.30,
            "SDR-60": 0.38,
            "SDR-65": 0.46,
            "SDR-70": 0.54,
            "SDR-75": 0.61,
            "SDR-80": 0.68,
            "SDR-85": 0.76,
            "SDR-90": 0.82,
        },
        "cost_dict": {
            "Infra": 2257330,
            "Equip": 11359478,
            "POCUS": 5000 * 129.5,
            "Doppler": 4000,
            "CTG": 657900,
            "Taxi_Setup": 19000,
            "Ambulance_Setup": 130500,
            "PM_imple": 40207381,
            "PM_maintain_conservative": 4556971,
            "PM_maintain_moderate": 5693559,
            "PM_maintain_aggressive": 6833850,
            "Overhead_conservative": 5584689,
            "Overhead_moderate": 8461217,
            "Overhead_aggressive": 14386851,
            "Taxi_Monthly": 250,
            "Taxi_Dispatch": 5500,
            "Ambulance_Monthly": 2250,
            "Ambulance_Dispatch": 8500,
            "ANC": 312,
            "Fac_Delivery": 6148,
            "CS_Delivery": 29804,
            "surgical_staff": 119495,
            "nurse_staff": 43950,
            "anesthetist": 61170,
        },
    }


def annuity_factor(discount_rate, useful_life):
    """Return the annualization factor for a capital item."""
    if useful_life <= 0:
        raise ValueError("useful_life must be positive.")
    if discount_rate == 0:
        return 1 / useful_life
    return (
        discount_rate * (1 + discount_rate) ** useful_life
    ) / ((1 + discount_rate) ** useful_life - 1)


def add_discounted_cost(df, discount_rate):
    """Add present-value yearly cost using Year 1 as the undiscounted year."""
    df = df.copy()
    df["cost_discounted_yearly"] = (
        df["cost_yearly"] / ((1 + discount_rate) ** (df["year"] - 1))
    )
    return df


def summarize_by_t_ci(df, value_col, group_cols, confidence=0.95):
    """Summarize run-level values using a t-distribution confidence interval."""
    try:
        from scipy.stats import t
    except ImportError as exc:
        raise ImportError("scipy is required for t confidence intervals.") from exc

    summary = (
        df.groupby(group_cols, as_index=False)[value_col]
        .agg(mean="mean", sd="std", n="count")
    )
    summary["sd"] = summary["sd"].fillna(0)
    summary["t_crit"] = summary["n"].apply(
        lambda n: t.ppf((1 + confidence) / 2, n - 1) if n > 1 else np.nan
    )
    se = summary["sd"] / np.sqrt(summary["n"])
    summary["lower_CI"] = summary["mean"] - summary["t_crit"] * se
    summary["upper_CI"] = summary["mean"] + summary["t_crit"] * se
    one_run = summary["n"] <= 1
    summary.loc[one_run, "lower_CI"] = summary.loc[one_run, "mean"]
    summary.loc[one_run, "upper_CI"] = summary.loc[one_run, "mean"]
    return summary.drop(columns=["sd", "n", "t_crit"])


def _as_array(value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.array([float(arr)])
    return arr


def _sum_array(value):
    return float(np.nansum(_as_array(value)))
def _unit_costs_usd(cost_parameters):
    exchange_rate = cost_parameters["USD_to_Ksh"]
    return {
        key: value / exchange_rate
        for key, value in cost_parameters["cost_dict"].items()
    }


def _with_run_month_year(df):
    df = df.copy()
    if "Run" not in df.columns:
        df["Run"] = 1
    df["Month"] = df["Month"].astype(int)
    df["year"] = (df["Month"] // 12) + 1
    return df


def _prepare_monthly_counts(df, scenario):
    df = _with_run_month_year(df)
    out = pd.DataFrame(
        {
            "Run": df["Run"].astype(int),
            "Month": df["Month"],
            "year": df["year"],
            "Scenario": scenario,
            "n_ANC": df["ANC"].apply(_sum_array),
            "n_fac_delivery": df["Fac non-CS"].apply(_sum_array),
            "n_CS_delivery": df["CS"].apply(_sum_array),
            "n_referral": df["Free_referrals"].apply(_sum_array),
            "n_transfer": df["Emergency transfers"].apply(_sum_array),
            "Livebirths": df["Live Births Final"].apply(_sum_array),
            "M_DALYs": df["M_DALYs"].apply(_sum_array),
        }
    )
    out["M_YLLs"] = df["M_YLLs"].apply(_sum_array) if "M_YLLs" in df.columns else 0.0
    out["M_YLDs"] = df["M_YLDs"].apply(_sum_array) if "M_YLDs" in df.columns else 0.0
    return out


def _expand_array_column(df, column, level_names):
    df = _with_run_month_year(df)
    records = []
    for _, row in df.iterrows():
        values = _as_array(row[column])
        for i, level in enumerate(level_names):
            records.append(
                {
                    "Run": int(row["Run"]),
                    "Month": int(row["Month"]),
                    "year": int(row["year"]),
                    "Level": level,
                    column: float(values[i]) if i < len(values) else 0.0,
                }
            )
    return pd.DataFrame(records)


def _scenario_tier(scenario_name, cost_parameters, default_tier="moderate"):
    scenario_name = str(scenario_name)
    for tier, labels in cost_parameters["scenario_tiers"].items():
        if scenario_name in labels:
            return tier
    return default_tier


def _yearly_recurrent_from_monthly(merged, indicator, unit_cost, cost_type, discount_rate):
    monthly = merged.copy()
    monthly["resource_diff"] = (
        monthly[f"{indicator}_intervention"] - monthly[f"{indicator}_baseline"]
    ).clip(lower=0)
    monthly["cost_monthly"] = monthly["resource_diff"] * unit_cost
    yearly = (
        monthly.groupby(["Run", "year"], as_index=False)
        .agg(cost_yearly=("cost_monthly", "sum"), resource_count=("resource_diff", "sum"))
    )
    yearly["cost_type"] = cost_type
    return add_discounted_cost(yearly, discount_rate)


def _merge_monthly_counts(baseline_df, intervention_df):
    baseline = _prepare_monthly_counts(baseline_df, "Baseline")
    intervention = _prepare_monthly_counts(intervention_df, "Intervention")
    return intervention.merge(
        baseline,
        on=["Run", "Month", "year"],
        suffixes=("_intervention", "_baseline"),
    )


def calculate_service_delivery_costs(baseline_df, intervention_df, unit_costs, discount_rate):
    """Calculate recurrent ANC, facility delivery, and C-section costs."""
    merged = _merge_monthly_counts(baseline_df, intervention_df)
    return pd.concat(
        [
            _yearly_recurrent_from_monthly(
                merged, "n_ANC", unit_costs["ANC"], "ANC", discount_rate
            ),
            _yearly_recurrent_from_monthly(
                merged,
                "n_fac_delivery",
                unit_costs["Fac_Delivery"],
                "Facility Delivery",
                discount_rate,
            ),
            _yearly_recurrent_from_monthly(
                merged,
                "n_CS_delivery",
                unit_costs["CS_Delivery"],
                "CS Delivery",
                discount_rate,
            ),
        ],
        ignore_index=True,
    )


def calculate_labor_costs(
    baseline_df,
    intervention_df,
    unit_costs,
    discount_rate,
    planning_cycle_years=2,
    quantile_threshold=0.9,
):
    """Calculate staged staffing costs using two-year planning blocks by default."""
    labor_specs = [
        ("Surgical_actual", "surgical_staff", "Labor (Surgical)"),
        ("Nurse_actual", "nurse_staff", "Labor (Nurse)"),
        ("Anesthetist_actual", "anesthetist", "Labor (Anesthetist)"),
    ]
    outputs = []
    for column, unit_key, cost_type in labor_specs:
        baseline = _expand_array_column(baseline_df, column, ["L4", "L5"])
        intervention = _expand_array_column(intervention_df, column, ["L4", "L5"])
        merged = intervention.merge(
            baseline,
            on=["Run", "Month", "year", "Level"],
            suffixes=("_intervention", "_baseline"),
        )
        merged["staff_diff"] = (
            merged[f"{column}_intervention"] - merged[f"{column}_baseline"]
        ).clip(lower=0)
        merged["planning_block"] = ((merged["year"] - 1) // planning_cycle_years) + 1
        block = (
            merged.groupby(["Run", "Level", "planning_block"], as_index=False)
            .agg(n_staff_hired_level=("staff_diff", lambda x: np.quantile(x, quantile_threshold)))
        )
        block = (
            block.groupby(["Run", "planning_block"], as_index=False)
            .agg(n_staff_hired=("n_staff_hired_level", "sum"))
        )
        block["year_start"] = (block["planning_block"] - 1) * planning_cycle_years + 1
        rows = []
        max_year = int(merged["year"].max())
        for _, row in block.iterrows():
            for year in range(int(row["year_start"]), min(int(row["year_start"] + planning_cycle_years), max_year + 1)):
                rows.append(
                    {
                        "Run": int(row["Run"]),
                        "year": year,
                        "n_staff_hired": float(row["n_staff_hired"]),
                        "cost_yearly": float(row["n_staff_hired"]) * unit_costs[unit_key] * 12,
                        "cost_type": cost_type,
                    }
                )
        yearly = pd.DataFrame(rows)
        outputs.append(add_discounted_cost(yearly, discount_rate))
    return pd.concat(outputs, ignore_index=True)


def calculate_referral_recurrent_costs(
    baseline_df,
    intervention_df,
    unit_costs,
    cost_parameters,
    discount_rate,
):
    """Calculate recurrent referral and transfer dispatch/operation costs."""
    merged = _merge_monthly_counts(baseline_df, intervention_df)
    dispatches_per_vehicle = cost_parameters["dispatches_per_vehicle"]
    specs = [
        (
            "n_referral",
            "Taxi_Dispatch",
            "Taxi_Monthly",
            "n_taxi_used",
            "Referral Recurrent",
        ),
        (
            "n_transfer",
            "Ambulance_Dispatch",
            "Ambulance_Monthly",
            "n_ambulance_used",
            "Transfer Recurrent",
        ),
    ]
    outputs = []
    vehicle_use = []
    for indicator, dispatch_key, monthly_key, vehicle_col, cost_type in specs:
        monthly = merged[["Run", "Month", "year"]].copy()
        monthly["n_additional_dispatches"] = (
            merged[f"{indicator}_intervention"] - merged[f"{indicator}_baseline"]
        ).clip(lower=0)
        monthly[vehicle_col] = np.ceil(monthly["n_additional_dispatches"] / dispatches_per_vehicle)
        monthly["cost_monthly"] = (
            monthly["n_additional_dispatches"] * unit_costs[dispatch_key]
            + monthly[vehicle_col] * unit_costs[monthly_key]
        )
        yearly = (
            monthly.groupby(["Run", "year"], as_index=False)
            .agg(
                cost_yearly=("cost_monthly", "sum"),
                n_additional_dispatches=("n_additional_dispatches", "sum"),
                **{vehicle_col: (vehicle_col, "max")},
            )
        )
        yearly["cost_type"] = cost_type
        outputs.append(add_discounted_cost(yearly, discount_rate))
        vehicle_use.append(monthly[["Run", "Month", "year", vehicle_col]])
    return pd.concat(outputs, ignore_index=True), vehicle_use


def calculate_management_costs(
    baseline_df,
    intervention_df,
    unit_costs,
    cost_parameters,
    discount_rate,
    implementation_years,
    scenario_name="Intervention",
    scenario_tier=None,
):
    """Calculate program management and overhead costs.

    Appendix B describes implementation program management as a fixed total
    implementation-phase cost, so this implementation spreads PM_imple evenly
    across implementation months.
    """
    monthly = _prepare_monthly_counts(intervention_df, scenario_name)[["Run", "Month", "year"]]
    implementation_months = implementation_years * 12
    tier = scenario_tier or _scenario_tier(scenario_name, cost_parameters)
    pm_maintain_key = f"PM_maintain_{tier}"
    overhead_key = f"Overhead_{tier}"
    if implementation_months <= 0:
        raise ValueError("implementation_years must be positive for PM costing.")

    pm = monthly.copy()
    pm["cost_monthly"] = np.where(
        pm["Month"] < implementation_months,
        unit_costs["PM_imple"] / implementation_months,
        unit_costs[pm_maintain_key] / 12,
    )
    pm_yearly = (
        pm.groupby(["Run", "year"], as_index=False)
        .agg(cost_yearly=("cost_monthly", "sum"))
    )
    pm_yearly["cost_type"] = "Program Management"

    overhead = monthly.copy()
    overhead["cost_monthly"] = unit_costs[overhead_key] / 12
    overhead_yearly = (
        overhead.groupby(["Run", "year"], as_index=False)
        .agg(cost_yearly=("cost_monthly", "sum"))
    )
    overhead_yearly["cost_type"] = "Overhead"
    return pd.concat(
        [
            add_discounted_cost(pm_yearly, discount_rate),
            add_discounted_cost(overhead_yearly, discount_rate),
        ],
        ignore_index=True,
    )


def _annualize_purchase_blocks(
    purchases,
    unit_cost,
    useful_life,
    discount_rate,
    max_year,
    cost_type,
    quantity_col="purchased",
):
    factor = annuity_factor(discount_rate, useful_life)
    rows = []
    for _, row in purchases.iterrows():
        purchase_year = int(row["purchase_year"])
        annualized_cost = float(row[quantity_col]) * unit_cost * factor
        for year in range(purchase_year, max_year + 1):
            rows.append(
                {
                    "Run": int(row["Run"]),
                    "year": year,
                    "cost_yearly": annualized_cost,
                    "cost_type": cost_type,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["Run", "year", "cost_type", "cost_yearly", "cost_discounted_yearly"])
    yearly = pd.DataFrame(rows)
    yearly = (
        yearly.groupby(["Run", "year", "cost_type"], as_index=False)
        .agg(cost_yearly=("cost_yearly", "sum"))
    )
    return add_discounted_cost(yearly, discount_rate)


def _split_target_stock_purchases(target_stock, implementation_years):
    """Split final implementation-phase stock into two capital purchase phases."""
    midpoint_year = 1 + math.ceil(implementation_years / 2)
    purchases = []
    for _, row in target_stock.iterrows():
        required_stock = int(math.ceil(float(row["required_stock"])))
        phase1 = math.ceil(required_stock * 0.5)
        phase2 = required_stock - phase1
        purchases.append({"Run": row["Run"], "purchase_year": 1, "purchased": phase1})
        purchases.append(
            {"Run": row["Run"], "purchase_year": midpoint_year, "purchased": phase2}
        )
    return pd.DataFrame(purchases)


def calculate_infrastructure_costs(
    baseline_df,
    intervention_df,
    unit_costs,
    cost_parameters,
    discount_rate,
    implementation_years,
    scenario_name="Intervention",
):
    """Calculate staged infrastructure capital costs from added L4/5 capacity.

    For named scenario-analysis outputs, Appendix B defines the required
    infrastructure from the scenario-specific capacity expansion share. For
    generic dashboard runs, fall back to the observed intervention-baseline
    facility-capacity difference.
    """
    baseline = _with_run_month_year(baseline_df)
    intervention = _with_run_month_year(intervention_df)
    beds_per_month = 83 / 12
    scenario_expansion = cost_parameters.get("capacity_expansion_by_scenario", {}).get(
        scenario_name
    )
    if scenario_expansion is not None:
        baseline_capacity = 34777 / 12
        n_beds_needed = math.ceil(
            scenario_expansion * baseline_capacity / beds_per_month
        )
        beds = pd.DataFrame(
            {
                "Run": intervention["Run"].drop_duplicates().astype(int),
                "n_beds_needed": n_beds_needed,
            }
        )
        max_year = int(intervention["year"].max())
    else:
        merged = intervention[["Run", "Month", "year", "Facility_capacity_actual"]].merge(
            baseline[["Run", "Month", "year", "Facility_capacity_actual"]],
            on=["Run", "Month", "year"],
            suffixes=("_intervention", "_baseline"),
        )
        merged["added_capacity"] = (
            merged["Facility_capacity_actual_intervention"].astype(float)
            - merged["Facility_capacity_actual_baseline"].astype(float)
        ).clip(lower=0)
        beds = (
            merged.groupby("Run", as_index=False)
            .agg(n_beds_needed=("added_capacity", lambda x: math.ceil(float(np.max(x)) / beds_per_month)))
        )
        max_year = int(merged["year"].max())
    midpoint_year = 1 + math.ceil(implementation_years / 2)
    purchases = []
    for _, row in beds.iterrows():
        phase1 = math.ceil(row["n_beds_needed"] * 0.5)
        phase2 = row["n_beds_needed"] - phase1
        purchases.append({"Run": row["Run"], "purchase_year": 1, "purchased": phase1})
        purchases.append({"Run": row["Run"], "purchase_year": midpoint_year, "purchased": phase2})
    return _annualize_purchase_blocks(
        pd.DataFrame(purchases),
        unit_costs["Infra"],
        cost_parameters["useful_life_dict"]["Infra"],
        discount_rate,
        max_year,
        "Infrastructure",
    )


def calculate_general_equipment_costs(intervention_df, unit_costs, cost_parameters, discount_rate):
    """Calculate fixed general equipment capital costs for SDR-ready L4/5 facilities."""
    runs = _with_run_month_year(intervention_df)["Run"].drop_duplicates()
    purchases = pd.DataFrame(
        {"Run": runs.astype(int), "purchase_year": 1, "purchased": 1}
    )
    max_year = int(_with_run_month_year(intervention_df)["year"].max())
    return _annualize_purchase_blocks(
        purchases,
        unit_costs["Equip"],
        cost_parameters["useful_life_dict"]["Equip"],
        discount_rate,
        max_year,
        "Equipment (General)",
    )


def calculate_sensor_costs(
    baseline_df,
    intervention_df,
    unit_costs,
    cost_parameters,
    discount_rate,
    planning_cycle_years=2,
    implementation_years=None,
):
    """Calculate Doppler and CTG capital costs fixed by implementation-phase need."""
    implementation_years = implementation_years or cost_parameters["default_implementation_years"]
    specs = [
        ("Doppler_Actual", "Doppler", "Equipment (Doppler)"),
        ("CTG_Actual", "CTG", "Equipment (CTG)"),
    ]
    outputs = []
    max_year = int(_with_run_month_year(intervention_df)["year"].max())
    for column, unit_key, cost_type in specs:
        baseline = _expand_array_column(baseline_df, column, ["L23", "L4", "L5"])
        intervention = _expand_array_column(intervention_df, column, ["L23", "L4", "L5"])
        merged = intervention.merge(
            baseline,
            on=["Run", "Month", "year", "Level"],
            suffixes=("_intervention", "_baseline"),
        )
        merged["stock_diff"] = (
            merged[f"{column}_intervention"] - merged[f"{column}_baseline"]
        ).clip(lower=0)
        merged = merged[merged["year"] <= implementation_years]
        block = (
            merged.groupby(["Run", "Level"], as_index=False)
            .agg(required_stock_level=("stock_diff", "max"))
        )
        target_stock = (
            block.groupby("Run", as_index=False)
            .agg(required_stock=("required_stock_level", "sum"))
            .sort_values("Run")
        )
        purchases = _split_target_stock_purchases(target_stock, implementation_years)
        outputs.append(
            _annualize_purchase_blocks(
                purchases,
                unit_costs[unit_key],
                cost_parameters["useful_life_dict"][unit_key],
                discount_rate,
                max_year,
                cost_type,
            )
        )
    return pd.concat(outputs, ignore_index=True)


def calculate_vehicle_capital_costs(
    vehicle_use,
    unit_costs,
    cost_parameters,
    discount_rate,
    planning_cycle_years=2,
    implementation_years=None,
):
    """Calculate vehicle capital costs fixed by implementation-phase need."""
    implementation_years = implementation_years or cost_parameters["default_implementation_years"]
    specs = [
        (vehicle_use[0], "n_taxi_used", "Taxi_Setup", "Referral (Capital)"),
        (vehicle_use[1], "n_ambulance_used", "Ambulance_Setup", "Transfer (Capital)"),
    ]
    outputs = []
    for df, vehicle_col, unit_key, cost_type in specs:
        monthly = df.copy()
        target_stock = (
            monthly[monthly["year"] <= implementation_years]
            .groupby("Run", as_index=False)
            .agg(required_stock=(vehicle_col, "max"))
            .sort_values("Run")
        )
        purchases = _split_target_stock_purchases(target_stock, implementation_years)
        outputs.append(
            _annualize_purchase_blocks(
                purchases,
                unit_costs[unit_key],
                cost_parameters["useful_life_dict"][unit_key],
                discount_rate,
                int(monthly["year"].max()),
                cost_type,
            )
        )
    return pd.concat(outputs, ignore_index=True)


def calculate_dalys_averted_by_year(baseline_df, intervention_df):
    """Calculate annual maternal YLL, YLD, and DALYs averted by run."""
    merged = _merge_monthly_counts(baseline_df, intervention_df)
    merged["yll_averted"] = (
        merged.get("M_YLLs_baseline", 0.0) - merged.get("M_YLLs_intervention", 0.0)
    )
    merged["yld_averted"] = (
        merged.get("M_YLDs_baseline", 0.0) - merged.get("M_YLDs_intervention", 0.0)
    )
    merged["dalys_averted"] = (
        merged["M_DALYs_baseline"] - merged["M_DALYs_intervention"]
    )
    yearly = (
        merged.groupby(["Run", "year"], as_index=False)
        .agg(
            yll_averted=("yll_averted", "sum"),
            yld_averted=("yld_averted", "sum"),
            dalys_averted=("dalys_averted", "sum"),
        )
        .sort_values(["Run", "year"])
    )
    yearly["cumulative_yll_averted"] = yearly.groupby("Run")["yll_averted"].cumsum()
    yearly["cumulative_yld_averted"] = yearly.groupby("Run")["yld_averted"].cumsum()
    yearly["cumulative_dalys_averted"] = yearly.groupby("Run")["dalys_averted"].cumsum()
    return yearly



def calculate_sdr_costs(
    baseline_df,
    intervention_df,
    cost_parameters,
    implementation_years=None,
    maintenance_years=None,
    scenario_name="Intervention",
    scenario_tier=None,
    include_general_equipment=True,
):
    """Calculate Appendix B-style SDR costs from baseline/intervention outputs.

    Returns a dictionary with component-level yearly costs, total yearly costs,
    DALYs averted, and cumulative ICER inputs. Costs are returned in USD.
    """
    unit_costs = _unit_costs_usd(cost_parameters)
    discount_rate = cost_parameters["cost_discount_rate"]
    implementation_years = implementation_years or cost_parameters["default_implementation_years"]
    maintenance_years = maintenance_years or cost_parameters["default_maintenance_years"]
    planning_cycle_years = cost_parameters["planning_cycle_years"]

    recurrent_service = calculate_service_delivery_costs(
        baseline_df, intervention_df, unit_costs, discount_rate
    )
    labor = calculate_labor_costs(
        baseline_df,
        intervention_df,
        unit_costs,
        discount_rate,
        planning_cycle_years=planning_cycle_years,
    )
    referral_recurrent, vehicle_use = calculate_referral_recurrent_costs(
        baseline_df, intervention_df, unit_costs, cost_parameters, discount_rate
    )
    management = calculate_management_costs(
        baseline_df,
        intervention_df,
        unit_costs,
        cost_parameters,
        discount_rate,
        implementation_years,
        scenario_name=scenario_name,
        scenario_tier=scenario_tier,
    )
    infrastructure = calculate_infrastructure_costs(
        baseline_df,
        intervention_df,
        unit_costs,
        cost_parameters,
        discount_rate,
        implementation_years,
        scenario_name=scenario_name,
    )
    capital_components = [infrastructure]
    if include_general_equipment:
        capital_components.append(
            calculate_general_equipment_costs(
                intervention_df, unit_costs, cost_parameters, discount_rate
            )
        )
    capital_components.append(
        calculate_sensor_costs(
            baseline_df,
            intervention_df,
            unit_costs,
            cost_parameters,
            discount_rate,
            planning_cycle_years=planning_cycle_years,
            implementation_years=implementation_years,
        )
    )
    capital_components.append(
        calculate_vehicle_capital_costs(
            vehicle_use,
            unit_costs,
            cost_parameters,
            discount_rate,
            planning_cycle_years=planning_cycle_years,
            implementation_years=implementation_years,
        )
    )

    component_yearly = pd.concat(
        [
            recurrent_service,
            labor,
            referral_recurrent,
            management,
            *capital_components,
        ],
        ignore_index=True,
    )
    total_years = implementation_years + maintenance_years
    component_yearly = component_yearly[component_yearly["year"] <= total_years].copy()
    component_yearly["Scenario"] = scenario_name

    total_yearly = (
        component_yearly.groupby(["Run", "Scenario", "year"], as_index=False)
        .agg(
            cost_yearly=("cost_yearly", "sum"),
            cost_discounted_yearly=("cost_discounted_yearly", "sum"),
        )
        .sort_values(["Run", "year"])
    )
    total_yearly["cumulative_discounted_cost"] = total_yearly.groupby("Run")[
        "cost_discounted_yearly"
    ].cumsum()

    dalys = calculate_dalys_averted_by_year(baseline_df, intervention_df)
    icer = total_yearly.merge(dalys, on=["Run", "year"], how="left")
    icer["cost_per_daly_averted"] = (
        icer["cumulative_discounted_cost"] / icer["cumulative_dalys_averted"]
    )
    icer.loc[icer["cumulative_dalys_averted"] <= 0, "cost_per_daly_averted"] = np.nan
    dalys["Scenario"] = scenario_name

    return {
        "component_yearly": component_yearly,
        "total_yearly": total_yearly,
        "dalys_averted_yearly": dalys,
        "icer_yearly": icer,
    }


def infer_sdr_cost_tier(hss_settings):
    """Map dashboard HSS settings to the cost tier used for PM/overhead costs."""
    target_l45 = hss_settings.get("P_L45", 0)
    if target_l45 < 0.55:
        return "conservative"
    if target_l45 < 0.75:
        return "moderate"
    return "aggressive"


def clean_cost_table(df):
    """Combine sensor cost lines for dashboard display."""
    out = df.copy()
    out["cost_type"] = out["cost_type"].replace(
        {
            "Equipment (Doppler)": "Equipment (Sensors)",
            "Equipment (CTG)": "Equipment (Sensors)",
        }
    )
    return (
        out.groupby(["Run", "Scenario", "year", "cost_type"], as_index=False)
        .agg(
            cost_yearly=("cost_yearly", "sum"),
            cost_discounted_yearly=("cost_discounted_yearly", "sum"),
        )
        .sort_values(["Run", "year", "cost_type"])
    )
