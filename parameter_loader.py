"""
Parameter loader for SDR county-specific simulations.

Usage
-----
from parameter_loader import get_parameters, get_slider_params, calculate_derived_parameters

param = get_parameters("/path/to/SDR Parameters.xlsx", county="kisii", seed=123)
slider_params = get_slider_params("/path/to/SDR Parameters.xlsx", county="kisii")
param = calculate_derived_parameters(param)

Notes
-----
- County-specific sheets override shared/default sheets.
- Missing county values are ignored by default so you can fall back to defaults or Kakamega.
- Uncertain parameters are sampled through global_func.sample_from_ci.
"""

from __future__ import annotations

import ast
import copy
import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from global_func import (
    sample_from_ci,
    odds_prob,
    comps_riskstatus_vs_lowrisk,
    comp2_comp1_anemia,
    P_RDS,
    GA_assign_kenya,
    GA_by_ANC,
)

FACILITY_ORDER = ["home", "L2/3", "L4", "L5"]
FACILITY_SUFFIX = {"L2/3": "L2/3", "L4": "L4", "L5": "L5"}

# Excel names that are stored as multiple rows but expected as numpy arrays in the model.
SAMPLED_ARRAYS = {
    "E_Preterm_LMP": ["E_Preterm_LMP_preterm", "E_Preterm_LMP_atterm"],
    "E_Postterm_LMP": ["E_Postterm_LMP_preterm", "E_Postterm_LMP_atterm"],
    "p_PL_GA": ["p_PL_GA_37", "p_PL_GA_38", "p_PL_GA_39", "p_PL_GA_40", "p_PL_GA_41", "p_PL_GA_42"],
    "p_OL": ["p_OL_notprolonged", "p_OL_prolonged"],
}

# ---------------------------------------------------------------------------
# Module-level workbook path and county default.
# Override WORKBOOK_PATH at runtime by setting the SDR_PARAMS_PATH env var,
# e.g. for server deployment:  export SDR_PARAMS_PATH=/app/SDR_Parameters.xlsx
# ---------------------------------------------------------------------------
WORKBOOK_PATH: Path = Path(os.environ.get(
    "SDR_PARAMS_PATH",
    "/Users/poppy/Library/CloudStorage/OneDrive-SharedLibraries-JohnsHopkins/"
    "Meibin Chen - MOMISH interventions/SDR Parameters.xlsx",
))
DEFAULT_COUNTY: str = "kakamega"

# Disability-weight labels expected by existing DALY code.
DW_NAME_MAP = {
    "low_pph": "low pph",
    "high_pph": "high pph",
    "maternal_sepsis": "maternal sepsis",
    "obstructed_labor": "obstructed labor",
    "maternal_death": "maternal death",
    "neonatal_death": "neonatal death",
    "preterm_comp": "preterm comp",
    "neonatal_sepsis": "neonatal sepsis",
}

@dataclass(frozen=True)
class ParameterWorkbook:
    path: Path
    sheets: dict[str, pd.DataFrame]

    @classmethod
    def load(cls, path: str | Path) -> "ParameterWorkbook":
        path = Path(path)
        xls = pd.ExcelFile(path)
        sheets = {name: pd.read_excel(path, sheet_name=name) for name in xls.sheet_names}
        return cls(path=path, sheets=sheets)

    def sheet(self, name: str) -> pd.DataFrame:
        if name not in self.sheets:
            raise KeyError(f"Workbook is missing required sheet: {name}")
        return self.sheets[name].copy()


@functools.lru_cache(maxsize=None)
def _load_workbook_cached(resolved_path: str) -> ParameterWorkbook:
    """Load the Excel workbook from disk exactly once per resolved path per process.

    The return value is an immutable ParameterWorkbook whose .sheet() method
    always returns a fresh copy of the underlying DataFrame, so downstream
    sampling code cannot mutate the cached data.
    """
    return ParameterWorkbook.load(resolved_path)


def _clean_county(county: str) -> str:
    return str(county).strip().lower()


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value)


def _parse_array_string(value: Any) -> np.ndarray | None:
    """Parse a single cell like '[0.29, 0.47, 0.24]' into a float array, or None if not array-shaped."""
    if isinstance(value, str) and value.strip().startswith("[") and value.strip().endswith("]"):
        return np.array(ast.literal_eval(value.strip()), dtype=float)
    return None


def _sample_or_value(row: Mapping[str, Any], rng: np.random.Generator) -> float | np.ndarray | None:
    """Return a numeric value, array, or None for placeholder strings like 'TBD'."""
    value = row.get("value")
    kind = row.get("kind")
    ci_lower = row.get("ci_lower")
    ci_upper = row.get("ci_upper")
    n = row.get("n")

    # A whole array pasted into one cell, e.g. '[0.29, 0.47, 0.24]' -- always deterministic.
    array_value = _parse_array_string(value)
    if array_value is not None:
        return array_value

    # Treat blank/fixed kinds or rows without CIs as deterministic.
    if _is_missing(kind) or str(kind).lower() == "fixed" or _is_missing(ci_lower) or _is_missing(ci_upper):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None  # placeholder text like 'TBD' or 'L3 calc TBD' — skip silently

    n_arg = None if _is_missing(n) else int(n)
    return float(sample_from_ci(float(value), float(ci_lower), float(ci_upper), n=n_arg, kind=str(kind), size=1, rng=rng)[0])


def _scalar_table(df: pd.DataFrame, key_col: str, value_col: str = "value") -> dict[str, float]:
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        key = row.get(key_col)
        val = row.get(value_col)
        if not _is_missing(key) and not _is_missing(val):
            out[str(key)] = float(val)
    return out


def _as_optional_float(value: Any) -> float | None:
    """Return a numeric value, or None for blanks/placeholders such as 'x'."""
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sampled_params(wb: ParameterWorkbook, rng: np.random.Generator) -> dict[str, Any]:
    rows = wb.sheet("sampled_params")
    sampled = {row["parameter_name"]: _sample_or_value(row, rng) for _, row in rows.iterrows()}

    params: dict[str, Any] = {}
    used = set()
    for new_name, old_names in SAMPLED_ARRAYS.items():
        params[new_name] = np.array([sampled[name] for name in old_names], dtype=float)
        used.update(old_names)

    for name, value in sampled.items():
        if name not in used:
            params[name] = value
    return params


# "*_implementation_index" parameters describe how much a MOMISH intervention
# has actually been rolled out in a county. A county with no explicit row for
# one of these has never had that intervention, i.e. the correct read is 0.0 --
# never inherit whichever county happens to be listed last in the sheet.
NO_FALLBACK_INTERV_PARAMS = {
    "mentors_implementation_index",
    "prompts_implementation_index",
    "fqa_implementation_index",
    "pulse_implementation_index",
    "referral_implementation_index",
}


def _intervention_params(wb: ParameterWorkbook, rng: np.random.Generator, county: str) -> dict[str, Any]:
    """Load intervention-level parameters, preferring the row for this county.

    Some parameters (e.g. mentors_implementation_index) have one row per county
    instead of a single universal row. Grouping by parameter_name and always
    keeping the last row silently picked whichever county happened to be listed
    last in the sheet, regardless of the county actually requested. Now: use the
    row for `county` when one exists, otherwise fall back to the last row for
    that parameter_name (covers universal "All" rows and counties without their
    own row) -- except for NO_FALLBACK_INTERV_PARAMS, which default to 0.0
    instead of inheriting another county's row.
    """
    rows = wb.sheet("interv_params")
    rows = rows.dropna(subset=["parameter_name", "value"])
    county_clean = _clean_county(county)

    params: dict[str, Any] = {}
    for name, group in rows.groupby("parameter_name", sort=False):
        row = None
        if "county" in group.columns:
            county_match = group[group["county"].astype(str).str.strip().str.lower() == county_clean]
            if not county_match.empty:
                row = county_match.iloc[0]
        if row is None:
            if str(name) in NO_FALLBACK_INTERV_PARAMS:
                params[str(name)] = 0.0
                continue
            row = group.iloc[-1]
        params[str(name)] = _sample_or_value(row, rng)
    return params


def _constants(wb: ParameterWorkbook) -> dict[str, Any]:
    rows = wb.sheet("constants")
    return _scalar_table(rows, "parameter_name")


def _array_constants(wb: ParameterWorkbook) -> dict[str, np.ndarray]:
    rows = wb.sheet("array_constants")
    out: dict[str, np.ndarray] = {}
    for name, g in rows.dropna(subset=["parameter_name", "value"]).groupby("parameter_name", sort=False):
        g = g.sort_values("index") if "index" in g.columns else g
        arr = g["value"].to_numpy(dtype=float)
        # Preserve GA_sequence as integers, matching existing code.
        if name == "GA_sequence":
            arr = arr.astype(int)
        out[str(name)] = arr
    return out


def _county_rows(wb: ParameterWorkbook, sheet: str, county: str) -> pd.DataFrame:
    df = wb.sheet(sheet)
    if "county" not in df.columns:
        return df
    county_clean = _clean_county(county)
    return df[df["county"].astype(str).str.strip().str.lower().eq(county_clean)].copy()


def _county_demographics(wb: ParameterWorkbook, county: str, rng: np.random.Generator) -> dict[str, Any]:
    rows = _county_rows(wb, "county_demographics", county)
    return {
        row["parameter_name"]: _sample_or_value(row, rng)
        for _, row in rows.iterrows()
        if not _is_missing(row.get("value"))
    }


def _county_calibrated(wb: ParameterWorkbook, county: str) -> dict[str, Any]:
    rows = _county_rows(wb, "county_calibrated", county)
    rows = rows.copy()
    rows["parameter_name"] = rows["parameter_name"].map(
        lambda value: value.strip() if isinstance(value, str) else value
    )
    out: dict[str, Any] = {}
    for name, g in rows.dropna(subset=["parameter_name", "value"]).groupby("parameter_name", sort=False):
        if "index" in g.columns and g["index"].notna().any():
            out[str(name)] = g.sort_values("index")["value"].to_numpy(dtype=float)
        else:
            out[str(name)] = float(g.iloc[0]["value"])
    return out


def _county_supply(wb: ParameterWorkbook, county: str, rng: np.random.Generator) -> dict[str, Any]:
    rows = _county_rows(wb, "county_supply", county)
    out: dict[str, Any] = {}

    # These may have an extra L3 row in Excel, but the model expects
    # [home, L2/3, L4, L5]. Here, L2/3 is already the combined model value.
    collapse_l3_params = {
        "S_MgSO4",
        "S_antibiotics",
        "S_oxytocin",
    }

    model_facility_order = ["home", "L2/3", "L4", "L5"]

    for name, g in rows.dropna(subset=["parameter_name", "value"]).groupby("parameter_name", sort=False):
        name = str(name)

        if (
            name in collapse_l3_params
            and "facility_level" in g.columns
            and g["facility_level"].notna().any()
        ):
            # Skip rows whose value is a placeholder string (e.g. 'TBD').
            lookup = {}
            for _, row in g.iterrows():
                v = _sample_or_value(row, rng)
                if v is not None:
                    lookup[str(row["facility_level"]).strip()] = v

            out[name] = np.array(
                [lookup.get(level, 0.0) for level in model_facility_order],
                dtype=float,
            )

        else:
            g = g.sort_values("index") if "index" in g.columns else g
            values = [v for v in (_sample_or_value(row, rng) for _, row in g.iterrows()) if v is not None]
            if not values:
                continue
            out[name] = np.array(values, dtype=float) if len(values) > 1 else float(values[0])

    return out


def _county_facilities(wb: ParameterWorkbook, county: str) -> dict[str, Any]:
    rows = _county_rows(wb, "county_facilities", county).dropna(subset=["parameter_name", "value", "facility_level"])
    out: dict[str, Any] = {}

    def vals_for(parameter_name: str, levels=("L2/3", "L4", "L5")) -> list[float] | None:
        sub = rows[rows["parameter_name"].eq(parameter_name)]
        if sub.empty:
            return None
        lookup = {str(r["facility_level"]): float(r["value"]) for _, r in sub.iterrows()}
        return [lookup[level] for level in levels if level in lookup]

    count = vals_for("count")
    if count and len(count) == 3:
        out["num_L2/3"], out["num_L4"], out["num_L5"] = [int(x) for x in count]

    staff_map = {
        "num_surgical": "base_surgical",
        "num_nurses": "base_nurse",
        "num_anesthetists": "base_anesthetist",
    }
    for src, dst in staff_map.items():
        vals = vals_for(src)
        if vals and len(vals) == 3:
            out[dst] = [int(x) for x in vals]

    equipment_map = {
        "num_dopplers": "num_dopplers",
        "num_CTGs": "num_CTGs",
    }
    for src, prefix in equipment_map.items():
        vals = vals_for(src)
        if vals and len(vals) == 3:
            out[f"{prefix}_L2/3"], out[f"{prefix}_L4"], out[f"{prefix}_L5"] = [int(x) for x in vals]

    # Optional raw facility variables if you want them later.
    for name, g in rows.groupby("parameter_name", sort=False):
        if str(name) in {"count", "num_surgical", "num_nurses", "num_anesthetists", "num_dopplers", "num_CTGs"}:
            continue
        vals = vals_for(str(name))
        if vals:
            out[str(name)] = vals
    return out


def _calibration_targets(wb: ParameterWorkbook, county: str) -> dict[str, float]:
    rows = _county_rows(wb, "calibration_targets", county)
    out = {}
    for _, row in rows.iterrows():
        name = row.get("target_name")
        value = _as_optional_float(row.get("value"))
        if not _is_missing(name) and value is not None:
            out[str(name)] = value
    return out


def _cost_dict(wb: ParameterWorkbook) -> dict[str, float]:
    rows = wb.sheet("cost_params")
    return {str(row["cost_item"]): float(row["value"]) for _, row in rows.iterrows() if not _is_missing(row.get("value"))}


def _disability_weights(wb: ParameterWorkbook, rng: np.random.Generator) -> dict[str, float]:
    rows = wb.sheet("disability_weights")
    out = {}
    for _, row in rows.iterrows():
        name = str(row["condition"])
        key = DW_NAME_MAP.get(name, name)
        pseudo_row = {
            "value": row.get("mean"),
            "ci_lower": row.get("ci_lower"),
            "ci_upper": row.get("ci_upper"),
            "n": None,
            "kind": "mean",
        }
        out[key] = _sample_or_value(pseudo_row, rng)
    return out


# Count-like parameters (facility/equipment/volunteer counts) are used downstream as
# np.zeros/np.arange/np.tile sizes and loop counts, which require Python ints. Excel
# loads whole numbers as floats (e.g. 5040.0), so cast them explicitly after loading.
_COUNT_PARAM_KEYS = (
    "n_CHV",
    "num_L2/3",
    "num_L4",
    "num_L5",
    "num_dopplers_L2/3",
    "num_dopplers_L4",
    "num_dopplers_L5",
    "num_CTGs_L2/3",
    "num_CTGs_L4",
    "num_CTGs_L5",
)


def _cast_count_params(param: dict[str, Any]) -> dict[str, Any]:
    """Cast known count-like parameters (and any other scalar 'num_*' param) to int in place."""
    for key, value in param.items():
        if key in _COUNT_PARAM_KEYS or key.startswith("num_"):
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                param[key] = int(round(value))
    return param


@functools.lru_cache(maxsize=1)
def _default_county_fallback_params(resolved_path: str) -> dict[str, Any]:
    """Kakamega's parameter dict, used to fill in values missing for other counties.

    Cached (deterministic seed) since this is only a fallback source, not the primary
    sampling path — callers must deepcopy values out of it before use, since dict/array
    values here are shared across every county that falls back to them.
    """
    wb = _load_workbook_cached(resolved_path)
    return _build_params(wb, DEFAULT_COUNTY, rng=np.random.default_rng(0), strict_county=False)


def _build_params(
    wb: ParameterWorkbook,
    county: str,
    rng: np.random.Generator,
    strict_county: bool = False,
) -> dict[str, Any]:
    """Build a fresh sampled parameter dict from an already-loaded workbook.

    This is the hot path called inside Monte Carlo loops.  It performs no disk
    I/O — all DataFrame access goes through wb.sheet() which returns a .copy(),
    so the cached workbook is never mutated in-place.
    """
    county_clean = _clean_county(county)

    counties = wb.sheet("counties")
    available = set(counties["county"].astype(str).str.strip().str.lower())
    if strict_county and county_clean not in available:
        raise ValueError(f"Unknown county {county!r}. Available counties: {sorted(available)}")

    param: dict[str, Any] = {}
    # Shared/default inputs.
    param.update(_constants(wb))
    param.update(_array_constants(wb))
    param.update(_sampled_params(wb, rng))
    param.update(_intervention_params(wb, rng, county))
    param["cost_dict"] = _cost_dict(wb)
    param["DW"] = _disability_weights(wb, rng)

    if "fqa_pulse_modifier" in param:
        medium = param["fqa_pulse_modifier"]
        param.setdefault("fqa_pulse_modifier_level", "Medium")
        param["fqa_pulse_modifier_options"] = {
            "Low": medium * 0.5,
            "Medium": medium,
            "High": medium * 1.5,
        }

    # County-specific overrides.
    param.update(_county_demographics(wb, county, rng))
    param.update(_county_facilities(wb, county))
    param.update(_county_supply(wb, county, rng))
    param.update(_county_calibrated(wb, county))
    param.update(_calibration_targets(wb, county))

    # Fall back to Kakamega's value for any parameter this county's sheets don't yet
    # provide (e.g. supply data not yet collected for a county), per this module's
    # documented behavior. Values are deep-copied since callers mutate arrays in place.
    if county_clean != DEFAULT_COUNTY:
        fallback = _default_county_fallback_params(str(WORKBOOK_PATH.resolve()))
        for key, value in fallback.items():
            if key not in param:
                param[key] = copy.deepcopy(value)

    # Legacy baseline values that are still required by unconditional model
    # pathways. Workbook/county values take precedence whenever present.
    required_defaults = {
        "n_CHV": 420 * 12,
        "PT_scale": 0.8250540888309176,
        # Kakamega Phase 2 reference values. County multipliers scale all PPH
        # and sepsis pathways relative to this calibrated reference structure.
        "pph_incidence_multiplier": 1.0,
        "sepsis_incidence_multiplier": 1.0,
        "p_pph_other_reference": 0.0100,
        "p_mat_sepsis_other_reference": 0.0356,
        "S_pph_bundle": np.zeros(4, dtype=float),
        "p_NM_home": 0.235,
        "weight_facility_neo": 3.15,
        "p_elec_CS|highrisk_us": 0.35,
        "p_elec_CS|preterm_us": 0.7799,
        "p_cs_capacity_sdr": np.array([0, 0, 0.1215, 0.1215], dtype=float),
        "p_cs_capacity_sensor": np.array([0, 0.0568, 0.1215, 0.1215], dtype=float),
        "p_cs_capacity_sdr_sensor": np.array([0, 0, 0.1215, 0.1215], dtype=float),
        "transfer_delay_shift_2plus": 0.60,
        "transfer_delay_shift_1_2": 0.40,
    }
    for key, default in required_defaults.items():
        param.setdefault(key, default)

    # Derived convenience values.
    if "base_LB" in param:
        base_lb = np.asarray(param["base_LB"], dtype=float)
        if base_lb.size == 4 and base_lb.sum() > 0:
            param["base_p_45"] = float((base_lb[2] + base_lb[3]) / base_lb.sum())
            param["p_l5_l45"] = float(base_lb[3] / (base_lb[2] + base_lb[3])) if (base_lb[2] + base_lb[3]) > 0 else np.nan
            param["Num_Exp_L45"] = float(base_lb[2] + base_lb[3])

    _cast_count_params(param)

    param["county"] = county_clean
    return param


def get_parameters(
    rng: np.random.Generator | None = None,
    county: str | None = None,
    *,
    seed: int | None = None,
    strict_county: bool = False,
) -> dict[str, Any]:
    """Build a freshly sampled parameter dictionary.

    Parameters
    ----------
    rng:
        A numpy Generator for reproducible uncertainty draws.  A new
        non-deterministic generator is created when None.
    county:
        County name, e.g. "kakamega", "kisii", "makueni", "mombasa".
        Defaults to DEFAULT_COUNTY ("kakamega").
    seed:
        Integer seed used only when rng is None.
    strict_county:
        If True, raise an error when the county is not in the workbook.

    Notes
    -----
    The workbook is loaded from WORKBOOK_PATH (or the SDR_PARAMS_PATH env var)
    exactly once per process and cached.  Repeated calls with different rng
    values produce independently sampled dictionaries without re-reading Excel.
    """
    if county is None:
        county = DEFAULT_COUNTY
    if rng is None:
        rng = np.random.default_rng(seed)
    wb = _load_workbook_cached(str(WORKBOOK_PATH.resolve()))
    return _build_params(wb, county, rng, strict_county)


def get_fqa_pulse_modifier_options() -> dict[str, float]:
    """Static Low/Medium/High multipliers for the FQA-PULSE interaction.

    Sourced directly from the workbook's base 'fqa_pulse_modifier' value (deterministic,
    no CI/sampling for this parameter), so it can be read once at import time without
    running a full get_parameters() sampling pass.
    """
    wb = _load_workbook_cached(str(WORKBOOK_PATH.resolve()))
    rows = wb.sheet("interv_params")
    medium = float(rows.loc[rows["parameter_name"] == "fqa_pulse_modifier", "value"].iloc[0])
    return {"Low": medium * 0.5, "Medium": medium, "High": medium * 1.5}


def get_available_counties() -> list[str]:
    """List county names available in the workbook (uses the cached workbook, no re-read)."""
    wb = _load_workbook_cached(str(WORKBOOK_PATH.resolve()))
    counties = wb.sheet("counties")
    if "enabled" in counties.columns:
        counties = counties[counties["enabled"].astype(bool)]
    return sorted(counties["county"].astype(str).str.strip().str.lower().unique().tolist())


def get_slider_params(county: str | None = None) -> dict[str, Any]:
    """Load dashboard slider defaults for one county.

    Uses the cached workbook (no re-read from disk).  County defaults to
    DEFAULT_COUNTY when not specified.
    """
    if county is None:
        county = DEFAULT_COUNTY
    county_clean = _clean_county(county)
    wb = _load_workbook_cached(str(WORKBOOK_PATH.resolve()))

    out: dict[str, Any] = {}

    # Lookup-table slider arrays, e.g. p_l45_anc_slider.
    lookup = wb.sheet("lookup_tables")
    for table_name, g in lookup.dropna(subset=["table_name", "key", "value"]).groupby("table_name", sort=False):
        out[str(table_name)] = g[["key", "value"]].to_numpy(dtype=float)

    sliders = wb.sheet("slider_parameters")
    if "county" in sliders.columns:
        sliders = sliders[sliders["county"].astype(str).str.strip().str.lower().eq(county_clean)]

    for _, row in sliders.iterrows():
        name = row.get("parameter_name")
        value = row.get("default_value")
        typ = str(row.get("type", "scalar")).lower()
        if _is_missing(name) or _is_missing(value):
            continue
        if typ == "array":
            # Bracket-string format "[0, 0, 0.77, 0.77]" or plain comma-separated.
            arr = _parse_array_string(value)
            if arr is not None:
                out[str(name)] = arr
            elif isinstance(value, str):
                out[str(name)] = np.array([float(x.strip()) for x in value.split(",") if x.strip()], dtype=float)
            else:
                out[str(name)] = np.array([float(value)], dtype=float)
        else:
            out[str(name)] = float(value)

    # Sensible fallbacks from the main county parameter table.
    # Use a fixed seed so slider defaults are deterministic regardless of call order.
    params = _build_params(wb, county, rng=np.random.default_rng(0))
    out.setdefault("base_knowledge_L45_slider", params.get("base_knowledge_L45"))
    out.setdefault("base_p_45_slider", params.get("base_p_45"))
    out.setdefault("p_ANC_base_slider", params.get("p_ANC_base"))
    out.setdefault("S_pph_bundle_slider", params.get("S_pph_bundle", np.zeros(4)))
    out.setdefault("S_iv_iron_slider", params.get("S_iv_iron"))
    out.setdefault("S_MgSO4_slider", params.get("S_MgSO4"))
    out.setdefault("S_antibiotics_slider", params.get("S_antibiotics"))
    out.setdefault("S_oxytocin_slider", params.get("S_oxytocin"))
    out.setdefault("t_l23_l45_notsevere_slider", params.get("t_l23_l45_notsevere"))
    return out


def calculate_derived_parameters(param: dict[str, Any]) -> dict[str, Any]:
    """Keep your existing derived-parameter logic in one reusable place."""
    param = dict(param)  # avoid mutating the input unexpectedly

    # PT_scale is calibrated by county. Regenerate the ANC-specific gestational
    # age distributions so changing PT_scale affects the simulation.
    ga_prob = {}
    ga_counts = {}
    ga_odds = {}
    GA_assign_kenya(param, ga_counts, ga_prob)
    param["GA_anc"], param["GA_noanc"] = GA_by_ANC(param, ga_odds, ga_prob)

    param["p_anemia_anc"] = odds_prob(param["or_anc_anemia"], param["p_comp_anemia"], (1 - param["p_ANC_base"]))
    param["severe_highrisk"] = param["p_comp_severe_lowrisk"] * param["RR_comp_severe_highrisk_vs_lowrisk"]
    param["severe"] = np.array([param["p_comp_severe_lowrisk"], param["severe_highrisk"]])
    param["OL"] = np.asarray(param["p_OL"], dtype=float) * param["p_OL_scale"]

    param["OL_highrisk"], param["OL_lowrisk"] = comps_riskstatus_vs_lowrisk(
        param["OL"][0], param["p_highrisk"], param["RR_comp_highrisk_vs_lowrisk"]
    )
    param["ruptured_uterus_highrisk"], param["ruptured_uterus_lowrisk"] = comps_riskstatus_vs_lowrisk(
        param["p_ruptured_uterus"], param["p_highrisk"], param["RR_comp_highrisk_vs_lowrisk"]
    )
    param["aph_highrisk"], param["aph_lowrisk"] = comps_riskstatus_vs_lowrisk(
        param["p_aph"], param["p_highrisk"], param["RR_comp_highrisk_vs_lowrisk"]
    )
    param["eclampsia_highrisk"], param["eclampsia_lowrisk"] = comps_riskstatus_vs_lowrisk(
        param["p_eclampsia"], param["p_highrisk"], param["RR_comp_highrisk_vs_lowrisk"]
    )

    param["eclampsia_highrisk_anemia"] = comp2_comp1_anemia(param["eclampsia_highrisk"], param["or_anemia_eclampsia"])
    param["eclampsia_lowrisk_anemia"] = comp2_comp1_anemia(param["eclampsia_lowrisk"], param["or_anemia_eclampsia"])

    # Scale PPH/sepsis incidence pathways by the county-specific multiplier,
    # relative to the Kakamega Phase 2 calibrated reference structure.
    pph_multiplier = float(param.get("pph_incidence_multiplier", 1.0))
    sepsis_multiplier = float(param.get("sepsis_incidence_multiplier", 1.0))

    def scaled_probability(probability, multiplier):
        return float(np.clip(float(probability) * multiplier, 0.0, 1.0))

    param["p_pph_OL"] = scaled_probability(param["p_pph_OL"], pph_multiplier)
    param["p_pph_elective_CS"] = scaled_probability(param["p_pph_elective_CS"], pph_multiplier)
    param["p_pph_emergency_CS"] = scaled_probability(param["p_pph_emergency_CS"], pph_multiplier)
    param["p_pph_other"] = scaled_probability(param["p_pph_other_reference"], pph_multiplier)

    param["p_mat_sepsis_OL"] = scaled_probability(param["p_mat_sepsis_OL"], sepsis_multiplier)
    param["p_mat_sepsis_elective_CS"] = scaled_probability(param["p_mat_sepsis_elective_CS"], sepsis_multiplier)
    param["p_mat_sepsis_emergency_CS"] = scaled_probability(param["p_mat_sepsis_emergency_CS"], sepsis_multiplier)
    param["p_mat_sepsis_other"] = scaled_probability(param["p_mat_sepsis_other_reference"], sepsis_multiplier)

    param["pph_OL_anemia"] = comp2_comp1_anemia(param["p_pph_OL"], param["or_anemia_pph"])
    param["mat_sepsis_OL_anemia"] = comp2_comp1_anemia(param["p_mat_sepsis_OL"], param["or_anemia_sepsis"])
    param["pph_elective_CS_anemia"] = comp2_comp1_anemia(param["p_pph_elective_CS"], param["or_anemia_pph"])
    param["mat_sepsis_elective_CS_anemia"] = comp2_comp1_anemia(param["p_mat_sepsis_elective_CS"], param["or_anemia_sepsis"])
    param["pph_emergency_CS_anemia"] = comp2_comp1_anemia(param["p_pph_emergency_CS"], param["or_anemia_pph"])
    param["mat_sepsis_emergency_CS_anemia"] = comp2_comp1_anemia(param["p_mat_sepsis_emergency_CS"], param["or_anemia_sepsis"])
    param["pph_other_anemia"] = comp2_comp1_anemia(param["p_pph_other"], param["or_anemia_pph"])
    param["mat_sepsis_other_anemia"] = comp2_comp1_anemia(param["p_mat_sepsis_other"], param["or_anemia_sepsis"])

    param["RDS_T"] = P_RDS(param)

    if "S_oxytocin_l45" in param:
        param["S_oxytocin"] = np.array([0, 0 * 0.3157 + 0.33 * (1 - 0.3157), param["S_oxytocin_l45"], param["S_oxytocin_l45"]])
    if "S_preterm_treat_l45" in param:
        param["S_preterm_treat"] = np.array([0, 0, param["S_preterm_treat_l45"], param["S_preterm_treat_l45"]])
    return param


def reset_inputs(param: dict[str, Any], n_months: int) -> dict[str, Any]:
    """County-aware version of your reset function."""
    LB = np.asarray(param["base_LB"], dtype=float)
    ANC = float(param["p_ANC_base"])
    CLASS = float(param["class"])
    highrisk = float(param["p_highrisk"])
    n_chv = int(param.get("n_CHV", 0))

    track = {
        "LB": LB,
        "ANC": ANC,
        "CLASS": CLASS,
        "LB_Track": np.zeros((n_months, 4)),
        "ANC_Track": np.zeros((n_months, 4)),
        "HighRisk_Track": np.zeros((n_months, 4)),
        "Facility_Capacity_Track": np.zeros((n_months, 1)),
        "Referral_Capacity_Track": np.zeros((n_months, 1)),
        "Num_Exp_L45_Track": np.zeros((n_months, 1)),
        "Constraint_Ratio_Track": np.zeros((n_months, 1)),
        "CS_Capacity_Track": np.zeros((n_months, 1)),
        "CHV_negative_Track": np.zeros((n_months, n_chv), dtype=int),
        "CHV_memory_Track": np.zeros((n_months, n_chv), dtype=int),
    }

    track["LB_Track"][0, :] = track["LB"]
    track["ANC_Track"][0, :] = np.repeat(ANC, 4)
    track["HighRisk_Track"][0, :] = np.repeat(highrisk, 4)
    track["Facility_Capacity_Track"][0, 0] = float(param["Capacity"])
    track["Referral_Capacity_Track"][0, 0] = 0
    track["Num_Exp_L45_Track"][0, 0] = float(LB[2] + LB[3])
    track["Constraint_Ratio_Track"][0, 0] = 1
    track["CS_Capacity_Track"][0, 0] = float(np.asarray(param["p_cs_capacity"])[3])
    return track
