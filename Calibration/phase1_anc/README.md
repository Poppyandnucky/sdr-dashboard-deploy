# Phase 1 ANC calibration

## Purpose

Phase 1 calibrates initial delivery location after ANC attendance, risk
classification, and normal antenatal referral, but before intrapartum emergency
transfers. The shared implementation will run for Kakamega, Kisii, Makueni, and
Mombasa using county-specific settings rather than separate copies of the
calibration code.

The calibration runs the production ANC pathway with all interventions off. It
assigns mothers to the no-ANC, ANC predicted-low-risk, or ANC predicted-high-risk
pathway and then simulates initial delivery location. The production function
also applies the existing L4/5 capacity constraint and L4-versus-L5 split.

### Calibration specification

| Treatment | Model variable | Meaning in Phase 1 | Source or constraint |
| --- | --- | --- | --- |
| Fitted | `home_lowrisk` | Probability of home delivery among mothers with ANC who are predicted low risk | Optimized separately for each county; bounds 0.00-0.40 |
| Fitted | `close_to_L23` | Among predicted-low-risk ANC mothers choosing a facility, probability of L2/3 rather than L4/5 | Optimized separately for each county; bounds 0.05-0.98 |
| Derived | `L23_highrisk` | Probability of L2/3 delivery among ANC mothers predicted high risk; the complementary probability goes to L4/5 and home is not allowed | Recalculated from each trial's `close_to_L23`; not independently fitted |
| Fixed to Kakamega | `sen_risk_trad` | Sensitivity of traditional ANC risk classification | Kakamega value used for every county in this calibration |
| Fixed to Kakamega | `spec_risk_trad` | Specificity of traditional ANC risk classification | Kakamega value used for every county in this calibration |

All other model parameters retain their county values from `SDR Parameters.xlsx`.

### Derived high-risk routing

`L23_highrisk` changes with `close_to_L23`, but it is not an additional free
parameter. The calculation preserves Kakamega's odds ratio between L4/5 referral
for predicted-high-risk and routine predicted-low-risk mothers, using the
reference values `close_to_L23 = 0.888944` and `L23_highrisk = 0.3255`:

```text
reference_OR = odds(1 - Kakamega L23_highrisk)
               / odds(1 - Kakamega close_to_L23)

county high-risk L4/5 odds = reference_OR
                             * odds(1 - fitted close_to_L23)

L23_highrisk = 1 - P(high-risk L4/5)
```

This lets routine facility routing vary by county while retaining the reference
relationship that high-risk mothers are more likely to be directed to L4/5. It
also avoids trying to identify two closely related routing parameters from the
same delivery-location targets.

### Calibration targets

All targets are read from the workbook's `calibration_targets` sheet. Objective
targets contribute to the normalized RMSE optimized by Optuna. Diagnostic
targets are reported in `target_comparison.csv` but do not affect the fitted
parameters.

| Simulated metric | Workbook target | Definition |
| --- | --- | --- |
| `home_all` | `home_all_target` | Proportion of all mothers initially assigned to home |
| `l23_all` | `l23_all_target` | Proportion of all mothers initially assigned to L2/3 |
| `l4_all` | `l4_all_target` | Proportion of all mothers initially assigned to L4 |
| `l5_all` | `l5_all_target` | Proportion of all mothers initially assigned to L5 |
| `l45_all` | `l45_all_target` | Combined proportion initially assigned to L4 or L5 |
| `anc_among_home` | `home_anc_target` | Among home births, proportion with 4+ ANC |
| `anc_among_l23` | `l23_anc_target` | Among L2/3 births, proportion with 4+ ANC |
| `anc_among_l4` | `l4_anc_target` | Among L4 births, proportion with 4+ ANC |
| `anc_among_l5` | `l5_anc_target` | Among L5 births, proportion with 4+ ANC |
| `anc_among_l45` | `l45_anc_target` | Among combined L4/5 births, proportion with 4+ ANC |

| County | Objective targets | Diagnostic targets |
| --- | --- | --- |
| Kakamega | Home, L2/3, combined L4/5 among all mothers | Home, L2/3, combined L4/5 among mothers with 4+ ANC |
| Kisii | Home, L2/3, combined L4/5 among all mothers | Home, L2/3, combined L4/5 among mothers with 4+ ANC |
| Makueni | Home, L2/3, combined L4/5 among all mothers | Home, L2/3, combined L4/5 among mothers with 4+ ANC |
| Mombasa | Home, L2/3, and L4 among all mothers | L5 among all mothers; home, L2/3, L4, and L5 among mothers with 4+ ANC |

Mombasa excludes L5 from the objective because observed L5 deliveries include
external referrals that are outside the current closed-county model. Its split
L4 and L5 targets are prepared with `data/phase1_anc_targets.ipynb` and then
entered manually in the workbook.

## Workflow

```text
SDR Parameters.xlsx
        |
        v
county settings + Phase 1 targets
        |
        v
shared ANC simulation and objective
        |
        v
timestamped county result directory
```

The optimizer stops early when normalized RMSE is below `0.5`; otherwise it
runs to the configured maximum number of trials. With the current 5% relative
tolerance, an RMSE of `0.5` corresponds to errors averaging about half of that
tolerance.

The runner loads the canonical workbook through `parameter_loader.py` and calls
`f_ANC_LB_effect_vectorized()` from `LB_effect.py`. The production model is
therefore the sole implementation of ANC assignment, risk classification,
capacity-constrained initial delivery location, and the L4/L5 split.

## County configuration

[`config.py`](config.py) separates choices that legitimately vary by county
from shared calibration code. It records:

- targets included in the optimization objective;
- diagnostic targets reported but not optimized;
- fitted parameters and their bounds;
- parameters fixed to reference values; and
- county-specific target selections.

## Results

Each run should create:

```text
results/<county>/<timestamp>/
  calibrated_parameters.csv
  target_comparison.csv
  trials.csv
```

Generated results are ignored by Git by default. A reviewed result can be
committed deliberately as a release artifact if needed for a report.

## Implementation sequence

Install the calibration-only dependency if needed:

```bash
python -m pip install -r Calibration/phase1_anc/requirements-calibration.txt
```

Run a county calibration from the repository root:

```bash
python -m Calibration.phase1_anc.calibrate --county mombasa
```

For a short smoke test:

```bash
python -m Calibration.phase1_anc.calibrate \
  --county kakamega --trials 2 --repetitions 2
```

## Remaining validation

1. Confirm that `base_LB` represents final delivery locations for every county.
2. Reproduce the existing county results before archiving the old notebooks.
3. Confirm whether facility-capacity relocation should remain part of the Phase
   1 calibration definition; the production ANC function currently applies it.
