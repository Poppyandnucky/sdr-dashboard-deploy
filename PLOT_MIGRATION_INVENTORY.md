# Plot migration inventory

This document records the first migration pass from the plotting code in
`SDR_Dash_TI.py` to the JSON-driven frontend. It is an inventory and validation
worksheet, not a replacement for the model's scientific documentation.

## Current outcome menus

The Streamlit UI exposes 19 outcomes under five categories. Their proposed
stable IDs are recorded in `frontend/plot-catalog.json`.

| Category | Outcomes |
| --- | --- |
| System Features | Facility capacity ratio; Labor force ratio; Equipment inventory ratio |
| Process Indicators | Distribution of live births; High-risk pregnancies; ANC rate; C-section rate; Normal referral; Emergency transfer |
| Implementation Outcomes | Cost Effectiveness; DALYs; DALYs averted |
| Maternal Outcomes | Maternal complication rate; Severe maternal outcomes; Maternal mortality rate |
| Neonatal Outcomes | Preterm rate; Neonatal complication rate; Neonatal mortality rate; Stillbirth rate |

Each of these outcomes has a corresponding `selected_plot` branch in the
current Streamlit file. Many branches contain multiple tabs and plots, so an
outcome is not the same thing as a plot. The catalog therefore gives every
individual visualization its own `plotId` while retaining the category and
outcome IDs needed to build the two menus.

## Pilot outcome: maternal mortality

The first catalog pass covers the four visualizations currently nested under
the Streamlit outcome labeled **Maternal mortality rate**:

| Plot ID | Current UI location | Proposed dataset |
| --- | --- | --- |
| `maternal_mortality_ratio_by_month_and_level` | MMR by location: line chart | `monthlyMmrByLevel` |
| `maternal_mortality_ratio_period_summary_by_level` | MMR by location: bar chart | `periodMmrByLevel` |
| `maternal_death_cause_distribution_by_scenario` | Distribution of Causes | `maternalDeathsByCause` |
| `maternal_deaths_by_cause_scenario_comparison` | Death by cause | `maternalDeathsByCause` |

The last two plots deliberately share a dataset. Presentation specifications
are plot-specific, while authoritative death counts should be calculated once.

## Current MMR calculation traced from Streamlit

### Source values

- Numerator column: `Deaths`
- Denominator column: `Live Births Final`
- Multiplier: `100000`
- Original four locations: `Home`, `L2/3`, `L4`, and `L5`
- Derived levels:
  - `All = Home + L2/3 + L4 + L5`
  - `Facilities = L2/3 + L4 + L5`
  - `L4/5 = L4 + L5`
- `L4` and `L5` are removed after `L4/5` is constructed.

For each run, month, scenario, and level, the code calculates:

```text
MMR = Deaths / Live Births Final * 100,000
```

### Current monthly line calculation

The current `create_line_data()` implementation:

1. Keeps only the target/intervention scenario.
2. Calculates an exact Poisson interval for the death count of each run,
   month, and level using chi-squared quantiles.
3. Converts each count interval to a rate interval using that row's live-birth
   denominator.
4. Averages the rate, lower bound, and upper bound across runs for each month
   and level.
5. Rounds displayed values to two decimal places.

The interval is therefore the **mean of per-run Poisson interval bounds**. It
is not currently the 2.5th and 97.5th percentiles of simulated MMR values.

### Current period bar calculation

The current `create_bar_data()` implementation:

1. Sums deaths and live births over all included months for each scenario,
   run, and level.
2. Calculates MMR and a Poisson interval for each run and level.
3. Averages the rate and interval bounds across runs.

Although the current chart title says **MMR annually**, the active code uses
all simulation months. Code that once selected the final 12 months is
commented out.

### Current cause-distribution calculation

The **Distribution of Causes** plot:

1. Reads `Run`, `Scenario`, and `death_cause` from individual outcomes.
2. Removes rows where `death_cause == "none"`.
3. Counts rows by scenario and cause across all runs.
4. Divides each cause count by all attributed deaths in that scenario.

This produces the composition of attributed maternal deaths, not the cause-
specific risk among pregnancies or live births.

### Current death-by-cause comparison

The **Death by cause** plot:

1. Removes `death_cause == "none"`.
2. Counts deaths by run and cause.
3. Uses the mean count across runs when multiple runs are present.
4. Compares a reference and target scenario using absolute and percentage
   changes.
5. Uses the fixed cause order PPH, sepsis, eclampsia, obstructed labor, APH,
   and other.

## MMR decisions reviewed by Poppy

1. The display term is **maternal mortality ratio**. The calculation is deaths
   per 100,000 live births.
2. MMR is `Deaths / Live Births Final * 100,000` at every delivery level.
3. The monthly plot displays only one selected scenario at a time.
4. The validated method is the mean of the per-run Poisson lower bounds and
   the mean of the per-run Poisson upper bounds. The catalog retains the
   previously requested empirical-quantile method as an optional alternative.
5. The bar chart summarizes the full simulation and is titled **Average Ratio
   Over Full Simulation Period**. Within each run, deaths and live births are
   summed across all simulation months and MMR is calculated from those sums.
   With multiple runs, the displayed ratio is the arithmetic mean of the
   run-level full-period ratios.
6. A zero live-birth denominator produces JSON `null` for MMR and its interval
   values.
7. The displayed levels are `L2/3`, `L4`, and `L5`. Home is excluded, and L4
   and L5 remain separate.
8. Cause proportions retain the current pooled-across-runs calculation because
   individual run-level cause counts can be small.
9. The complete cause set and labels are Postpartum hemorrhage, Sepsis,
   Eclampsia, Obstructed labor, Antepartum hemorrhage, and Other.
10. The user selects the reference scenario explicitly. The frontend may use
    the first scenario as an initial fallback, but it must not infer the
    reference by comparing implementation settings.

The selectable confidence-interval method should be recorded in both the
scenario request and the result metadata so a saved result remains
interpretable and reproducible.

## Frontend decisions

1. The initial chart library is Plotly.js. It supports the required line,
   confidence-band, grouped-bar, faceted, and interactive charts, and the
   frontend developer already has Plotly experience.
2. Selecting an outcome renders every plot associated with that outcome. An
   additional plot-selection menu is not required.
3. Users do not select delivery levels. Each plot renders all levels specified
   by its catalog definition.
4. Array order in the catalog controls category, outcome, plot, level, and
   cause ordering in the frontend.
5. Editable display text—including category labels, outcome labels, titles,
   descriptions, axes, legends, levels, and cause labels—comes from the plot
   catalog rather than being hard-coded in JavaScript.

The frontend implementation should contain generic rendering behavior but as
little plot-specific wording as possible. Stable IDs remain code-facing; the
corresponding labels remain editor-facing.

## Scenario contract decisions

The initial request and result schemas are in `frontend/schemas/`, with
validated examples in `frontend/examples/`.

- The frontend creates a reference scenario automatically, but its planning
  inputs remain editable.
- One request contains between one and three ordered scenarios.
- The frontend initially selects the first scenario as the reference, but the
  request records an explicit user-selected `referenceScenarioId`.
- County, run count, duration, random-seed strategy, and confidence-interval
  method are shared by every scenario in a request.
- Scenarios use matched random seeds for fairer comparisons.
- Every model run produces all outcome datasets.
- Users edit understandable `planningInputs`.
- The backend validates those inputs and creates exact `resolvedModelInputs`.
- Results preserve both forms for reproducibility.
- Advanced users may inspect resolved model inputs but cannot edit them in the
  initial frontend.

## Proposed next implementation step

After Poppy validates the questions above, extract the MMR transformations
from the Streamlit display branch into backend functions that return two
JSON-safe datasets:

```text
monthlyMmrByLevel
periodMmrByLevel
```

Then extend the existing mortality-by-cause aggregation to produce the shared
`maternalDeathsByCause` dataset. The Streamlit charts and JSON exporter should
call the same functions so their calculations cannot drift apart.
