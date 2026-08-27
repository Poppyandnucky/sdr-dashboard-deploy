# SDR HTML Frontend Architecture

This document explains the primitive HTML/JavaScript frontend added for the SDR
dashboard and compares it with the original `SDR_Dash_TI.py` design.

## Original design

Originally, `SDR_Dash_TI.py` contained nearly the entire application:

- Streamlit page layout and styling
- Checkboxes, toggles, sliders, select boxes, and buttons
- Conditional display of dependent controls
- Translation of user selections into model flags and parameters
- Session-state management
- Baseline and intervention model execution
- Charts, tables, and file downloads

Streamlit reran the Python script whenever a user changed a control. During each
rerun, functions such as `render_hss()`, `render_single()`, and
`render_prompts()` rebuilt the interface and populated four important model
dictionaries:

| Dictionary | Purpose |
|---|---|
| `i_flags` | Turns individual interventions and model mechanisms on or off. |
| `i_HSS` | Holds health-system and MOMISH parameters. |
| `i_S` | Holds treatment, supply, and coverage parameters. |
| `i_E` | Holds diagnostic sensitivity and specificity parameters. |

The original controls were linked procedurally. For example, enabling CHVs made
the ANC and L4/5-delivery controls available. Enabling intrapartum sensors made
the AI-algorithm option available. Selecting a preset populated several flags
and parameter values at once.

This approach works well inside Streamlit, but frontend presentation and model
logic are tightly coupled in one large Python file.

## New split design

The new design separates browser presentation from Python model execution:

```text
Browser controls
    ↓
frontend/main.js
    ↓
frontend/resolveModelConfig.js
    ↓ complete {county, flags, E, S, HSS, model} object
SDR_Dash_TI.py
    ↓
run_model_dash(...)
    ↓
JSON results file + browser result preview
```

The frontend owns what the user sees and how controls behave. Python remains
the authority for validating configuration, loading county parameters, running
the simulation, and saving results.

## New files

### `frontend/index.html`

This is the entry page for the custom Streamlit component. It:

- Provides the page title and basic document structure.
- Loads `styles.css`.
- Loads `main.js` as a JavaScript module.
- Provides the root element into which JavaScript renders the interface.

It contains very little application logic intentionally.

### `frontend/styles.css`

This contains the primitive frontend's visual rules:

- Page dimensions and typography
- Fieldsets and grouped controls
- Input and button appearance
- Status and JSON-preview areas
- A small responsive layout for narrow screens

Changing this file changes appearance without changing model behavior.

### `frontend/main.js`

This is the browser-interface controller. It:

- Receives bootstrap data from `SDR_Dash_TI.py`.
- Creates semantic UI state for the visible controls.
- Renders the HTML interface.
- Conditionally displays dependent controls.
- Handles input changes.
- Calls `resolveModelConfig()` before sending data to Python.
- Sends `stateChanged`, `runModel`, and `reset` events to Streamlit.
- Displays Python errors, the saved JSON path, and result previews.
- Coordinates iframe sizing and the Streamlit component handshake.

Examples of conditional behavior currently implemented include:

- CHV-dependent ANC and L4/5-delivery controls
- Facility-upgrade child controls
- Referral and emergency-transfer dependencies
- Treatment coverage inputs
- Diagnostic accuracy inputs
- MOMISH fidelity selectors
- Multiple-run count input

This replaces much of the display logic previously expressed through calls such
as `st.toggle()`, `st.checkbox()`, `st.slider()`, and `st.selectbox()`.

### `frontend/resolveModelConfig.js`

This converts semantic user selections into the raw configuration expected by
the Python model. It:

- Starts from the current Python-provided defaults.
- Resets disabled intervention flags and values.
- Applies manual HSS selections.
- Applies Conservative, Moderate, and Aggressive presets.
- Applies Match Demand and Cannot Meet Demand supply settings.
- Resolves treatment coverage.
- Resolves diagnostic flags, sensitivity, and specificity.
- Maps MOMISH fidelity choices to implementation indices.
- Applies MOMISH facility-context choices.
- Calculates implementation period and total model duration.
- Returns the final `county`, `flags`, `E`, `S`, `HSS`, and `model` object.

In the original interface, these transformations occurred throughout several
Python rendering functions. They are now gathered into one frontend resolver,
which makes the dependency rules easier to test and explain.

The bootstrap now includes baseline slider values, the ANC-to-L4/5 lookup, and
current MOMISH implementation indices for every available county. JavaScript
can therefore change county and resolve its linked inputs without another
frontend/backend request. Python continues to apply model-level MOMISH/PULSE
interactions before simulation.

### `run_html_dev.py`

This is the development launcher. It starts two local services:

1. A small static-file server for `frontend/`, normally on port 3000.
2. Streamlit, normally on port 8501.

It also:

- Sets `SDR_HTML_UI_URL` so Streamlit knows where to find the frontend.
- Uses the bundled `data/SDR Parameters.xlsx` file when no parameter path was
  supplied explicitly.
- Adds the cross-origin headers required by Streamlit's component loader.
- Stops the frontend server when the Streamlit process ends.

Run both services with:

```bash
python3 run_html_dev.py
```

Then open `http://localhost:8501`.

Press `Ctrl+C` in the launching terminal to stop both services.

### `frontend/README.md`

This is a short operational reference with the development and direct-loading
commands.

## Changes inside `SDR_Dash_TI.py`

`SDR_Dash_TI.py` remains the backend entry point. New bridge code allows it to:

- Load a frontend from a development URL or local directory.
- Send county options, initial configuration, prior results, and errors to
  JavaScript.
- Receive a complete configuration object from JavaScript.
- Convert JSON lists back into NumPy arrays where required.
- Validate county and model-run settings.
- Run matched-seed baseline and intervention simulations.
- Convert results into JSON-safe browser data.
- Save the complete results to `outputs/sdr_results_<timestamp>.json`.
- Prevent the legacy Streamlit controls from rendering in HTML-component mode.

The original Streamlit functions remain in this file as a fallback and as a
reference for continued migration. Set the following environment variable to
show the legacy interface:

```bash
SDR_USE_HTML_UI=0 streamlit run SDR_Dash_TI.py
```

## Communication contract

Python sends a `bootstrap` object containing:

```text
countyOptions
defaultCounty
countyDefaultsByCounty
fqaPulseModifierOptions
pulseImplementationBoostOptions
state
results
error
```

JavaScript sends events containing a unique `eventId` and an event type:

| Event | Meaning |
|---|---|
| `stateChanged` | Send and validate the currently resolved frontend configuration. |
| `runModel` | Apply the resolved configuration and run the models. |
| `reset` | Restore defaults and clear HTML-interface results. |
| `ready` | Reserved for frontend readiness signaling. |

For `stateChanged` and `runModel`, JavaScript sends:

```javascript
{
  eventId: crypto.randomUUID(),
  type: "runModel",
  config: {
    county,
    flags,
    E,
    S,
    HSS,
    model
  }
}
```

## Results

Each successful HTML-interface run writes a timestamped JSON file under
`outputs/` using this filename pattern:

```text
outputs/sdr_results_YYYYMMDDTHHMMSSffffffZ.json
```

The timestamp is UTC. A new file is created for every successful run; an older
result file is not overwritten.

### Top-level JSON structure

The complete saved-file structure is:

```json
{
  "metadata": {
    "formatVersion": 3,
    "tableEncoding": "named-column-records",
    "createdAt": "2026-08-27T15:57:21.035996+00:00",
    "county": "kisii",
    "nRuns": 1,
    "nMonths": 36,
    "configuration": {
      "county": "kisii",
      "flags": {},
      "E": {},
      "S": {},
      "HSS": {},
      "model": {}
    }
  },
  "baseline": {
    "rowCount": 36,
    "rows": []
  },
  "intervention": {
    "rowCount": 36,
    "rows": []
  },
  "mortalityDeathCauses": {
    "perRun": {
      "rowCount": 0,
      "rows": []
    },
    "summary": {
      "rowCount": 0,
      "rows": []
    }
  }
}
```

It contains:

- Run metadata and the resolved configuration
- Baseline aggregate results
- Intervention aggregate results
- Mortality death-cause counts by run and scenario
- Mortality death-cause totals, mean counts per run, and proportions

### Metadata

| Field | Meaning |
|---|---|
| `formatVersion` | Saved JSON contract version. Frontend code should check this before parsing. |
| `tableEncoding` | Indicates that every table row is an object containing named columns. |
| `createdAt` | UTC date and time at which Python created the file. |
| `county` | County used to load model parameters. |
| `nRuns` | Number of stochastic model runs represented in each scenario. |
| `nMonths` | Number of simulated months per run. |
| `configuration` | Complete resolved input sent to the Python model. |

`metadata.configuration` contains:

| Field | Meaning |
|---|---|
| `county` | Selected county. |
| `flags` | Integer/Boolean intervention switches used by the model. |
| `E` | Diagnostic sensitivity and specificity parameters. |
| `S` | Treatment supply and coverage parameters. |
| `HSS` | Health-system and MOMISH parameters. |
| `model` | Implementation duration, maintenance duration, and run-count settings. |

Keeping the resolved configuration with the results makes a file reproducible
and lets Poppy confirm exactly which values reached the backend.

### Baseline and intervention tables

The `baseline` table comes from Python's `b_df`. The `intervention` table comes
from `i_df`. `run_model_dash()` originally names these returned values `df`.

These are aggregated model results, normally one row per simulated month and
run. A 36-month single run therefore normally produces 36 baseline rows and 36
intervention rows. Multiple runs produce approximately:

```text
nMonths × nRuns
```

rows per scenario.

Typical columns include:

- `Month`, `Run`, and `Scenario`
- Live births and delivery-location distributions
- ANC and high-risk pregnancy counts
- Referral and emergency-transfer counts
- Treatment coverage
- Maternal and neonatal complications
- Maternal and neonatal deaths
- C-sections and delivery methods
- DALYs
- Facility capacity, staffing, and equipment values

Some aggregate cells contain arrays rather than single numbers. These usually
represent facility-level values. NumPy arrays are written as ordinary JSON
arrays:

```json
{
  "Month": 0,
  "Scenario": "Baseline",
  "ANC": [100, 250, 300, 150]
}
```

Poppy should identify which aggregate columns are stable backend outputs and
which were created only as intermediate inputs for the original Streamlit
charts. Ayo can then consume only the approved columns.

### Mortality death-cause tables

The death-cause tables are the only current JSON results derived from the large
individual DataFrames `b_ind_outcomes` and `i_ind_outcomes`.

Python performs the aggregation before JSON conversion. Rows whose
`death_cause` value is missing or equals `"none"` are excluded.

#### `mortalityDeathCauses.perRun`

This table contains one row for each observed run, scenario, and attributed
death cause:

```json
{
  "Run": 1,
  "Scenario": "Intervention",
  "death_cause": "pph",
  "count": 8
}
```

Its columns are:

| Column | Meaning |
|---|---|
| `Run` | One-based stochastic run number. |
| `Scenario` | Normally `Baseline` or `Intervention`. |
| `death_cause` | Cause label produced by the mortality model. |
| `count` | Number of attributed maternal deaths for that cause in that run. |

A cause with zero deaths in a run is not emitted as a row.

#### `mortalityDeathCauses.summary`

This table contains one row for each scenario and observed cause:

```json
{
  "Scenario": "Intervention",
  "death_cause": "pph",
  "totalCount": 21,
  "meanCountPerRun": 7.0,
  "proportionOfDeaths": 0.35
}
```

Its calculated fields are:

```text
totalCount
    = sum of the cause count across all runs

meanCountPerRun
    = totalCount ÷ number of runs in the scenario

proportionOfDeaths
    = totalCount ÷ total attributed deaths from all causes in the scenario
```

`proportionOfDeaths` is stored as a fraction. For example, `0.35` means 35%.

The aggregation happens in `_aggregate_mortality_death_causes()` inside
`SDR_Dash_TI.py`. If Poppy changes mortality cause labels or attribution logic,
this contract should be reviewed.

### Individual-level data policy

Individual-level baseline and intervention records are retained only long
enough in Python to calculate the mortality death-cause aggregates. They are
not converted to JSON or included in the saved file. Additional individual
aggregations can be added later if the frontend requires them.

Specifically, these large Python variables are not serialized:

```text
b_ind_outcomes
i_ind_outcomes
```

If another frontend result needs individual data, the preferred pattern is:

1. Aggregate the necessary columns in Python.
2. Give the aggregate a documented JSON name and schema.
3. Add only that aggregate to the browser and saved file.
4. Avoid restoring the complete individual tables unless a separate raw-data
   export is explicitly required.

### Table representation

Saved result tables use a readable `formatVersion: 3` representation:

```json
{
  "rowCount": 100,
  "rows": [
    {"Month": 0, "Scenario": "Baseline", "outcome_1": 12.3},
    {"Month": 1, "Scenario": "Baseline", "outcome_1": 12.8}
  ]
}
```

Every row includes its column names so frontend developers can read and consume
the file without separately matching a `columns` array to positional data. The
individual-level tables remain excluded, so this readable representation stays
small. Files are written without indentation whitespace; a text editor can
prettify them when needed.

Missing Pandas values and non-finite floating-point values are serialized as
JSON `null`. NumPy scalar values become ordinary JSON numbers, and NumPy arrays
become JSON arrays.

### Reading the saved file in JavaScript

```javascript
const response = await fetch("sdr_results_example.json");
const results = await response.json();

if (results.metadata.formatVersion !== 3) {
  throw new Error("Unsupported SDR result format");
}

const monthlyBaseline = results.baseline.rows;
const monthlyIntervention = results.intervention.rows;
const mortalitySummary =
  results.mortalityDeathCauses.summary.rows;

const pphResults = mortalitySummary.filter(
  row => row.death_cause === "pph"
);
```

### Reading the saved file in Python

```python
import json
import pandas as pd

with open("sdr_results_example.json", encoding="utf-8") as file:
    results = json.load(file)

if results["metadata"]["formatVersion"] != 3:
    raise ValueError("Unsupported SDR result format")

baseline = pd.DataFrame(results["baseline"]["rows"])
intervention = pd.DataFrame(results["intervention"]["rows"])
mortality = pd.DataFrame(
    results["mortalityDeathCauses"]["summary"]["rows"]
)
```

### Saved file versus browser payload

The saved JSON file and live browser payload contain the same categories of
results, but their table wrappers differ slightly:

- Saved file: `{rowCount, rows}`
- Browser payload: `{columns, rows, rowCount, truncated}`

The browser's `columns` list helps render empty tables and establish display
order. `truncated` indicates whether a browser preview was limited by
`SDR_HTML_UI_MAX_ROWS`. The saved file is not subject to that preview limit.

The browser also receives result previews. Browser previews can be truncated by
`SDR_HTML_UI_MAX_ROWS`, while the saved JSON file contains the complete result
tables.

### Versioning responsibilities

Changes that rename top-level sections, rename columns, change table encoding,
or alter a field's meaning should increment `metadata.formatVersion`. Additive
fields that do not change existing meanings may remain within the current
version, but should still be documented here.

For coordination:

- Poppy owns the meaning and validity of model-result columns and aggregations.
- Ayo owns their presentation and frontend transformations.
- `SDR_Dash_TI.py` owns validation, serialization, and the boundary between the
  two.

## Development versus deployment

During development, the frontend and Streamlit run as separate local services:

```text
http://127.0.0.1:3000  → HTML/JavaScript/CSS
http://localhost:8501  → Streamlit and Python model
```

For direct local or production-style loading, Streamlit can serve the frontend
directory itself:

```bash
SDR_HTML_UI_BUILD_DIR="$PWD/frontend" streamlit run SDR_Dash_TI.py
```

In that mode, a separate frontend server is unnecessary.

## Suggested next steps

1. Let Ayo replace the primitive markup and CSS while preserving the event
   contract.
2. Add frontend tests for `resolveModelConfig()`.
3. Migrate result charts and table-selection behavior into JavaScript.
4. Remove the legacy Streamlit rendering code only after all original features
   have equivalent frontend implementations.
