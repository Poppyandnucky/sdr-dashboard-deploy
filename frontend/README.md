# SDR primitive HTML frontend

From the repository root, start the frontend and Streamlit together:

```bash
python3 run_html_dev.py
```

Then open `http://localhost:8501`.

The frontend uses plain HTML, CSS, and JavaScript modules. It has no npm or
build dependency. Changes to frontend files appear after refreshing the page.
Plotly.js is loaded from its pinned CDN build.

After a model run, the Streamlit bridge supplies `results.scenarioResult` in
the shape defined by `schemas/scenario-result.schema.json`. Outcome category,
outcome, plot ordering, chart fields, and display labels are read from
`plot-catalog.json`; the first implemented outcome is maternal mortality ratio.
The legacy baseline/intervention table payload remains available during the
migration so the existing Streamlit workflow and saved exports keep working.

For direct component loading without a separate development server:

```bash
SDR_HTML_UI_BUILD_DIR="$PWD/frontend" streamlit run SDR_Dash_TI.py
```
