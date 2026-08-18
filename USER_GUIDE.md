USER GUIDE — SDR Dashboard
===========================

This user guide explains how to use the SDR Dashboard (Streamlit app) and summarizes the deployment README. It is written for dashboard users and analysts — not a developer manual. You do not need to run the dashboard to follow this guide.

---

Included README (deployment steps)
----------------------------------

# SDR Dashboard Docker Deployment

This project runs as a Streamlit app. The Docker setup keeps the existing pinned Python package versions from `requirements.txt` and uses the same app entrypoint as the existing `Procfile`.

## 1. Build and run locally

Install Docker Desktop, then run:

```sh
docker compose up --build
```

Open:

http://localhost:8509

To use a different host port:

```sh
HOST_PORT=8080 docker compose up --build
```

Then open `http://localhost:8080`.

## 2. Stop the app

Press `Ctrl+C` in the terminal running Docker Compose, then run:

```sh
docker compose down
```

## 3. Build a standalone image

```sh
docker build -t sdr-dashboard:latest .
```

Run it with:

```sh
docker run --rm -p 8509:8501 sdr-dashboard:latest
```

## 4. Deploy to a cloud Docker host

Use any host that can run Docker containers, such as a VM, container service, or app platform. The container listens on `0.0.0.0` and uses `PORT`, defaulting to `8501`.

Typical steps:

1. Push this repository to your Git provider.
2. Configure the cloud service to build from the `Dockerfile`.
3. Expose container port `8501`, or set the platform's `PORT` variable if it assigns one.
4. Deploy the container.

The parameter workbook is included at `data/SDR Parameters.xlsx`, and the container sets:

```text
SDR_PARAMS_PATH=/app/data/SDR Parameters.xlsx
```

If you replace the workbook later, rebuild and redeploy the image so the container includes the new file.

---

Using the Dashboard — Quick Overview
-----------------------------------

This section explains the dashboard controls, modes, and recommended usage patterns.

Main layout and controls
- Top right: county selector — choose a single county (or region) to run analyses for.
- Central panel: intervention controls (sliders, toggles) and scenario presets.
- Right or bottom (depending on layout): action buttons such as `Run Model`, `Reset Model`, `Clear Cache`, and `Export CSV`.

Modes
- Preset Scenario Mode: choose a named preset. This immediately loads a set of default slider values and toggles curated for common scenarios. Recommended for most users.
- Customize (Analyst) Mode: manually adjust sliders and toggles. In this mode, switches and sliders reflect the analyst's explicit choices. Many switches will use default slider values when first toggled on.

Health System Strength (HSS) control
- There are two HSS modes: `Preset` and `Customize`.
- In `Customize`, sliders allow granular HSS adjustments (coverage, access, quality). Use the collapse/expand control next to each group to hide or show detailed sliders.
- The collapse/expand control: click the chevron or section title to collapse (hide) or expand (show) the sliders for that group.
- HSS quick toggles: you can switch between `High`, `Low`, or `No HSS` to simulate broad programmatic changes. These quick options override the detailed sliders until you re-open the group and edit values.

Intervention groups
- Treatment interventions: the dashboard exposes five treatment intervention controls. Each intervention is represented by either a slider (coverage/effect size) or an on/off toggle in the UI. These controls map directly to model inputs (see parameters workbook).
  - Tip: hover or click the label to see a short tooltip describing what the intervention represents and which model parameter it modifies.
- Momish interventions: a second group for maternal/newborn-focused ("momish") interventions. These use similar sliders or toggles and may include coverage, timing, or intensity settings.
- Combined HSS + Intervention experiments: you can enable `Low/High/No HSS` in the momish or treatment panels to explore combined effects. This is useful to test whether an intervention performs differently under weak versus strong health systems.

Outcome comparison modes
- Baseline vs User Settings: Compare model outputs when all interventions are OFF (baseline) versus the current dashboard settings. This mode shows the incremental effect of your configuration.
- Scenario A vs Scenario B: Compare two saved or current configurations side-by-side. Use `Save as A` / `Save as B` or rename the scenario labels to meaningful names (e.g., "Intervention package A: Community + antibiotics").
  - To rename scenario A or B: open the comparison panel and edit the label text.

Running the model
- Select the county at the top right before running the model.
- Press `Run Model` to execute a single model run using the current settings and the selected county.
- The app uses a stochastic model: we recommend running the model multiple times (10+ runs) to stabilize results and average outcomes. Use the `Number of runs` control if available, or press `Run Model` repeatedly.
- If you change many settings, click `Reset Model` or `Clear Cache` before running to ensure no stale cached results affect your run.

Exporting results (CSV)
- After running a model, you can export results to CSV using the `Export CSV` or `Download` button.
- CSV contents typically include:
  - Run ID and timestamp
  - Selected county/region
  - Mode used (Preset, Customize, A vs B)
  - For each intervention: slider value or toggle state used in the run
  - HSS state and any quick-toggles applied (High/Low/No HSS)
  - Aggregated model outcomes (for each outcome shown in the dashboard): e.g., mortality, cases averted, or other model-specific metrics
  - If you ran multiple stochastic iterations, the CSV will include iteration-level rows or averaged summary rows depending on the app export option.
- Use the CSV to post-process results in Excel, Python, or R. Filenames include the timestamp and scenario name for traceability.

Useful UI tips
- `All` mode: includes all counties or regions in a single sweep run if the app supports it. This is useful for broad comparisons, but runs may take longer.
- Preset mode is recommended for reproducible comparisons — presets use fixed default slider values chosen by analysts.
- Analyst mode defaults: many toggles that enable an intervention will populate the slider with a recommended default value the first time they are turned on.
- Rename scenarios: meaningful names in A vs B comparisons make exported CSVs easier to read.
- Run multiple times: stochastic models often require repeated runs — we recommend 10+ runs for stable summaries.

Troubleshooting and recommended workflow
- Before making bulk changes or running long experiments, click `Reset Model` / `Clear Cache` to ensure a clean state.
- If results look unexpected, check: county selection, whether HSS quick toggles are active, and that your scenario mode is the intended one (Preset vs Customize).
- When sharing results, export the CSV and include the scenario name and the README parameter workbook reference (`data/SDR Parameters.xlsx`) so recipients can trace inputs.

Advanced notes (where to find more information)
- Parameter workbook: `data/SDR Parameters.xlsx` contains the model parameters and mappings from UI controls to model inputs.
- For deployment and running locally, see the top "Included README (deployment steps)" section above.

Feedback and edits
- If you want more user-facing screenshots, a short video walkthrough, or an interactive notebook that steps through common workflows, tell me which scenario you'd like documented and I will add it.

---

End of user guide.
