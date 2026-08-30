import { resolveModelConfig } from "./resolveModelConfig.js";

const app = document.getElementById("app");
let bootstrap = null;
let defaultConfig = null;
let ui = null;
let selectedOutcomeCategoryId = null;
let selectedOutcomeId = null;
let selectedLineScenarioId = null;

const clone = value => JSON.parse(JSON.stringify(value));
const esc = value => String(value).replace(/[&<>"']/g, character => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[character]);
const checked = value => value ? "checked" : "";
const selected = (a, b) => a === b ? "selected" : "";

function send(value) {
  parent.postMessage({ isStreamlitMessage: true, type: "streamlit:setComponentValue", value }, "*");
}

function resize() {
  parent.postMessage({
    isStreamlitMessage: true,
    type: "streamlit:setFrameHeight",
    height: document.documentElement.scrollHeight,
  }, "*");
}

function emit(type, includeConfig = true) {
  const value = {
    eventId: crypto.randomUUID?.() || `${Date.now()}-${Math.random()}`,
    type,
  };
  if (includeConfig) {
    value.config = resolveModelConfig(
      ui,
      defaultConfig,
      bootstrap.countyDefaultsByCounty
    );
  }
  send(value);
}

function initialUI(state) {
  const flags = state.flags || {};
  const HSS = state.HSS || {};
  const S = state.S || {};
  const E = state.E || {};
  const model = state.model || {};
  const treatment = (flag, parameter) => ({
    enabled: Boolean(flags[flag]),
    coveragePercent: Math.round(Number(S[parameter] || 1) * 100),
  });
  const program = (flag, level = "Current") => ({ enabled: Boolean(flags[flag]), level });

  return {
    county: state.county,
    hss: {
      mode: "manual", demandPreset: "Conservative", supplyScenario: "Match Demand",
      employCHV: Boolean(flags.flag_CHV), increaseANC: Boolean(flags.flag_ANC),
      pANCPercent: Math.round(Number(HSS.P_ANC || 0) * 100),
      increaseL45Delivery: Boolean(flags.flag_LB),
      pL45Percent: Math.round(Number(HSS.P_L45 || 0) * 100),
      memoryDecayMonths: HSS.tau_decay || 6,
      chvMemoryModel: HSS.CHV_memory || "Logistic Decay",
      upgradeFacilities: Boolean(flags.flag_performance || flags.flag_capacity || flags.flag_labor || flags.flag_equipment),
      improvePerformance: Boolean(flags.flag_performance), performancePercent: Math.round(Number(HSS.knowledge || 0) * 100),
      increaseCapacity: Boolean(flags.flag_capacity), capacityPercent: Math.round(Number(HSS.capacity_added || 0) * 100),
      increaseLabor: Boolean(flags.flag_labor), laborPercent: Math.round(Number(HSS.labor_ratio || 0) * 100),
      increaseEquipment: Boolean(flags.flag_equipment), equipmentPercent: Math.round(Number(HSS.sensor_ratio || 0) * 100),
      upgradeRescueNetwork: Boolean(flags.flag_refer_voucher || flags.flag_transfer_capacity),
      improveReferralCapacity: Boolean(flags.flag_refer_voucher), referralPercent: Math.round(Number(HSS.P_refer || 0) * 100),
      enableEmergencyTransfer: Boolean(flags.flag_transfer_capacity), transferPercent: Number(HSS.transfer_capacity_target || 0),
    },
    treatments: {
      pphBundle: treatment("flag_pph_bundle", "pph_bundle"), ivIron: treatment("flag_iv_iron", "iv_iron"),
      magnesiumSulfate: treatment("flag_MgSO4", "MgSO4"), antibiotics: treatment("flag_antibiotics", "antibiotics"),
      oxytocin: treatment("flag_oxytocin", "oxytocin"),
    },
    diagnosis: {
      ultrasound: { enabled: Boolean(flags.flag_us), sensitivity: E.sens_us ?? 0.95, specificity: E.spec_us ?? 0.95 },
      intrapartumSensor: { enabled: Boolean(flags.flag_intrasensor), aiEnabled: Boolean(flags.flag_sensor_ai), sensitivity: E.sens_sensor ?? 0.95, specificity: E.spec_sensor ?? 0.95 },
    },
    momish: {
      facilityContext: "follow", prompts: program("flag_PROMPTS"), mentors: program("flag_MENTOR"),
      pulse: program("flag_pulse"), fqa: program("flag_fqa"), blood: program("flag_blood"),
      transfer: program("flag_transfer_delay"), fqaPulseModifierLevel: HSS.fqa_pulse_modifier_level || "Medium",
      fqaPulseModifier: HSS.fqa_pulse_modifier ?? 0.2,
      pulseBoostLevel: HSS.pulse_implementation_boost_level || "Moderate",
    },
    model: {
      implementationYears: model.imple_time ?? 3, maintenanceYears: model.main_time ?? 0,
      multipleRun: Boolean(model.multiple_run), numberOfRuns: model.n_runs ?? 1,
    },
  };
}

function applyCountyDefaults(county) {
  const defaults = bootstrap.countyDefaultsByCounty?.[county];
  if (!defaults) return;
  ui.hss.pANCPercent = Math.round(Number(defaults.basePANC) * 100);
  ui.hss.pL45Percent = Math.round(Number(defaults.basePL45) * 100);
  ui.hss.performancePercent = Math.round(Number(defaults.baseKnowledge) * 100);
}

function renderCountyDefaults() {
  const defaults = bootstrap.countyDefaultsByCounty?.[ui.county];
  if (!defaults) return '<div class="status">No county defaults available.</div>';
  const percent = value => `${(Number(value || 0) * 100).toFixed(1)}%`;
  const values = [
    ["Baseline ANC", percent(defaults.basePANC)],
    ["Baseline L4/5 delivery", percent(defaults.basePL45)],
    ["Baseline worker knowledge", percent(defaults.baseKnowledge)],
    ["Current PROMPTS", percent(defaults.promptsImplementationIndex)],
    ["Current MENTORS", percent(defaults.mentorsImplementationIndex)],
    ["Current PULSE", percent(defaults.pulseImplementationIndex)],
    ["Current FQA", percent(defaults.fqaImplementationIndex)],
    ["Current referral", percent(defaults.referralImplementationIndex)],
  ];
  return `<div class="help">These values come from the parameter workbook and update locally when the county changes.</div>
    <div class="defaults-grid">${values.map(([label, value]) => `<div class="default-value">${esc(label)}<strong>${esc(value)}</strong></div>`).join("")}</div>`;
}

function field(label, path, value, type = "number", attributes = "") {
  return `<div class="field"><label>${esc(label)}</label><input data-path="${path}" type="${type}" value="${esc(value)}" ${attributes}></div>`;
}

function toggle(label, path, value) {
  return `<div class="field"><label>${esc(label)}</label><input data-path="${path}" type="checkbox" ${checked(value)}></div>`;
}

function fidelity(name, label) {
  const item = ui.momish[name];
  return `${toggle(`Enable ${label}`, `momish.${name}.enabled`, item.enabled)}${item.enabled ? `
    <div class="field"><label>${esc(label)} implementation level</label><select data-path="momish.${name}.level">
      ${["Current", "Moderate", "High"].map(level => `<option ${selected(level, item.level)}>${level}</option>`).join("")}
    </select></div>` : ""}`;
}

function treatment(name, label) {
  const item = ui.treatments[name];
  return `${toggle(label, `treatments.${name}.enabled`, item.enabled)}${item.enabled ? field("Coverage (%)", `treatments.${name}.coveragePercent`, item.coveragePercent, "number", 'min="0" max="100"') : ""}`;
}

function scenarioResult() {
  return bootstrap.results?.scenarioResult || null;
}

function datasetFor(scenario, datasetId) {
  return scenario?.datasets?.find(dataset => dataset.datasetId === datasetId);
}

function catalogOutcomeMenus() {
  const categories = bootstrap.plotCatalog?.outcomeCategories || [];
  if (!categories.length) return "";
  const availableOutcomeIds = new Set((bootstrap.plotCatalog.plots || []).map(plot => plot.outcomeId));
  const firstAvailableCategory = categories.find(category =>
    category.outcomes.some(outcome => availableOutcomeIds.has(outcome.outcomeId))
  ) || categories[0];
  if (!categories.some(category => category.outcomeCategoryId === selectedOutcomeCategoryId)) {
    selectedOutcomeCategoryId = firstAvailableCategory.outcomeCategoryId;
  }
  const category = categories.find(item => item.outcomeCategoryId === selectedOutcomeCategoryId);
  if (!category.outcomes.some(outcome => outcome.outcomeId === selectedOutcomeId)) {
    selectedOutcomeId = category.outcomes.find(outcome => availableOutcomeIds.has(outcome.outcomeId))?.outcomeId
      || category.outcomes[0]?.outcomeId;
  }
  return `<section class="outcome-controls"><div class="field"><label>Outcome category</label>
    <select id="outcome-category">${categories.map(item => `<option value="${esc(item.outcomeCategoryId)}" ${selected(item.outcomeCategoryId, selectedOutcomeCategoryId)}>${esc(item.outcomeCategoryLabel)}</option>`).join("")}</select></div>
    <div class="field"><label>Outcome</label><select id="outcome">${category.outcomes.map(outcome =>
      `<option value="${esc(outcome.outcomeId)}" ${selected(outcome.outcomeId, selectedOutcomeId)} ${availableOutcomeIds.has(outcome.outcomeId) ? "" : "disabled"}>${esc(outcome.outcomeLabel)}</option>`
    ).join("")}</select></div></section>`;
}

function resultMarkup(result) {
  if (!result) return `<h2>Outcomes</h2><div class="status">Run the model to view outcomes.</div>`;
  const contract = result.scenarioResult;
  if (!contract) return `<h2>Outcomes</h2><div class="status">This result does not include scenario datasets.</div>`;
  const menus = catalogOutcomeMenus();
  const plots = (bootstrap.plotCatalog?.plots || []).filter(plot => plot.outcomeId === selectedOutcomeId);
  return `<h2>Outcomes</h2>${menus}<div id="plot-status" class="status chart-status" hidden></div>
    <div class="plot-grid">${plots.map(plot => `<article class="plot-card"><h3>${esc(plot.display?.title || plot.plotLabel)}</h3><p>${esc(plot.description || "")}</p><div class="plot" id="plot-${esc(plot.plotId)}"></div></article>`).join("")}</div>`;
}

const plotColors = ["#17745d", "#e7765b", "#476f95", "#c49a36", "#775a9b", "#769b55"];
const seriesValues = plot => plot.series?.values || plot.categories || [];
const seriesLabel = (plot, id) => seriesValues(plot).find(value => value.id === id)?.label || id;
const catalogValuesForField = field => (bootstrap.plotCatalog?.plots || [])
  .find(item => item.series?.field === field)?.series?.values || [];
const commonLayout = plot => ({
  title: { text: plot.display?.title || plot.plotLabel, font: { size: 17 } },
  margin: { l: 70, r: 25, t: 55, b: 70 }, paper_bgcolor: "#fff", plot_bgcolor: "#fff",
  xaxis: { title: plot.x?.label, automargin: true, gridcolor: "#edf1ef" },
  yaxis: { title: plot.y?.label, rangemode: "tozero", automargin: true, gridcolor: "#edf1ef" },
  legend: { title: { text: plot.display?.legendTitle || plot.series?.label || "" }, orientation: "h", y: -0.25 },
});

function renderLinePlot(element, plot, contract) {
  const scenario = contract.scenarios.find(item => item.scenarioId === selectedLineScenarioId) || contract.scenarios[0];
  selectedLineScenarioId = scenario.scenarioId;
  const rows = datasetFor(scenario, plot.datasetId)?.rows || [];
  const traces = [];
  seriesValues(plot).forEach((series, index) => {
    const values = rows.filter(row => row[plot.series.field] === series.id).sort((a, b) => a[plot.x.field] - b[plot.x.field]);
    traces.push({ x: values.map(row => row[plot.x.field]), y: values.map(row => row[plot.confidenceInterval.lowerField]), mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false });
    traces.push({ x: values.map(row => row[plot.x.field]), y: values.map(row => row[plot.confidenceInterval.upperField]), mode: "lines", fill: "tonexty", fillcolor: `${plotColors[index]}26`, line: { width: 0 }, hoverinfo: "skip", showlegend: false });
    traces.push({ x: values.map(row => row[plot.x.field]), y: values.map(row => row[plot.y.field]), name: series.label, mode: "lines", line: { color: plotColors[index], width: 2.5 } });
  });
  const layout = commonLayout(plot);
  layout.updatemenus = [{ type: "dropdown", x: 1, xanchor: "right", y: 1.18, buttons: contract.scenarios.map(item => ({
    label: item.label, method: "skip", args: [item.scenarioId], execute: false,
  })) }];
  Plotly.newPlot(element, traces, layout, { responsive: true, displaylogo: false });
  element.on("plotly_buttonclicked", event => { selectedLineScenarioId = event.button.args[0]; renderPlots(); });
}

function renderBarPlot(element, plot, contract) {
  const facetValues = catalogValuesForField(plot.facet.field);
  const facetLabel = id => facetValues.find(value => value.id === id)?.label || id;
  const traces = contract.scenarios.map((scenario, index) => {
    const rows = datasetFor(scenario, plot.datasetId)?.rows || [];
    return { type: "bar", name: scenario.label, x: rows.map(row => facetLabel(row[plot.facet.field])), y: rows.map(row => row[plot.y.field]),
      marker: { color: plotColors[index] }, error_y: { type: "data", symmetric: false, array: rows.map(row => row[plot.confidenceInterval.upperField] - row[plot.y.field]), arrayminus: rows.map(row => row[plot.y.field] - row[plot.confidenceInterval.lowerField]) } };
  });
  Plotly.newPlot(element, traces, { ...commonLayout(plot), barmode: "group" }, { responsive: true, displaylogo: false });
}

function renderNormalizedPlot(element, plot, contract) {
  const traces = seriesValues(plot).map((series, index) => ({ type: "bar", name: series.label, x: contract.scenarios.map(item => item.label),
    y: contract.scenarios.map(item => datasetFor(item, plot.datasetId)?.rows.find(row => row[plot.series.field] === series.id)?.[plot.y.field] ?? 0), marker: { color: plotColors[index] } }));
  Plotly.newPlot(element, traces, { ...commonLayout(plot), barmode: "stack", yaxis: { ...commonLayout(plot).yaxis, range: [0, 1], tickformat: ".0%" } }, { responsive: true, displaylogo: false });
}

function renderDeltaPlot(element, plot, contract) {
  const reference = contract.scenarios.find(item => item.scenarioId === contract.referenceScenarioId) || contract.scenarios[0];
  const comparison = contract.scenarios.find(item => item.scenarioId !== reference.scenarioId);
  if (!comparison) return;
  const refRows = datasetFor(reference, plot.datasetId)?.rows || [];
  const cmpRows = datasetFor(comparison, plot.datasetId)?.rows || [];
  const categories = seriesValues(plot);
  const base = categories.map(item => refRows.find(row => row.deathCauseId === item.id)?.[plot.y.field] ?? 0);
  const changes = categories.map((item, index) => (cmpRows.find(row => row.deathCauseId === item.id)?.[plot.y.field] ?? 0) - base[index]);
  const title = (plot.display?.titleTemplate || plot.plotLabel).replace("{referenceScenarioLabel}", reference.label).replace("{comparisonScenarioLabel}", comparison.label);
  const layout = commonLayout(plot); layout.title.text = title; layout.barmode = "relative";
  Plotly.newPlot(element, [{ type: "bar", name: reference.label, x: categories.map(item => item.label), y: base, marker: { color: plotColors[0] } },
    { type: "bar", name: `${comparison.label} change`, x: categories.map(item => item.label), y: changes, marker: { color: plotColors[1] } }], layout, { responsive: true, displaylogo: false });
}

function renderPlots() {
  const contract = scenarioResult();
  if (!contract || typeof Plotly === "undefined") return;
  (bootstrap.plotCatalog?.plots || []).filter(plot => plot.outcomeId === selectedOutcomeId).forEach(plot => {
    const element = document.getElementById(`plot-${plot.plotId}`); if (!element) return;
    if (plot.chartType === "lineWithConfidenceInterval") renderLinePlot(element, plot, contract);
    else if (plot.chartType === "barWithConfidenceInterval") renderBarPlot(element, plot, contract);
    else if (plot.chartType === "normalizedStackedBar") renderNormalizedPlot(element, plot, contract);
    else if (plot.chartType === "referencePlusDeltaBar") renderDeltaPlot(element, plot, contract);
  });
  resize();
}

function renderHSS() {
  const h = ui.hss;
  const mode = `<div class="field"><label>Configuration mode</label><select data-path="hss.mode"><option value="manual" ${selected(h.mode, "manual")}>Manual</option><option value="preset" ${selected(h.mode, "preset")}>Preset</option></select></div>`;
  if (h.mode === "preset") return `<fieldset><legend>Health Systems Strengthening</legend>${mode}
    <div class="field"><label>Demand preset</label><select data-path="hss.demandPreset">${["Conservative", "Moderate", "Aggressive"].map(x => `<option ${selected(x, h.demandPreset)}>${x}</option>`).join("")}</select></div>
    <div class="field"><label>Supply</label><select data-path="hss.supplyScenario"><option ${selected(h.supplyScenario, "Match Demand")}>Match Demand</option><option ${selected(h.supplyScenario, "Cannot Meet Demand")}>Cannot Meet Demand</option></select></div></fieldset>`;

  return `<fieldset><legend>Health Systems Strengthening</legend>${mode}
    <fieldset><legend>Demand</legend>${toggle("Employ CHVs", "hss.employCHV", h.employCHV)}
      ${h.employCHV ? `${toggle("Increase 4+ ANC visits", "hss.increaseANC", h.increaseANC)}
        ${h.increaseANC ? field("Expected 4+ ANC rate (%)", "hss.pANCPercent", h.pANCPercent, "number", 'min="0" max="100"') : ""}
        ${toggle("Increase L4/5 live births", "hss.increaseL45Delivery", h.increaseL45Delivery)}
        ${h.increaseL45Delivery ? `${field("Expected L4/5 deliveries (%)", "hss.pL45Percent", h.pL45Percent, "number", 'min="0" max="100"')}${field("Memory decay (months)", "hss.memoryDecayMonths", h.memoryDecayMonths, "number", 'min="1" max="36"')}
          <div class="field"><label>CHV memory model</label><select data-path="hss.chvMemoryModel">${["Logistic Decay", "Always Forget", "Always Remember"].map(x => `<option ${selected(x, h.chvMemoryModel)}>${x}</option>`).join("")}</select></div>` : ""}` : ""}</fieldset>
    <fieldset><legend>Facilities</legend>${toggle("Upgrade L4/5 facilities", "hss.upgradeFacilities", h.upgradeFacilities)}
      ${h.upgradeFacilities ? `${toggle("Improve worker performance", "hss.improvePerformance", h.improvePerformance)}${h.improvePerformance ? field("Performance (%)", "hss.performancePercent", h.performancePercent, "number", 'min="0" max="100"') : ""}
        ${toggle("Increase capacity", "hss.increaseCapacity", h.increaseCapacity)}${h.increaseCapacity ? field("Capacity increase (%)", "hss.capacityPercent", h.capacityPercent, "number", 'min="0" max="100"') : ""}
        ${toggle("Increase skilled labor", "hss.increaseLabor", h.increaseLabor)}${h.increaseLabor ? field("Labor (%)", "hss.laborPercent", h.laborPercent, "number", 'min="0" max="100"') : ""}
        ${toggle("Increase equipment", "hss.increaseEquipment", h.increaseEquipment)}${h.increaseEquipment ? field("Equipment (%)", "hss.equipmentPercent", h.equipmentPercent, "number", 'min="0" max="100"') : ""}` : ""}</fieldset>
    <fieldset><legend>Rescue network</legend>${toggle("Upgrade rescue network", "hss.upgradeRescueNetwork", h.upgradeRescueNetwork)}
      ${h.upgradeRescueNetwork ? `${toggle("Improve referral capacity", "hss.improveReferralCapacity", h.improveReferralCapacity)}${h.improveReferralCapacity ? `${field("Free referral coverage (%)", "hss.referralPercent", h.referralPercent, "number", 'min="0" max="100"')}${toggle("Emergency transfer", "hss.enableEmergencyTransfer", h.enableEmergencyTransfer)}${h.enableEmergencyTransfer ? field("Emergency transfers supported (%)", "hss.transferPercent", h.transferPercent, "number", 'min="0" max="100"') : ""}` : ""}` : ""}</fieldset>
  </fieldset>`;
}

function render() {
  const counties = (bootstrap.countyOptions || []).map(county => `<option ${selected(county, ui.county)}>${esc(county)}</option>`).join("");
  const result = bootstrap.results;
  app.innerHTML = `<fieldset><legend>Location</legend><div class="field"><label>County</label><select data-path="county">${counties}</select></div>${renderCountyDefaults()}</fieldset>
    ${renderHSS()}
    <fieldset><legend>Treatment interventions</legend>${treatment("pphBundle", "PPH bundle")}${treatment("ivIron", "IV iron")}${treatment("magnesiumSulfate", "Magnesium sulfate")}${treatment("antibiotics", "Antibiotics")}${treatment("oxytocin", "Oxytocin")}</fieldset>
    <fieldset><legend>Diagnosis</legend>${toggle("AI portable ultrasound", "diagnosis.ultrasound.enabled", ui.diagnosis.ultrasound.enabled)}${ui.diagnosis.ultrasound.enabled ? `${field("Ultrasound sensitivity", "diagnosis.ultrasound.sensitivity", ui.diagnosis.ultrasound.sensitivity, "number", 'min="0" max="1" step="0.05"')}${field("Ultrasound specificity", "diagnosis.ultrasound.specificity", ui.diagnosis.ultrasound.specificity, "number", 'min="0" max="1" step="0.05"')}` : ""}
      ${toggle("Intrapartum sensors", "diagnosis.intrapartumSensor.enabled", ui.diagnosis.intrapartumSensor.enabled)}${ui.diagnosis.intrapartumSensor.enabled ? `${toggle("Apply AI algorithms", "diagnosis.intrapartumSensor.aiEnabled", ui.diagnosis.intrapartumSensor.aiEnabled)}${ui.diagnosis.intrapartumSensor.aiEnabled ? `${field("Sensor sensitivity", "diagnosis.intrapartumSensor.sensitivity", ui.diagnosis.intrapartumSensor.sensitivity, "number", 'min="0" max="1" step="0.05"')}${field("Sensor specificity", "diagnosis.intrapartumSensor.specificity", ui.diagnosis.intrapartumSensor.specificity, "number", 'min="0" max="1" step="0.05"')}` : ""}` : ""}</fieldset>
    <fieldset><legend>MOMISH</legend><div class="field"><label>Facility-delivery context</label><select data-path="momish.facilityContext"><option value="follow" ${selected(ui.momish.facilityContext, "follow")}>Follow HSS settings</option><option value="low" ${selected(ui.momish.facilityContext, "low")}>Low</option><option value="high" ${selected(ui.momish.facilityContext, "high")}>High</option><option value="off" ${selected(ui.momish.facilityContext, "off")}>Off</option></select></div>
      ${fidelity("prompts", "PROMPTS")}${fidelity("mentors", "MENTORS")}${fidelity("pulse", "PULSE")}${fidelity("fqa", "FQA")}${fidelity("blood", "Blood tracking")}${fidelity("transfer", "Transfer & EMT")}
      <div class="field"><label>FQA amplification level</label><select data-path="momish.fqaPulseModifierLevel">${Object.keys(bootstrap.fqaPulseModifierOptions || { Medium: 0.2 }).map(x => `<option ${selected(x, ui.momish.fqaPulseModifierLevel)}>${esc(x)}</option>`).join("")}</select></div>
      <div class="field"><label>PULSE boost level</label><select data-path="momish.pulseBoostLevel">${["Current", "Moderate", "High"].map(x => `<option ${selected(x, ui.momish.pulseBoostLevel)}>${x}</option>`).join("")}</select></div></fieldset>
    <fieldset><legend>Model settings</legend>${field("Implementation years", "model.implementationYears", ui.model.implementationYears, "number", 'min="3" max="6"')}${field("Maintenance years", "model.maintenanceYears", ui.model.maintenanceYears, "number", 'min="0" max="3"')}${toggle("Run multiple scenarios", "model.multipleRun", ui.model.multipleRun)}${ui.model.multipleRun ? field("Number of runs", "model.numberOfRuns", ui.model.numberOfRuns, "number", 'min="1" max="300"') : ""}</fieldset>
    <button id="apply">Apply inputs</button><button id="run" class="primary">Run model</button><button id="reset">Reset</button>
    <h2>Status</h2><div id="status" class="status">${bootstrap.error ? `ERROR: ${esc(bootstrap.error.message)}` : result ? `Finished. JSON saved to:\n${esc(result.savedFile)}` : "Ready."}</div>
    ${resultMarkup(result)}`;

  document.querySelectorAll("[data-path]").forEach(element => element.addEventListener("change", () => {
    const path = element.dataset.path.split(".");
    let target = ui;
    path.slice(0, -1).forEach(key => target = target[key]);
    target[path.at(-1)] = element.type === "checkbox" ? element.checked : element.type === "number" ? Number(element.value) : element.value;
    if (element.dataset.path === "county") applyCountyDefaults(element.value);
    if (element.dataset.path === "momish.fqaPulseModifierLevel") ui.momish.fqaPulseModifier = bootstrap.fqaPulseModifierOptions[element.value];
    render();
  }));
  document.getElementById("apply").onclick = () => emit("stateChanged");
  document.getElementById("run").onclick = event => { event.target.disabled = true; document.getElementById("status").textContent = "Running model…"; emit("runModel"); };
  document.getElementById("reset").onclick = () => { ui = null; emit("reset", false); };
  document.getElementById("outcome-category")?.addEventListener("change", event => { selectedOutcomeCategoryId = event.target.value; selectedOutcomeId = null; render(); });
  document.getElementById("outcome")?.addEventListener("change", event => { selectedOutcomeId = event.target.value; render(); });
  renderPlots();
  resize();
}

addEventListener("message", event => {
  if (event.data?.type !== "streamlit:render") return;
  clearInterval(readyTimer);
  try {
    bootstrap = event.data.args.bootstrap;
    if (!bootstrap?.state) throw new Error("Streamlit did not provide bootstrap.state");
    defaultConfig = clone(bootstrap.state);
    if (!ui) ui = initialUI(defaultConfig);
    render();
  } catch (error) {
    console.error("SDR frontend render failed", error);
    app.innerHTML = `<div class="status">Frontend error: ${esc(error.message)}</div>`;
    resize();
  }
});

// The iframe can finish loading just before Streamlit registers its message
// listener. Retry the ready handshake until the first render message arrives.
function announceReady() {
  parent.postMessage({
    isStreamlitMessage: true,
    type: "streamlit:componentReady",
    apiVersion: 1,
  }, "*");
}

const readyTimer = setInterval(announceReady, 250);
announceReady();
