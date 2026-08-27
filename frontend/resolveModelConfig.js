const DEMAND = {
  Conservative: { pANC: 70, pL45: 53 },
  Moderate: { pANC: 80, pL45: 68 },
  Aggressive: { pANC: 90, pL45: 90 },
};

const CAPACITY_MATCH = { Conservative: 25, Moderate: 50, Aggressive: 85 };
const CAPACITY_MISMATCH = { Conservative: 12.5, Moderate: 25, Aggressive: 42.5 };
const OVERRIDE = { Current: null, Moderate: 0.5, High: 0.95 };
const PROMPTS_RR = { Current: 1.02, Moderate: 1.18, High: 1.35 };
const BLOOD = { Current: 0.25, Moderate: 0.5, High: 0.95 };
const PULSE_BOOST = { Current: 0.25, Moderate: 0.5, High: 0.75 };

const clone = value => JSON.parse(JSON.stringify(value));
const clamp = (value, min, max) => Math.min(max, Math.max(min, Number(value)));
const proportion = value => clamp(value, 0, 100) / 100;

function currentOrOverride(level, currentValue) {
  return OVERRIDE[level] ?? clamp(currentValue ?? 0, 0, 1);
}

function resetHSS(config, base) {
  Object.assign(config.flags, {
    flag_SDR: 0, flag_CHV: 0, flag_ANC: 0, flag_LB: 0,
    flag_performance: 0, flag_capacity: 0, flag_labor: 0,
    flag_equipment: 0, flag_refer_voucher: 0, flag_transfer_capacity: 0,
  });
  Object.assign(config.HSS, {
    P_ANC: base.HSS.P_ANC, P_L45: base.HSS.P_L45,
    knowledge: base.HSS.knowledge, capacity_added: 0,
    labor_ratio: 0, sensor_ratio: 0, P_refer: 0,
    transfer_capacity_target: 0, tau_decay: 6,
    CHV_memory: "Always Forget",
  });
}

function applyPreset(config, base, demandName, supplyName) {
  resetHSS(config, base);
  const demand = DEMAND[demandName] || DEMAND.Conservative;
  const match = supplyName === "Match Demand";
  Object.assign(config.flags, {
    flag_SDR: 1, flag_CHV: 1, flag_ANC: 1, flag_LB: 1,
    flag_performance: 1, flag_capacity: 1, flag_labor: 1,
    flag_equipment: 1, flag_refer_voucher: 1, flag_transfer_capacity: 1,
  });
  Object.assign(config.HSS, {
    P_ANC: demand.pANC / 100,
    P_L45: demand.pL45 / 100,
    knowledge: match ? 1 : 0.75,
    capacity_added: (match ? CAPACITY_MATCH[demandName] : CAPACITY_MISMATCH[demandName]) / 100,
    labor_ratio: match ? 1 : 0.5,
    sensor_ratio: match ? 1 : 0.5,
    P_refer: match ? 1 : 0.5,
    transfer_capacity_target: match ? 100 : 50,
    tau_decay: 6,
    CHV_memory: "Logistic Decay",
  });
}

function resolveManualHSS(config, base, hss) {
  resetHSS(config, base);
  if (hss.employCHV) {
    config.flags.flag_SDR = 1;
    config.flags.flag_CHV = 1;
    if (hss.increaseANC) {
      config.flags.flag_ANC = 1;
      config.HSS.P_ANC = proportion(hss.pANCPercent);
    }
    if (hss.increaseL45Delivery) {
      config.flags.flag_LB = 1;
      // The county-specific ANC→L4/5 minimum will be added to bootstrap later.
      config.HSS.P_L45 = proportion(hss.pL45Percent);
      config.HSS.tau_decay = clamp(hss.memoryDecayMonths, 1, 36);
      config.HSS.CHV_memory = hss.chvMemoryModel;
    }
  }
  if (hss.upgradeFacilities) {
    config.flags.flag_SDR = 1;
    if (hss.improvePerformance) {
      config.flags.flag_performance = 1;
      config.HSS.knowledge = proportion(hss.performancePercent);
    }
    if (hss.increaseCapacity) {
      config.flags.flag_capacity = 1;
      config.HSS.capacity_added = proportion(hss.capacityPercent);
    }
    if (hss.increaseLabor) {
      config.flags.flag_labor = 1;
      config.HSS.labor_ratio = proportion(hss.laborPercent);
    }
    if (hss.increaseEquipment) {
      config.flags.flag_equipment = 1;
      config.HSS.sensor_ratio = proportion(hss.equipmentPercent);
    }
  }
  if (hss.upgradeRescueNetwork) {
    config.flags.flag_SDR = 1;
    if (hss.improveReferralCapacity) {
      config.flags.flag_refer_voucher = 1;
      config.HSS.P_refer = proportion(hss.referralPercent);
      if (hss.enableEmergencyTransfer) {
        config.flags.flag_transfer_capacity = 1;
        config.HSS.transfer_capacity_target = clamp(hss.transferPercent, 0, 100);
      }
    }
  }
}

function resolveTreatments(config, treatments) {
  const map = {
    pphBundle: ["flag_pph_bundle", "pph_bundle"],
    ivIron: ["flag_iv_iron", "iv_iron"],
    magnesiumSulfate: ["flag_MgSO4", "MgSO4"],
    antibiotics: ["flag_antibiotics", "antibiotics"],
    oxytocin: ["flag_oxytocin", "oxytocin"],
  };
  for (const [name, [flag, parameter]] of Object.entries(map)) {
    const item = treatments[name];
    config.flags[flag] = item.enabled ? 1 : 0;
    config.S[parameter] = item.enabled ? proportion(item.coveragePercent) : 0;
  }
}

function resolveDiagnostics(config, diagnosis) {
  const us = diagnosis.ultrasound;
  config.flags.flag_us = us.enabled ? 1 : 0;
  config.S.US = us.enabled ? 1 : 0;
  if (us.enabled) {
    config.E.sens_us = clamp(us.sensitivity, 0, 1);
    config.E.spec_us = clamp(us.specificity, 0, 1);
  }
  const sensor = diagnosis.intrapartumSensor;
  config.flags.flag_intrasensor = sensor.enabled ? 1 : 0;
  config.flags.flag_sensor_ai = sensor.enabled && sensor.aiEnabled ? 1 : 0;
  if (sensor.enabled && sensor.aiEnabled) {
    config.E.sens_sensor = clamp(sensor.sensitivity, 0, 1);
    config.E.spec_sensor = clamp(sensor.specificity, 0, 1);
  }
}

function resolveMomish(config, base, momish) {
  const programs = {
    prompts: ["flag_PROMPTS", "prompts_implementation_index"],
    mentors: ["flag_MENTOR", "mentor_implementation_index"],
    pulse: ["flag_pulse", "pulse_implementation_index"],
    fqa: ["flag_fqa", "fqa_implementation_index"],
    transfer: ["flag_transfer_delay", "referral_implementation_index"],
  };
  for (const [name, [flag, parameter]] of Object.entries(programs)) {
    const item = momish[name];
    config.flags[flag] = item.enabled ? 1 : 0;
    config.HSS[parameter] = item.enabled
      ? currentOrOverride(item.level, base.HSS[parameter]) : 0;
  }
  config.HSS.prompts_rr_anc4p = momish.prompts.enabled
    ? PROMPTS_RR[momish.prompts.level] : 1;
  config.flags.flag_blood = momish.blood.enabled ? 1 : 0;
  config.flags.flag_blood_tracking = config.flags.flag_blood;
  config.HSS.blood_adoption = momish.blood.enabled ? BLOOD[momish.blood.level] : 0;
  config.HSS.fqa_pulse_modifier_level = momish.fqaPulseModifierLevel;
  config.HSS.fqa_pulse_modifier = momish.fqaPulseModifier;
  config.HSS.pulse_implementation_boost_level = momish.pulseBoostLevel;
  config.HSS.pulse_implementation_boost = PULSE_BOOST[momish.pulseBoostLevel];
}

export function resolveModelConfig(ui, defaultConfig) {
  const config = clone(defaultConfig);
  const base = clone(defaultConfig);
  config.county = ui.county;

  if (ui.hss.mode === "preset") {
    applyPreset(config, base, ui.hss.demandPreset, ui.hss.supplyScenario);
  } else {
    resolveManualHSS(config, base, ui.hss);
  }

  resolveTreatments(config, ui.treatments);
  resolveDiagnostics(config, ui.diagnosis);
  resolveMomish(config, base, ui.momish);

  if (ui.momish.facilityContext === "off") resetHSS(config, base);
  if (ui.momish.facilityContext === "low") applyPreset(config, base, "Conservative", "Match Demand");
  if (ui.momish.facilityContext === "high") applyPreset(config, base, "Aggressive", "Match Demand");

  const implementationYears = Math.round(clamp(ui.model.implementationYears, 3, 6));
  const maintenanceYears = Math.round(clamp(ui.model.maintenanceYears, 0, 3));
  config.model = {
    imple_time: implementationYears,
    main_time: maintenanceYears,
    int_period: implementationYears * 12,
    n_months: (implementationYears + maintenanceYears) * 12,
    multiple_run: Boolean(ui.model.multipleRun),
    n_runs: ui.model.multipleRun ? Math.round(clamp(ui.model.numberOfRuns, 1, 300)) : 1,
  };

  return { county: config.county, flags: config.flags, E: config.E, S: config.S, HSS: config.HSS, model: config.model };
}
