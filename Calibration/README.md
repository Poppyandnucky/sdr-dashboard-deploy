# Model calibration

This directory contains calibration workflows that use the same model code as
the dashboard. Calibration logic belongs here; maternal pathway logic remains
in the shared model modules at the repository root.

Each calibration phase should have:

- one shared implementation used for every county;
- county-specific settings that describe targets, fitted parameters, bounds,
  and fixed assumptions;
- timestamped run outputs containing fitted values and diagnostics; and
- a short methodology document explaining the objective and target units.

Existing historical notebooks should be retained until the shared workflow has
reproduced their results. They can then be moved into an archive rather than
remaining as separate executable implementations.

The initial pilot is [`phase1_anc`](phase1_anc/README.md).
