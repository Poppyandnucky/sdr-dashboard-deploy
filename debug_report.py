"""Parameter debug report: search-friendly .txt snapshot of a dashboard run's
param dict at two points (right after parameter_loader, right before
run_model_dash), so parameters can be inspected with Ctrl+F instead of a
debugger."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


def _format_debug_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"np.ndarray shape={value.shape} value={value.tolist()}"

    if isinstance(value, (np.integer, np.floating)):
        return repr(value.item())

    if isinstance(value, pd.DataFrame):
        return f"DataFrame shape={value.shape}\n{value.to_string(index=False)}"

    if isinstance(value, pd.Series):
        return f"Series length={len(value)}\n{value.to_string()}"

    if isinstance(value, dict):
        try:
            return json.dumps(
                value,
                indent=2,
                sort_keys=True,
                default=lambda x: (
                    x.tolist() if isinstance(x, np.ndarray)
                    else x.item() if isinstance(x, (np.integer, np.floating))
                    else repr(x)
                ),
            )
        except TypeError:
            return repr(value)

    return repr(value)


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)
        except (TypeError, ValueError):
            return False

    try:
        result = left == right
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
        if isinstance(result, np.ndarray):
            return bool(np.all(result))
    except (TypeError, ValueError):
        pass

    return repr(left) == repr(right)


def build_parameter_debug_report(
    loader_param: dict[str, Any],
    final_param: dict[str, Any],
    *,
    county: str,
    scenario: str | None = None,
    flags: dict[str, Any] | None = None,
) -> str:
    """Create a searchable text report for one dashboard model run."""
    lines: list[str] = []

    lines.append("=" * 80)
    lines.append("SDR PARAMETER DEBUG REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"County: {county}")
    lines.append(f"Scenario: {scenario or 'Not specified'}")
    lines.append("")

    lines.append("=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Loader parameter count: {len(loader_param)}")
    lines.append(f"Final parameter count: {len(final_param)}")

    changed_keys, added_keys, removed_keys = [], [], []
    all_keys = sorted(set(loader_param) | set(final_param))
    for key in all_keys:
        if key not in loader_param:
            added_keys.append(key)
        elif key not in final_param:
            removed_keys.append(key)
        elif not _values_equal(loader_param[key], final_param[key]):
            changed_keys.append(key)

    lines.append(f"Changed after loader: {len(changed_keys)}")
    lines.append(f"Added after loader: {len(added_keys)}")
    lines.append(f"Removed after loader: {len(removed_keys)}")
    lines.append("")

    lines.append("=" * 80)
    lines.append("CHANGED PARAMETERS")
    lines.append("=" * 80)
    if changed_keys:
        for key in changed_keys:
            lines.append(f"\n[{key}]")
            lines.append("LOADER VALUE:")
            lines.append(_format_debug_value(loader_param[key]))
            lines.append("FINAL VALUE:")
            lines.append(_format_debug_value(final_param[key]))
    else:
        lines.append("No parameter values changed after loading.")

    lines.append("")
    lines.append("=" * 80)
    lines.append("ADDED PARAMETERS")
    lines.append("=" * 80)
    if added_keys:
        for key in added_keys:
            lines.append(f"\n[{key}]")
            lines.append(_format_debug_value(final_param[key]))
    else:
        lines.append("No parameters were added after loading.")

    lines.append("")
    lines.append("=" * 80)
    lines.append("ALL LOADER PARAMETERS")
    lines.append("=" * 80)
    for key in sorted(loader_param):
        lines.append(f"\n[{key}]")
        lines.append(_format_debug_value(loader_param[key]))

    lines.append("")
    lines.append("=" * 80)
    lines.append("ALL FINAL PARAMETERS PASSED TO MODEL")
    lines.append("=" * 80)
    for key in sorted(final_param):
        lines.append(f"\n[{key}]")
        lines.append(_format_debug_value(final_param[key]))

    if flags is not None:
        lines.append("")
        lines.append("=" * 80)
        lines.append("FLAGS")
        lines.append("=" * 80)
        for key in sorted(flags):
            lines.append(f"\n[{key}]")
            lines.append(_format_debug_value(flags[key]))

    return "\n".join(lines)
