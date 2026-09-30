import unittest
from unittest.mock import Mock, patch

import numpy as np
import optuna
import pandas as pd

from Calibration.phase1_anc.calibrate import (
    _calculate_metrics,
    run_phase1_once,
    stop_when_target_reached,
)
from Calibration.phase1_anc.config import COUNTY_CONFIG


class Phase1AncCalibrationTests(unittest.TestCase):
    @patch("Calibration.phase1_anc.calibrate.f_ANC_LB_effect_vectorized")
    @patch("Calibration.phase1_anc.calibrate.parameter_loader.reset_inputs")
    def test_adapter_calls_production_anc_with_baseline_flags(
        self, reset_inputs, production_anc
    ):
        reset_inputs.return_value = {"LB_Track": np.array([[1, 1, 1, 1]])}
        production_anc.return_value = (
            pd.DataFrame(
                {
                    "i_loc": [0, 1, 2, 3],
                    "i_ANC": [1, 1, 0, 0],
                }
            ),
            None,
            None,
            None,
            None,
        )

        metrics = run_phase1_once({}, seed=123)

        production_anc.assert_called_once()
        call = production_anc.call_args.kwargs
        self.assertEqual(call["flags"]["flag_ANC"], 0)
        self.assertEqual(call["flags"]["flag_PROMPTS"], 0)
        self.assertEqual(call["flags"]["flag_us"], 0)
        self.assertEqual(metrics["home_all"], 0.25)
        self.assertEqual(metrics["l45_all"], 0.5)

    def test_location_metrics_reconcile(self):
        locations = np.array([0, 1, 2, 3, 2, 3])
        anc = np.array([1, 1, 1, 1, 0, 0])

        metrics = _calculate_metrics(locations, anc)

        self.assertAlmostEqual(
            metrics["home_all"]
            + metrics["l23_all"]
            + metrics["l4_all"]
            + metrics["l5_all"],
            1.0,
        )
        self.assertEqual(metrics["anc_among_home"], 1.0)
        self.assertEqual(metrics["anc_among_l23"], 1.0)
        self.assertEqual(metrics["anc_among_l4"], 0.5)
        self.assertEqual(metrics["anc_among_l5"], 0.5)
        self.assertEqual(metrics["anc_among_l45"], 0.5)

    def test_mombasa_l5_is_not_an_objective(self):
        objective_metrics = COUNTY_CONFIG["mombasa"]["objective_targets"]
        diagnostic_metrics = COUNTY_CONFIG["mombasa"]["diagnostic_targets"]

        self.assertEqual(
            objective_metrics,
            {
                "home_all": "home_all_target",
                "l23_all": "l23_all_target",
                "l4_all": "l4_all_target",
            },
        )
        self.assertNotIn("l5_all", objective_metrics)
        self.assertEqual(diagnostic_metrics["l5_all"], "l5_all_target")

    def test_optimizer_stops_below_rmse_threshold(self):
        study = Mock()
        trial = Mock(
            state=optuna.trial.TrialState.COMPLETE,
            value=0.49,
        )

        stop_when_target_reached(study, trial)

        study.stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
