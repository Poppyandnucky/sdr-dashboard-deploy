import unittest

import numpy as np
import pandas as pd

from scenario_results import (
    build_mmr_datasets,
    maternal_deaths_by_cause,
    monthly_mmr_by_level,
    period_mmr_by_level,
)


def monthly_frame(deaths, births):
    return pd.DataFrame(
        {
            "Deaths": [np.asarray(deaths, dtype=float)],
            "Live Births Final": [np.asarray(births, dtype=float)],
        }
    )


class ScenarioResultsTests(unittest.TestCase):
    def test_monthly_mmr_retains_l23_l4_l5_and_excludes_home(self):
        frame = monthly_frame(
            deaths=[[99, 1, 2, 3], [99, 2, 3, 4]],
            births=[[100, 100, 100, 100], [100, 100, 100, 100]],
        )
        dataset = monthly_mmr_by_level(frame, n_months=2)

        self.assertEqual(dataset["rowCount"], 6)
        self.assertEqual(
            {row["levelId"] for row in dataset["rows"]},
            {"l23", "l4", "l5"},
        )
        self.assertNotIn("home", {row["levelId"] for row in dataset["rows"]})
        l23_month_1 = next(
            row for row in dataset["rows"]
            if row["levelId"] == "l23" and row["month"] == 1
        )
        self.assertEqual(l23_month_1["mmr"], 1000.0)

    def test_period_mmr_averages_ratios_calculated_per_run(self):
        frame = monthly_frame(
            deaths=[
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ],
            births=[
                [1, 100, 100, 100],
                [1, 100, 100, 100],
                [1, 200, 100, 100],
                [1, 200, 100, 100],
            ],
        )
        dataset = period_mmr_by_level(frame, n_months=2)
        l23 = next(row for row in dataset["rows"] if row["levelId"] == "l23")

        # Run ratios are 1000 and 500; their arithmetic mean is 750.
        self.assertEqual(l23["mmr"], 750.0)
        self.assertEqual(
            dataset["metadata"]["acrossRuns"],
            "arithmetic_mean_of_run_level_ratios",
        )

    def test_zero_denominator_becomes_json_null(self):
        frame = monthly_frame(
            deaths=[[0, 1, 1, 1]],
            births=[[100, 0, 100, 100]],
        )
        dataset = monthly_mmr_by_level(frame, n_months=1)
        l23 = next(row for row in dataset["rows"] if row["levelId"] == "l23")

        self.assertIsNone(l23["mmr"])
        self.assertIsNone(l23["lower95"])
        self.assertIsNone(l23["upper95"])

    def test_empirical_interval_method_is_available(self):
        frame = monthly_frame(
            deaths=[[0, 1, 0, 0], [0, 3, 0, 0]],
            births=[[100, 100, 100, 100], [100, 100, 100, 100]],
        )
        dataset = monthly_mmr_by_level(
            frame,
            n_months=1,
            ci_method="empirical_run_quantiles",
        )
        l23 = next(row for row in dataset["rows"] if row["levelId"] == "l23")

        self.assertEqual(l23["mmr"], 2000.0)
        self.assertEqual(l23["lower95"], 1050.0)
        self.assertEqual(l23["upper95"], 2950.0)

    def test_cause_proportions_pool_runs_and_include_zero_categories(self):
        people = pd.DataFrame(
            {
                "Run": [1, 1, 2, 2, 2],
                "death_cause": ["pph", "none", "pph", "sepsis", "none"],
            }
        )
        dataset = maternal_deaths_by_cause(people, number_of_runs=2)
        rows = {row["deathCauseId"]: row for row in dataset["rows"]}

        self.assertEqual(rows["pph"]["pooledCount"], 2)
        self.assertEqual(rows["pph"]["meanCountPerRun"], 1.0)
        self.assertEqual(rows["pph"]["proportionOfAttributedDeaths"], 0.666667)
        self.assertEqual(rows["aph"]["pooledCount"], 0)
        self.assertEqual(rows["aph"]["proportionOfAttributedDeaths"], 0.0)

    def test_build_mmr_datasets_returns_catalog_dataset_ids(self):
        frame = monthly_frame(
            deaths=[[0, 1, 2, 3]],
            births=[[100, 100, 100, 100]],
        )
        people = pd.DataFrame({"death_cause": ["pph", "none"]})
        datasets = build_mmr_datasets(frame, people, n_months=1, number_of_runs=1)

        self.assertEqual(
            [dataset["datasetId"] for dataset in datasets],
            ["monthlyMmrByLevel", "periodMmrByLevel", "maternalDeathsByCause"],
        )


if __name__ == "__main__":
    unittest.main()
