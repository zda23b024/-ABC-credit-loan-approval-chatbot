"""Tests for unseen CSV batch scoring against the saved inference path."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ABC_Credit.results_v6_local.predict_batch import predict_csv
from credit_model_runtime import DEFAULT_MODEL_PATH, load_model_bundle, predict_applicants
ROOT = Path(__file__).resolve().parents[1]


class BatchPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_model_bundle(DEFAULT_MODEL_PATH)
        cls.vehicle = {
            'Product_Code': 'MC', 'Make_Code': 'APACHE',
            'Model_Description': 'APACHE 200 4V 1CH - R MODE BSVI',
            'Model_Variant': 'RTR 200CC',
        }
        cls.rows = [
            {
                'ID': '000123', 'Qualifications': 'GRAD', 'Employment_Type': 'SAL',
                'Age': 22, **cls.vehicle, 'Loan_Amount': 40000,
                'LTV': 61.53846153846154, 'Resident_Type': 'O', 'Net_salary': 0,
                'Final_Tier': '01 Megapolis (A)', 'PAST_LOANS_ACTIVE': 'NO_PAST_LOANS',
                'Pincode': 'ignored', 'Gender': 'ignored',
            },
            {
                'ID': '000124', 'Qualifications': 'SSC', 'Employment_Type': 'SAL',
                'Age': 22, **cls.vehicle, 'Loan_Amount': 112500,
                'LTV': 132.35294117647058, 'Resident_Type': 'O', 'Net_salary': 0,
                'Final_Tier': '01 Megapolis (A)', 'PAST_LOANS_ACTIVE': 'NO_PAST_LOANS',
            },
            {
                'ID': '000125', 'Qualifications': None, 'Employment_Type': 'UNSEEN_JOB',
                'Age': None, **cls.vehicle, 'Loan_Amount': 70000,
                'LTV': None, 'Resident_Type': 'O', 'Net_salary': None,
                'Final_Tier': 'UNSEEN_TIER', 'PAST_LOANS_ACTIVE': None,
            },
        ]

    def test_cli_csv_outputs_preserve_ids_and_match_direct_inference(self):
        input_path = ROOT / 'tests' / '_batch_unseen.csv'
        output_path = ROOT / 'tests' / '_batch_predictions.csv'
        try:
            pd.DataFrame(self.rows).to_csv(input_path, index=False)

            output = predict_csv(input_path, output_path)
            reloaded = pd.read_csv(output_path, dtype={'ID': 'string'})
            self.assertEqual(reloaded['ID'].tolist(), ['000123', '000124', '000125'])
            self.assertEqual(output['Decision'].tolist()[0:2], ['APPROVED', 'DECLINED'])
            self.assertEqual(list(reloaded.columns), [
                'ID', 'decline_probability', 'approval_probability', 'Decision',
            ])
            self.assertEqual(len(reloaded), len(self.rows))
            self.assertTrue(np.isfinite(reloaded[['decline_probability', 'approval_probability']]).all().all())
            self.assertTrue(np.allclose(
                reloaded['decline_probability'] + reloaded['approval_probability'], 1.0,
                rtol=0, atol=1e-12,
            ))

            model_input = pd.DataFrame(self.rows).reindex(columns=self.bundle['features'])
            direct = predict_applicants(model_input, bundle=self.bundle)
            np.testing.assert_allclose(
                reloaded['decline_probability'], direct['decline_probability'], rtol=0, atol=1e-12,
            )
            np.testing.assert_allclose(
                reloaded['approval_probability'], direct['approval_probability'], rtol=0, atol=1e-12,
            )
            self.assertEqual(reloaded['Decision'].tolist(), direct['Decision'].tolist())
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_rejects_missing_model_columns_and_missing_id(self):
        input_path = ROOT / 'tests' / '_batch_invalid.csv'
        output_path = ROOT / 'tests' / '_batch_invalid_output.csv'
        try:
            frame = pd.DataFrame(self.rows).drop(columns=['LTV'])
            frame.to_csv(input_path, index=False)
            with self.assertRaisesRegex(ValueError, 'missing required model feature'):
                predict_csv(input_path, output_path)

            pd.DataFrame(self.rows).drop(columns=['ID']).to_csv(input_path, index=False)
            with self.assertRaisesRegex(ValueError, "must contain an 'ID'"):
                predict_csv(input_path, output_path)
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_same_input_and_output_path_is_rejected(self):
        path = ROOT / 'tests' / '_batch_same.csv'
        try:
            pd.DataFrame(self.rows).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'must be different'):
                predict_csv(path, path)
        finally:
            path.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
