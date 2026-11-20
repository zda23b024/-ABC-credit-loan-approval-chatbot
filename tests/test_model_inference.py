"""End-to-end checks for the saved bundle, helper, and chatbot API."""
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from credit_model_runtime import DEFAULT_MODEL_PATH, load_model_bundle, predict_applicants
from chatbot import app as chatbot_app
from chatbot.app import ApplicationAnswers, build_model_row
from train_credit_model import make_features

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / 'ABC_Credit' / 'results_v6_local' / 'abc_credit_inference.py'
SPEC = importlib.util.spec_from_file_location('abc_credit_inference_helper', HELPER_PATH)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


class ModelInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_audit_path = ROOT / 'tests' / '_audit_smoke_test.jsonl'
        cls.test_audit_path.unlink(missing_ok=True)
        cls.original_audit_path = chatbot_app.AUDIT_PATH
        chatbot_app.AUDIT_PATH = cls.test_audit_path
        cls.client = TestClient(chatbot_app.app)
        cls.bundle = load_model_bundle(DEFAULT_MODEL_PATH)
        cls.base_answers = ApplicationAnswers(
            age=32, employment_type=chatbot_app.CATALOG['Employment_Type'][0],
            monthly_net_income=42000,
            vehicle_type=chatbot_app.CATALOG['Product_Code'][0],
            loan_amount=110000, vehicle_price=150000,
            has_active_loan=False, consent_to_log=True,
        )
        cls.row, cls.derived = build_model_row(cls.base_answers)

    @classmethod
    def tearDownClass(cls):
        chatbot_app.AUDIT_PATH = cls.original_audit_path
        cls.test_audit_path.unlink(missing_ok=True)

    def test_saved_bundle_loads_with_expected_threshold_and_features(self):
        self.assertTrue(DEFAULT_MODEL_PATH.is_file())
        self.assertEqual(self.bundle['model_name'], 'full_smooth_w1')
        self.assertEqual(self.bundle['positive_class'], 'DECLINED')
        self.assertAlmostEqual(self.bundle['threshold_declined'], 0.13995426939656921)
        self.assertEqual(self.bundle['feature_set'], 'full')

    def test_seven_answers_map_to_model_and_unasked_features_stay_missing(self):
        expected_features = [
            'Qualifications', 'Employment_Type', 'Age', 'Product_Code', 'Loan_Amount',
            'LTV', 'Resident_Type', 'Net_salary', 'Final_Tier', 'Make_Code',
            'Model_Description', 'Model_Variant', 'PAST_LOANS_ACTIVE',
        ]
        self.assertEqual(list(self.row), expected_features)
        self.assertEqual(self.row['Employment_Type'], self.base_answers.employment_type)
        self.assertEqual(self.row['Age'], 32)
        self.assertEqual(self.row['Product_Code'], self.base_answers.vehicle_type)
        self.assertEqual(self.row['Net_salary'], 42000)
        self.assertEqual(self.row['Loan_Amount'], 110000)
        self.assertEqual(self.row['PAST_LOANS_ACTIVE'], 'NO_PAST_LOANS')
        self.assertAlmostEqual(self.row['LTV'], 100 * 110000 / 150000)
        self.assertEqual(self.derived['Vehicle_Price'], 150000)
        for feature in (
            'Qualifications', 'Resident_Type', 'Final_Tier', 'Make_Code',
            'Model_Description', 'Model_Variant',
        ):
            self.assertIsNone(self.row[feature])

        transformed = make_features([self.row], expected_features)
        encoded = self.bundle['model'].named_steps['prep'].transform(transformed)
        encoded_values = encoded.toarray() if hasattr(encoded, 'toarray') else np.asarray(encoded)
        self.assertEqual(encoded_values.shape[0], 1)
        self.assertTrue(np.isfinite(encoded_values).all())

        catalog = self.client.get('/api/catalog').json()
        self.assertEqual(set(catalog), {'employment_types', 'vehicle_types'})
        self.assertEqual(
            {item['value'] for item in catalog['employment_types']},
            set(chatbot_app.CATALOG['Employment_Type']),
        )
        self.assertEqual(
            {item['value'] for item in catalog['vehicle_types']},
            set(chatbot_app.CATALOG['Product_Code']),
        )

    def test_shared_preprocessing_calibration_and_threshold_are_consistent(self):
        direct = predict_applicants([self.row], bundle=self.bundle).iloc[0]
        helper = HELPER.predict_applicants([self.row]).iloc[0]
        repeated = predict_applicants([self.row], bundle=self.bundle).iloc[0]
        self.assertAlmostEqual(direct['decline_probability'], helper['decline_probability'], places=12)
        self.assertAlmostEqual(direct['decline_probability'], repeated['decline_probability'], places=12)
        self.assertAlmostEqual(direct['approval_probability'] + direct['decline_probability'], 1.0, places=12)
        expected = 'DECLINED' if direct['decline_probability'] >= self.bundle['threshold_declined'] else 'APPROVED'
        self.assertEqual(direct['Decision'], expected)
        self.assertEqual(direct['Decision'], helper['Decision'])

    def test_complete_chatbot_api_prediction_validation_and_audit(self):
        self.test_audit_path.unlink(missing_ok=True)
        health = self.client.get('/api/health')
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()['model_name'], self.bundle['model_name'])
        self.assertAlmostEqual(health.json()['threshold'], self.bundle['threshold_declined'])

        payload = self.base_answers.model_dump()
        response = self.client.post('/api/decision', json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        expected = predict_applicants([self.row], bundle=self.bundle).iloc[0]
        self.assertAlmostEqual(result['decline_probability'], expected['decline_probability'], places=12)
        self.assertAlmostEqual(result['approval_probability'] + result['decline_probability'], 1.0, places=12)
        self.assertEqual(result['decision'], expected['Decision'])
        self.assertAlmostEqual(result['threshold_declined'], self.bundle['threshold_declined'])
        self.assertGreaterEqual(result['latency_ms'], 0)
        self.assertTrue(any('were not asked and were imputed' in warning for warning in result['warnings']))

        for invalid in (
            dict(payload, employment_type='FABRICATED'),
            dict(payload, vehicle_type='FABRICATED'),
            dict(payload, age=61),
            dict(payload, loan_amount=200000, vehicle_price=50000),
            dict(payload, qualification='GRAD'),
        ):
            self.assertEqual(self.client.post('/api/decision', json=invalid).status_code, 422)

        records = [json.loads(line) for line in self.test_audit_path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(set(record['answers']), {
            'age', 'employment_type', 'monthly_net_income', 'vehicle_type',
            'loan_amount', 'vehicle_price', 'has_active_loan',
        })
        self.assertEqual(list(record['model_features']), list(self.bundle['features']))
        self.assertIsNone(record['model_features']['Qualifications'])
        self.assertAlmostEqual(record['model_features']['LTV'], 100 * 110000 / 150000)

    def test_api_outcomes_and_ui_expose_seven_question_flow(self):
        self.test_audit_path.unlink(missing_ok=True)
        scenarios = [
            ('APPROVED', {
                'age': 18, 'employment_type': 'AGR', 'monthly_net_income': 0,
                'vehicle_type': 'EB', 'loan_amount': 40000, 'vehicle_price': 65000,
                'has_active_loan': False, 'consent_to_log': True,
            }),
            ('DECLINED', {
                'age': 18, 'employment_type': 'AGR', 'monthly_net_income': 100000,
                'vehicle_type': 'EB', 'loan_amount': 250000, 'vehicle_price': 300000,
                'has_active_loan': True, 'consent_to_log': True,
            }),
        ]
        ui_source = (ROOT / 'chatbot' / 'static' / 'app.js').read_text(encoding='utf-8')
        index_source = (ROOT / 'chatbot' / 'static' / 'index.html').read_text(encoding='utf-8')
        self.assertEqual(ui_source.count("{key:'"), 7)
        self.assertIn("textContent='Review your 7 answers'", ui_source)
        self.assertIn("I’ll ask 7 quick questions", ui_source)
        self.assertIn('Calculated LTV:', ui_source)
        self.assertNotIn('model_description', ui_source)
        self.assertNotIn('qualification', ui_source)
        self.assertNotIn('name="phone"', index_source.lower())
        self.assertNotIn('name="email"', index_source.lower())

        for expected_decision, payload in scenarios:
            with self.subTest(decision=expected_decision):
                response = self.client.post('/api/decision', json=payload)
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                row, _ = build_model_row(ApplicationAnswers(**payload))
                direct = predict_applicants([row], bundle=self.bundle).iloc[0]
                self.assertEqual(result['decision'], expected_decision)
                self.assertEqual(result['decision'], direct['Decision'])
                self.assertAlmostEqual(result['decline_probability'], direct['decline_probability'], places=12)
                self.assertEqual(list(row), list(self.bundle['features']))
                self.assertAlmostEqual(result['approval_probability'] + result['decline_probability'], 1.0)
                self.assertIn('historical ' + expected_decision + ' label', result['message'])

        records = [json.loads(line) for line in self.test_audit_path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual([record['score']['decision'] for record in records], ['APPROVED', 'DECLINED'])
        for record in records:
            serialized = json.dumps(record).lower()
            for forbidden in ('phone_number', 'email', 'full_name', 'pincode', 'ip_address', 'account_id'):
                self.assertNotIn(forbidden, serialized)


if __name__ == '__main__':
    unittest.main()
