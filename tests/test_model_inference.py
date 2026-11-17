"""End-to-end checks for the saved bundle, helper, and chatbot API."""
from __future__ import annotations

import importlib.util
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
        cls.vehicle = chatbot_app.CATALOG['vehicles'][0]
        cls.base_answers = ApplicationAnswers(
            age_band='18-25', qualification=chatbot_app.CATALOG['Qualifications'][0],
            employment_type=chatbot_app.CATALOG['Employment_Type'][0], income_band='0',
            residence='O', area_tier=chatbot_app.CATALOG['Final_Tier'][0],
            vehicle_type=cls.vehicle['product_code'], model_description=cls.vehicle['description'],
            loan_band='under_50000', vehicle_price_band='under_75000',
            has_active_loan=False, consent_to_log=True,
        )
        cls.row = {
            'Qualifications': chatbot_app.CATALOG['Qualifications'][0],
            'Employment_Type': chatbot_app.CATALOG['Employment_Type'][0],
            'Age': 32,
            'Product_Code': cls.vehicle['product_code'],
            'Loan_Amount': 110000,
            'LTV': 100 * 110000 / 150000,
            'Resident_Type': 'O',
            'Net_salary': 42000,
            'Final_Tier': chatbot_app.CATALOG['Final_Tier'][0],
            'Make_Code': cls.vehicle['make_code'],
            'Model_Description': cls.vehicle['description'],
            'Model_Variant': cls.vehicle['model_variant'],
            'PAST_LOANS_ACTIVE': 'NO_PAST_LOANS',
        }

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

    def test_all_11_question_mappings_and_all_catalog_choices(self):
        expected_features = list(self.bundle['features'])
        self.assertEqual(expected_features, [
            'Qualifications', 'Employment_Type', 'Age', 'Product_Code', 'Loan_Amount',
            'LTV', 'Resident_Type', 'Net_salary', 'Final_Tier', 'Make_Code',
            'Model_Description', 'Model_Variant', 'PAST_LOANS_ACTIVE',
        ])

        cases = []
        # Q1 Age range -> Age. All range estimates are explicit representative values.
        cases.extend(({'age_band': band}, 'Age', value) for band, value in chatbot_app.AGE_REPRESENTATIVES.items())
        # Q2 Qualification -> Qualifications.
        cases.extend(({'qualification': value}, 'Qualifications', value)
                     for value in chatbot_app.CATALOG['Qualifications'])
        # Q3 Employment type -> Employment_Type.
        cases.extend(({'employment_type': value}, 'Employment_Type', value)
                     for value in chatbot_app.CATALOG['Employment_Type'])
        # Q4 Monthly net income -> Net_salary.
        cases.extend(({'income_band': band}, 'Net_salary', value)
                     for band, value in chatbot_app.INCOME_REPRESENTATIVES.items())
        # Q5 Residence -> Resident_Type; codes are defined in the assignment brief.
        cases.extend(({'residence': code}, 'Resident_Type', code)
                     for code in chatbot_app.CATALOG['Resident_Type'])
        # Q6 Broad area category -> Final_Tier.
        cases.extend(({'area_tier': value}, 'Final_Tier', value)
                     for value in chatbot_app.CATALOG['Final_Tier'])
        # Q7 Vehicle type -> Product_Code; Q8 model -> description, make, and variant.
        for vehicle in chatbot_app.CATALOG['vehicles']:
            cases.append(({'vehicle_type': vehicle['product_code'],
                           'model_description': vehicle['description']},
                          'Model_Description', vehicle['description']))
            row, _ = build_model_row(self.base_answers.model_copy(update={
                'vehicle_type': vehicle['product_code'],
                'model_description': vehicle['description'],
            }))
            self.assertEqual(row['Product_Code'], vehicle['product_code'])
            self.assertEqual(row['Make_Code'], vehicle['make_code'])
            self.assertEqual(row['Model_Variant'], vehicle['model_variant'])
        # Q9 Loan amount -> Loan_Amount.
        cases.extend(({'loan_band': band}, 'Loan_Amount', value)
                     for band, value in chatbot_app.LOAN_REPRESENTATIVES.items())
        # Q10 Vehicle price -> calculated LTV only, not a model input field.
        for band, value in chatbot_app.VEHICLE_PRICE_REPRESENTATIVES.items():
            row, derived = build_model_row(self.base_answers.model_copy(update={'vehicle_price_band': band}))
            self.assertEqual(derived['Vehicle_Price'], value)
            self.assertAlmostEqual(row['LTV'], 100 * row['Loan_Amount'] / value)
            cases.append(({'vehicle_price_band': band}, 'LTV', row['LTV']))
        # Q11 Active prior loan -> exact category values present in the training data.
        cases.extend(({'has_active_loan': active}, 'PAST_LOANS_ACTIVE', value)
                     for active, value in chatbot_app.ACTIVE_LOAN_VALUES.items())

        for updates, feature, expected in cases:
            with self.subTest(question_input=updates, feature=feature):
                safe_updates = dict(updates)
                if 'loan_band' in safe_updates:
                    safe_updates['vehicle_price_band'] = '175000_plus'
                answers = self.base_answers.model_copy(update=safe_updates)
                row, _ = build_model_row(answers)
                self.assertEqual(list(row), expected_features)
                self.assertEqual(row[feature], expected)
                transformed = make_features([row], expected_features)
                self.assertEqual(list(transformed.columns[:len(expected_features)]), expected_features)
                encoded = self.bundle['model'].named_steps['prep'].transform(transformed)
                encoded_values = encoded.toarray() if hasattr(encoded, 'toarray') else np.asarray(encoded)
                self.assertEqual(encoded_values.shape[0], 1)
                self.assertTrue(np.isfinite(encoded_values).all())

        # Exact typed numbers are passed through and LTV is calculated from the exact values.
        exact, derived = build_model_row(self.base_answers.model_copy(update={
            'age_band': None, 'age_value': 37,
            'income_band': None, 'monthly_net_income': 37890.5,
            'loan_band': None, 'loan_amount': 123456.78,
            'vehicle_price_band': None, 'vehicle_price': 160000,
        }))
        self.assertEqual(exact['Age'], 37)
        self.assertEqual(exact['Net_salary'], 37890.5)
        self.assertEqual(exact['Loan_Amount'], 123456.78)
        self.assertAlmostEqual(exact['LTV'], 100 * 123456.78 / 160000)
        self.assertEqual(derived['Vehicle_Price'], 160000)
        exact_features = make_features([exact], expected_features)
        self.assertAlmostEqual(exact_features.iloc[0]['estimated_vehicle_price'], 160000)

        # The API catalogs expose the same server-owned range representatives used above.
        catalog = self.client.get('/api/catalog').json()
        self.assertEqual(catalog['representative_values']['age_band'], chatbot_app.AGE_REPRESENTATIVES)
        self.assertEqual(catalog['representative_values']['income_band'], chatbot_app.INCOME_REPRESENTATIVES)
        self.assertEqual(catalog['representative_values']['loan_band'], chatbot_app.LOAN_REPRESENTATIVES)
        self.assertEqual(catalog['representative_values']['vehicle_price_band'], chatbot_app.VEHICLE_PRICE_REPRESENTATIVES)
        self.assertEqual({x['value'] for x in catalog['vehicle_types']}, set(chatbot_app.CATALOG['Product_Code']))
        self.assertEqual({x['value'] for x in catalog['residences']}, set(chatbot_app.CATALOG['Resident_Type']))

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

    def test_complete_chatbot_api_prediction_and_audit(self):
        self.test_audit_path.unlink(missing_ok=True)
        health = self.client.get('/api/health')
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()['model_name'], self.bundle['model_name'])
        self.assertAlmostEqual(health.json()['threshold'], self.bundle['threshold_declined'])

        payload = {
            'age_value': self.row['Age'],
            'qualification': self.row['Qualifications'],
            'employment_type': self.row['Employment_Type'],
            'monthly_net_income': self.row['Net_salary'],
            'residence': self.row['Resident_Type'],
            'area_tier': self.row['Final_Tier'],
            'vehicle_type': self.vehicle['product_code'],
            'model_description': self.vehicle['description'],
            'loan_amount': self.row['Loan_Amount'],
            'vehicle_price': 150000,
            'has_active_loan': False,
            'consent_to_log': True,
        }
        response = self.client.post('/api/decision', json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        expected = predict_applicants([self.row], bundle=self.bundle).iloc[0]
        self.assertAlmostEqual(result['decline_probability'], expected['decline_probability'], places=12)
        self.assertAlmostEqual(result['approval_probability'] + result['decline_probability'], 1.0, places=12)
        self.assertEqual(result['decision'], expected['Decision'])
        self.assertAlmostEqual(result['threshold_declined'], self.bundle['threshold_declined'])
        self.assertGreaterEqual(result['latency_ms'], 0)

        # Exercise range-option submission as sent by the browser as well as
        # exact typed amounts. Range-to-value explanations must name each value.
        ranged = dict(payload)
        ranged.pop('age_value')
        ranged.update({'age_band': '18-25'})
        ranged.pop('monthly_net_income')
        ranged.update({'income_band': 'under_15000'})
        ranged.pop('loan_amount')
        ranged.update({'loan_band': 'under_50000'})
        ranged.pop('vehicle_price')
        ranged.update({'vehicle_price_band': '75000_94999'})
        range_response = self.client.post('/api/decision', json=ranged)
        self.assertEqual(range_response.status_code, 200, range_response.text)
        range_result = range_response.json()
        self.assertIn('Age range represented as 22 years', ' '.join(range_result['warnings']))
        self.assertIn('Income range represented as ₹10,000 per month', ' '.join(range_result['warnings']))
        self.assertIn('Loan range represented as ₹40,000', ' '.join(range_result['warnings']))
        self.assertIn('Vehicle-price range represented as ₹85,000', ' '.join(range_result['warnings']))

        # Values outside catalog categories and invalid derived LTV are rejected.
        invalid_category = dict(payload, qualification='FABRICATED_CATEGORY')
        self.assertEqual(self.client.post('/api/decision', json=invalid_category).status_code, 422)
        invalid_ltv = dict(payload, loan_amount=200000, vehicle_price=50000)
        self.assertEqual(self.client.post('/api/decision', json=invalid_ltv).status_code, 422)

        self.assertTrue(chatbot_app.AUDIT_PATH.is_file())
        records = [__import__('json').loads(line) for line in chatbot_app.AUDIT_PATH.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(records), 2)
        record = records[0]
        self.assertTrue(all(x['event'] == 'model_decision' for x in records))
        self.assertEqual(list(record['model_features']), list(self.bundle['features']))
        self.assertEqual(record['model_features']['Resident_Type'], 'O')
        self.assertAlmostEqual(record['model_features']['LTV'], 100 * 110000 / 150000)

    def test_both_outcomes_display_mapping_validation_and_privacy(self):
        self.test_audit_path.unlink(missing_ok=True)
        vehicle = self.vehicle
        shared = {
            'age_band': '18-25', 'employment_type': 'SAL', 'income_band': '0',
            'residence': 'O', 'area_tier': '01 Megapolis (A)',
            'vehicle_type': vehicle['product_code'],
            'model_description': vehicle['description'],
            'has_active_loan': False, 'consent_to_log': True,
        }
        scenarios = [
            ('APPROVED', dict(shared, qualification='GRAD', loan_band='under_50000',
                              vehicle_price_band='under_75000')),
            ('DECLINED', dict(shared, qualification='SSC', loan_band='100000_124999',
                              vehicle_price_band='75000_94999')),
        ]
        ui_source = (ROOT / 'chatbot' / 'static' / 'app.js').read_text(encoding='utf-8')
        index_source = (ROOT / 'chatbot' / 'static' / 'index.html').read_text(encoding='utf-8')
        self.assertEqual(ui_source.count("{key:'"), 11)
        self.assertIn('Next step: ABC Credit can explain', ui_source)
        self.assertIn('Next step: ABC Credit will confirm', ui_source)
        self.assertIn('data.decision', ui_source)
        self.assertIn('data.decline_probability.toFixed(8)', ui_source)
        self.assertIn('threshold_declined', ui_source)
        self.assertNotIn('name="phone"', index_source.lower())
        self.assertNotIn('name="email"', index_source.lower())

        results = []
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
                self.assertAlmostEqual(result['threshold_declined'], self.bundle['threshold_declined'])
                self.assertEqual(list(row), list(self.bundle['features']))
                self.assertAlmostEqual(result['approval_probability'] + result['decline_probability'], 1.0)
                expected_message = 'historical ' + expected_decision + ' label'
                self.assertIn(expected_message, result['message'])
                results.append(result)

        # Reject malformed/missing values and fabricated categories without logging.
        invalid_payload = dict(scenarios[0][1], qualification='FABRICATED')
        self.assertEqual(self.client.post('/api/decision', json=invalid_payload).status_code, 422)
        invalid_payload = dict(scenarios[0][1], age_value=999, age_band=None)
        self.assertEqual(self.client.post('/api/decision', json=invalid_payload).status_code, 422)
        invalid_payload = dict(scenarios[0][1], loan_amount=200000, loan_band=None,
                               vehicle_price=50000, vehicle_price_band=None)
        self.assertEqual(self.client.post('/api/decision', json=invalid_payload).status_code, 422)

        records = [__import__('json').loads(line) for line in self.test_audit_path.read_text(encoding='utf-8').splitlines()]
        decisions = [r['score']['decision'] for r in records]
        self.assertIn('APPROVED', decisions)
        self.assertIn('DECLINED', decisions)
        for record in records:
            serialized = __import__('json').dumps(record).lower()
            for forbidden in ('phone_number', 'email', 'full_name', 'pincode', 'ip_address', 'account_id'):
                self.assertNotIn(forbidden, serialized)
            self.assertEqual(list(record['model_features']), list(self.bundle['features']))

        # The UI exposes a full answer review, per-question edits, backward
        # navigation, and clears dependent model inputs when their parent changes.
        self.assertIn('function showReview()', ui_source)
        self.assertIn("textContent='Review your 11 answers'", ui_source)
        self.assertIn('fields.forEach((field,index)', ui_source)
        self.assertIn('Edit answer ${index+1}', ui_source)
        self.assertIn('function navigateBack()', ui_source)
        self.assertIn("removeAnswer('model_description')", ui_source)
        self.assertIn("removeAnswer('vehicle_price_band')", ui_source)
        self.assertIn('Calculated LTV:', ui_source)
        self.assertIn("textContent='Confirm and get assessment'", ui_source)

        # Restart clears both the API payload and display/edit state.
        self.assertIn('function restart() {current=0;answers={};savedResponses={};busy=false;editResumeIndex=null;chat.replaceChildren();begin();}', ui_source)


if __name__ == '__main__':
    unittest.main()
