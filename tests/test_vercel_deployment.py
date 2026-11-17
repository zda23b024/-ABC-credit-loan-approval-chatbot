"""Vercel-mode compatibility checks using FastAPI's local ASGI test client."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from chatbot import app as backend
from credit_model_runtime import load_model_bundle, predict_applicants
from chatbot.app import ApplicationAnswers, build_model_row
from app import app as vercel_app


class VercelCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(vercel_app)
        cls.bundle = load_model_bundle()
        cls.vehicle = backend.CATALOG['vehicles'][0]

    def answers(self, qualification, loan_band, price_band):
        return {
            'age_band': '18-25', 'qualification': qualification,
            'employment_type': 'SAL', 'income_band': '0', 'residence': 'O',
            'area_tier': '01 Megapolis (A)',
            'vehicle_type': self.vehicle['product_code'],
            'model_description': self.vehicle['description'],
            'loan_band': loan_band, 'vehicle_price_band': price_band,
            'has_active_loan': False, 'consent_to_log': True,
        }

    def test_deployed_style_routes_and_hosted_logging_notice(self):
        with patch.object(backend, 'VERCEL_DEPLOYMENT', True):
            home = self.client.get('/')
            script = self.client.get('/static/app.js?v=answer-review')
            css = self.client.get('/static/styles.css?v=answer-review')
            health = self.client.get('/api/health').json()
            self.assertEqual(home.status_code, 200)
            self.assertEqual(script.status_code, 200)
            self.assertEqual(css.status_code, 200)
            self.assertEqual(health['deployment_mode'], 'synthetic_demo')
            self.assertFalse(health['persistent_audit_logging'])
            self.assertEqual(health['threshold'], 0.13995426939656921)

    def test_hosted_approved_declined_inference_matches_saved_bundle_without_file_log(self):
        cases = [
            ('APPROVED', self.answers('GRAD', 'under_50000', 'under_75000')),
            ('DECLINED', self.answers('SSC', '100000_124999', '75000_94999')),
        ]
        with patch.object(backend, 'VERCEL_DEPLOYMENT', True), \
             patch.object(backend, '_append_audit', side_effect=AssertionError('hosted mode must not write a file')):
            for expected_label, payload in cases:
                with self.subTest(decision=expected_label):
                    response = self.client.post('/api/decision', json=payload)
                    self.assertEqual(response.status_code, 200, response.text)
                    result = response.json()
                    row, _ = build_model_row(ApplicationAnswers(**payload))
                    direct = predict_applicants([row], bundle=self.bundle).iloc[0]
                    self.assertEqual(result['decision'], expected_label)
                    self.assertEqual(result['decision'], direct['Decision'])
                    self.assertAlmostEqual(result['decline_probability'], direct['decline_probability'], places=12)
                    self.assertAlmostEqual(result['threshold_declined'], 0.13995426939656921)
                    self.assertFalse(result['persistent_audit_logging'])
                    self.assertEqual(result['deployment_mode'], 'synthetic_demo')


if __name__ == '__main__':
    unittest.main()
