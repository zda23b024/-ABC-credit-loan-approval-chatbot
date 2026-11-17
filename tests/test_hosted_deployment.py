"""Hosted deployment checks using FastAPI's local ASGI test client."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import lightgbm
from fastapi.testclient import TestClient

from chatbot import app as backend
from credit_model_runtime import load_model_bundle, predict_applicants
from chatbot.app import ApplicationAnswers, build_model_row
from app import app as hosted_app


class HostedDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(hosted_app)
        cls.bundle = load_model_bundle()
        cls.vehicle = backend.CATALOG['vehicles'][0]

    def test_railpack_installs_openmp_in_the_runtime_environment(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / 'railpack.json').read_text(encoding='utf-8'))
        requirements = (root / 'requirements.txt').read_text(encoding='utf-8')
        self.assertEqual(config['provider'], 'python')
        self.assertEqual(config['packages']['python'], '3.12')
        self.assertIn('libgomp1', config['deploy']['aptPackages'])
        self.assertIn('$PORT', config['deploy']['startCommand'])
        self.assertNotIn('OMP_NUM_THREADS', config['deploy']['startCommand'])
        self.assertIn('lightgbm==4.7.0', requirements)
        self.assertEqual(lightgbm.__version__, '4.7.0')
        self.assertFalse((root / 'Dockerfile.vercel').exists())
        self.assertTrue(self.bundle['model'])
        self.assertEqual(self.bundle['threshold_declined'], 0.13995426939656921)

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

    def test_hosted_style_routes_and_logging_notice(self):
        with patch.object(backend, 'HOSTED_DEMO_DEPLOYMENT', True):
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
        with patch.object(backend, 'HOSTED_DEMO_DEPLOYMENT', True), \
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
