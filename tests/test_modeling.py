from __future__ import annotations

import unittest
from unittest.mock import patch

from utils.modeling import load_or_train_pipeline, predict_credit_risk, validate_prediction_input


SAMPLE_INPUT = {
    'age': 35,
    'gender': 'Male',
    'marital_status': 'Single',
    'education_level': "Bachelor's",
    'annual_income': 50000,
    'monthly_income': 50000 / 12,
    'employment_status': 'Employed',
    'debt_to_income_ratio': 0.15,
    'credit_score': 700,
    'loan_amount': 15000,
    'loan_purpose': 'Debt consolidation',
    'interest_rate': 12.0,
    'loan_term': 36,
    'installment': 498.22,
    'grade_subgrade': 'B3',
    'num_of_open_accounts': 5,
    'total_credit_limit': 50000,
    'current_balance': 10000,
    'delinquency_history': 0,
    'public_records': 0,
    'num_of_delinquencies': 0,
}


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = load_or_train_pipeline()

    def test_validate_prediction_input_rejects_missing_fields(self):
        bad_input = dict(SAMPLE_INPUT)
        bad_input.pop('credit_score')

        with self.assertRaises(ValueError):
            validate_prediction_input(bad_input)

    def test_prediction_probabilities_are_well_formed(self):
        prediction, repayment_probability, default_probability = predict_credit_risk(
            SAMPLE_INPUT,
            self.pipeline,
        )

        self.assertIn(prediction, (0, 1))
        self.assertGreaterEqual(repayment_probability, 0.0)
        self.assertGreaterEqual(default_probability, 0.0)
        self.assertAlmostEqual(repayment_probability + default_probability, 1.0, places=8)

    def test_corrupt_artifact_falls_back_to_retraining(self):
        with patch('pathlib.Path.exists', return_value=True), \
             patch('utils.modeling.joblib.load', side_effect=RuntimeError("corrupt")), \
             patch('utils.modeling.train_random_forest_pipeline') as mock_train:
            mock_train.return_value = type('Artifacts', (), {'pipeline': 'PIPELINE'})()
            pipeline = load_or_train_pipeline()

        self.assertEqual(pipeline, 'PIPELINE')
