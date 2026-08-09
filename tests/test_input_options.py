from __future__ import annotations

import unittest

from utils.input_options import calculate_installment, collect_constraint_notes


class InputOptionsTests(unittest.TestCase):
    def test_zero_interest_installment_is_simple_division(self):
        installment = calculate_installment(1200.0, 0.0, 12)
        self.assertAlmostEqual(installment, 100.0, places=8)

    def test_constraint_notes_capture_unseen_ranges_and_categories(self):
        notes = collect_constraint_notes(
            {
                "age": 19,
                "annual_income": 5000.0,
                "debt_to_income_ratio": 1.2,
                "credit_score": 320,
                "loan_amount": 100.0,
                "interest_rate": 0.0,
                "loan_term": 48,
                "num_of_open_accounts": 20,
                "total_credit_limit": 1000.0,
                "current_balance": 100.0,
                "delinquency_history": 12,
                "public_records": 3,
                "num_of_delinquencies": 15,
                "marital_status": "Other / Not listed",
                "employment_status": "Other / Not listed",
            }
        )

        self.assertTrue(any("Age is outside the training range" in note for note in notes))
        self.assertTrue(any("Loan term is outside the trained term values" in note for note in notes))
        self.assertTrue(any("Marital status is outside the training categories" in note for note in notes))
