from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.nodes import regulation_retriever_node
from agent import rag
from utils.reporting import infer_decision_label, parse_report_sections


class RagAndReportingTests(unittest.TestCase):
    def tearDown(self):
        rag.reset_retriever_cache()

    def test_rag_falls_back_to_tfidf_when_semantic_backend_unavailable(self):
        with patch('agent.rag._try_build_semantic_retriever', side_effect=RuntimeError("offline")):
            retriever = rag.get_retriever()

        self.assertEqual(retriever['type'], 'lexical')
        self.assertEqual(retriever['backend'], 'tfidf')
        self.assertTrue(rag.retrieve('credit score lending guidelines', k=2))

    def test_report_sections_are_parsed_by_header(self):
        report = """## BORROWER PROFILE & RISK ANALYSIS
Profile text.

## LENDING DECISION
DECLINE due to elevated default risk.

## REGULATORY REFERENCES
Reference text.

## LEGAL DISCLAIMER
Disclaimer text.
"""
        sections = parse_report_sections(report)

        self.assertEqual(sections["BORROWER PROFILE & RISK ANALYSIS"], "Profile text.")
        self.assertEqual(sections["LENDING DECISION"], "DECLINE due to elevated default risk.")
        self.assertEqual(infer_decision_label(sections["LENDING DECISION"], 1), "DECLINE")

    def test_regulation_retriever_preserves_deterministic_unique_order(self):
        side_effects = [
            "A\n\n---\n\nB",
            "B\n\n---\n\nC",
            "C\n\n---\n\nD",
            "D\n\n---\n\nE",
            "E\n\n---\n\nF",
            "F\n\n---\n\nG",
        ]
        state = {
            "borrower": {
                "credit_score": 700,
                "debt_to_income_ratio": 0.15,
                "employment_status": "Employed",
                "loan_amount": 15000,
            }
        }

        with patch('agent.nodes.retrieve', side_effect=side_effects):
            result = regulation_retriever_node(state)

        self.assertEqual(result["retrieved_regulations"], "A\n\n---\n\nB\n\n---\n\nC\n\n---\n\nD\n\n---\n\nE")
