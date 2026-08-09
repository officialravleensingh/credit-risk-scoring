from __future__ import annotations

import re


REPORT_HEADERS = [
    "BORROWER PROFILE & RISK ANALYSIS",
    "LENDING DECISION",
    "REGULATORY REFERENCES",
    "LEGAL DISCLAIMER",
]


def parse_report_sections(report_text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    matches = list(
        re.finditer(
            r"^##\s+(BORROWER PROFILE & RISK ANALYSIS|LENDING DECISION|REGULATORY REFERENCES|LEGAL DISCLAIMER)\s*$",
            report_text,
            flags=re.MULTILINE,
        )
    )

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(report_text)
        sections[match.group(1)] = report_text[start:end].strip()

    if not sections:
        sections["BORROWER PROFILE & RISK ANALYSIS"] = report_text.strip()

    return sections


def infer_decision_label(content: str, prediction: int) -> str:
    normalized = content.upper()
    if "DECLINE" in normalized:
        return "DECLINE"
    if "APPROVE" in normalized:
        return "APPROVE"
    return "APPROVE" if prediction == 1 else "DECLINE"
