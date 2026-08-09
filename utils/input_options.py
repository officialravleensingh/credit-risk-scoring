from __future__ import annotations


GENDER_OPTIONS = ["Male", "Female", "Other"]
MARITAL_STATUS_OPTIONS = ["Single", "Married", "Divorced", "Widowed", "Other / Not listed"]
EDUCATION_LEVEL_OPTIONS = ["High School", "Bachelor's", "Master's", "PhD", "Other"]
EMPLOYMENT_STATUS_OPTIONS = ["Employed", "Self-employed", "Unemployed", "Student", "Retired", "Other / Not listed"]
LOAN_PURPOSE_OPTIONS = [
    "Debt consolidation",
    "Car",
    "Home",
    "Business",
    "Medical",
    "Education",
    "Vacation",
    "Other",
]
GRADE_SUBGRADE_OPTIONS = [
    "A1", "A2", "A3", "A4", "A5",
    "B1", "B2", "B3", "B4", "B5",
    "C1", "C2", "C3", "C4", "C5",
    "D1", "D2", "D3", "D4", "D5",
    "E1", "E2", "E3", "E4", "E5",
    "F1", "F2", "F3", "F4", "F5",
]

TRAINED_CATEGORY_VALUES = {
    "gender": {"Male", "Female", "Other"},
    "marital_status": {"Single", "Married", "Divorced", "Widowed"},
    "education_level": {"High School", "Bachelor's", "Master's", "PhD", "Other"},
    "employment_status": {"Employed", "Self-employed", "Unemployed", "Student", "Retired"},
    "loan_purpose": {"Debt consolidation", "Car", "Home", "Business", "Medical", "Education", "Vacation", "Other"},
    "grade_subgrade": set(GRADE_SUBGRADE_OPTIONS),
}

TRAINED_LOAN_TERMS = (36, 60)

NUMERIC_TRAINING_RANGES = {
    "age": (21, 75),
    "annual_income": (6000.0, 400000.0),
    "debt_to_income_ratio": (0.01, 0.667),
    "credit_score": (373, 850),
    "loan_amount": (500.0, 49039.69),
    "interest_rate": (3.14, 22.51),
    "loan_term": (36, 60),
    "num_of_open_accounts": (0, 15),
    "total_credit_limit": (6157.80, 454394.19),
    "current_balance": (496.35, 352177.90),
    "delinquency_history": (0, 11),
    "public_records": (0, 2),
    "num_of_delinquencies": (0, 11),
}

FIELD_LABELS = {
    "age": "Age",
    "annual_income": "Annual income",
    "debt_to_income_ratio": "Debt-to-income ratio",
    "credit_score": "Credit score",
    "loan_amount": "Loan amount",
    "interest_rate": "Interest rate",
    "loan_term": "Loan term",
    "num_of_open_accounts": "Open accounts",
    "total_credit_limit": "Total credit limit",
    "current_balance": "Current balance",
    "delinquency_history": "Delinquency history",
    "public_records": "Public records",
    "num_of_delinquencies": "Number of delinquencies",
    "marital_status": "Marital status",
    "employment_status": "Employment status",
}


def calculate_installment(loan_amount: float, interest_rate: float, loan_term: int) -> float:
    if loan_term <= 0:
        return 0.0

    monthly_rate = interest_rate / 100 / 12
    if monthly_rate == 0:
        return loan_amount / loan_term

    numerator = loan_amount * monthly_rate * (1 + monthly_rate) ** loan_term
    denominator = (1 + monthly_rate) ** loan_term - 1
    return numerator / denominator


def _format_value(field_name: str, value: float | int) -> str:
    if field_name in {"annual_income", "loan_amount", "total_credit_limit", "current_balance"}:
        return f"${float(value):,.2f}"
    if field_name == "interest_rate":
        return f"{float(value):.2f}%"
    if field_name == "debt_to_income_ratio":
        return f"{float(value) * 100:.1f}%"
    if field_name == "loan_term":
        return f"{int(value)} months"
    return str(value)


def collect_constraint_notes(input_data: dict) -> list[str]:
    notes: list[str] = []

    for field_name, (low, high) in NUMERIC_TRAINING_RANGES.items():
        if field_name not in input_data:
            continue

        value = input_data[field_name]
        if value < low or value > high:
            notes.append(
                f"{FIELD_LABELS[field_name]} is outside the training range "
                f"({_format_value(field_name, low)} to {_format_value(field_name, high)})."
            )

    loan_term = input_data.get("loan_term")
    loan_term_low, loan_term_high = NUMERIC_TRAINING_RANGES["loan_term"]
    if loan_term_low <= loan_term <= loan_term_high and loan_term not in TRAINED_LOAN_TERMS:
        notes.append("Loan term is outside the trained term values (36 or 60 months).")

    for field_name in ("marital_status", "employment_status"):
        value = str(input_data.get(field_name, "")).strip()
        if value and value not in TRAINED_CATEGORY_VALUES[field_name]:
            notes.append(
                f"{FIELD_LABELS[field_name]} is outside the training categories and will be treated as an unseen value."
            )

    return notes
