from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class BorrowerProfile(TypedDict):
    age: int
    gender: str
    marital_status: str
    education_level: str
    employment_status: str
    annual_income: float
    monthly_income: float
    debt_to_income_ratio: float
    credit_score: int
    loan_amount: float
    loan_purpose: str
    interest_rate: float
    loan_term: int
    installment: float
    grade_subgrade: str
    num_of_open_accounts: int
    total_credit_limit: float
    current_balance: float
    delinquency_history: int
    public_records: int
    num_of_delinquencies: int
    ml_prediction: int
    repayment_probability: float
    default_probability: float


class AgentState(TypedDict):
    borrower: BorrowerProfile
    risk_summary: Optional[str]
    retrieved_regulations: Optional[str]
    final_report: Optional[str]
    messages: Annotated[list, add_messages]
