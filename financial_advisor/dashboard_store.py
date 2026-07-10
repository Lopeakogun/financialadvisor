"""Local JSON persistence for the user's financial dashboard inputs.

Mirrors profile_store.py: single-local-user demo storage in
financial_advisor/data/ (gitignored). Computed metrics (net worth, cash
flow, etc.) are derived in plain Python from the raw stored numbers rather
than left to the LLM to calculate.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DASHBOARD_PATH = DATA_DIR / "dashboard.json"

# numeric fields are stored as floats; required fields gate access to the
# specialist agents, same pattern as PROFILE_SCHEMA in profile_store.py.
DASHBOARD_SCHEMA = {
    "total_assets": {"label": "Total assets (cash, investments, property, etc.), in dollars", "required": True, "numeric": True},
    "total_liabilities": {"label": "Total liabilities/debts, in dollars", "required": True, "numeric": True},
    "monthly_income": {"label": "Monthly take-home income, in dollars", "required": True, "numeric": True},
    "monthly_expenses": {"label": "Typical monthly expenses, in dollars", "required": True, "numeric": True},
    "emergency_fund_balance": {"label": "Emergency fund balance, in dollars", "required": True, "numeric": True},
    "debt_breakdown": {"label": "Breakdown of debts by type (credit card, student loan, mortgage, auto, other)", "required": False, "numeric": False},
    "investment_balance": {"label": "Total invested balance across accounts, in dollars", "required": False, "numeric": True},
    "retirement_balance": {"label": "Current retirement account balance, in dollars", "required": False, "numeric": True},
    "retirement_goal": {"label": "Target retirement savings goal, in dollars", "required": False, "numeric": True},
    "credit_score": {"label": "Credit score, if known", "required": False, "numeric": True},
    # Spending breakdown — how monthly_expenses actually breaks down by
    # category, not just the aggregate total. Optional (doesn't block
    # specialist access, same as the other optional fields above) but
    # proactively offered right after the initial dashboard summary, since
    # this is what turns "how much do you spend" into "what do you spend
    # it on" — see financial_dashboard_agent.py.
    "housing_expense": {"label": "Monthly housing cost (rent or mortgage), in dollars", "required": False, "numeric": True},
    "transportation_expense": {"label": "Monthly transportation cost (car payment, gas, transit), in dollars", "required": False, "numeric": True},
    "food_expense": {"label": "Monthly food and groceries cost, in dollars", "required": False, "numeric": True},
    "utilities_expense": {"label": "Monthly utilities cost (electric, water, internet, phone), in dollars", "required": False, "numeric": True},
    "insurance_expense": {"label": "Monthly insurance cost (health, auto, renters/home), in dollars", "required": False, "numeric": True},
    "debt_payments_expense": {"label": "Monthly debt payments (credit cards, loans, student loans), in dollars", "required": False, "numeric": True},
    "entertainment_expense": {"label": "Monthly entertainment, subscriptions, and discretionary spending, in dollars", "required": False, "numeric": True},
    "other_expense": {"label": "Other monthly expenses not covered above, in dollars", "required": False, "numeric": True},
}

REQUIRED_FIELDS = [key for key, spec in DASHBOARD_SCHEMA.items() if spec["required"]]

SPENDING_CATEGORY_FIELDS = [
    "housing_expense",
    "transportation_expense",
    "food_expense",
    "utilities_expense",
    "insurance_expense",
    "debt_payments_expense",
    "entertainment_expense",
    "other_expense",
]


def load_dashboard() -> dict:
    if not DASHBOARD_PATH.exists():
        return {}
    with open(DASHBOARD_PATH) as f:
        return json.load(f)


def set_field(field: str, value) -> dict:
    """Save one field and return the full updated dashboard, or {'error': ...}."""
    if field not in DASHBOARD_SCHEMA:
        return {"error": f"Unknown field '{field}'. Valid fields: {', '.join(DASHBOARD_SCHEMA)}"}
    spec = DASHBOARD_SCHEMA[field]
    if spec["numeric"]:
        try:
            value = float(str(value).replace(",", "").replace("$", "").strip())
        except ValueError:
            return {"error": f"'{field}' must be a plain number, got {value!r}"}
    data = load_dashboard()
    data[field] = value
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DASHBOARD_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return data


def missing_required(data: dict) -> list:
    return [field for field in REQUIRED_FIELDS if data.get(field) in (None, "")]


def is_complete(data: dict) -> bool:
    return not missing_required(data)


# Tracked as persistent data (not ADK session state), same as every other
# field here — session state doesn't survive a returning user landing on
# a brand new session, but the coordinator's top-level gate (agent.py)
# needs to know "was this ever offered" across restarts, the same way it
# already knows "are the required fields done" from the file, not from
# session state.
_SPENDING_BREAKDOWN_OFFERED_KEY = "_spending_breakdown_offered"


def mark_spending_breakdown_offered() -> None:
    data = load_dashboard()
    data[_SPENDING_BREAKDOWN_OFFERED_KEY] = True
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DASHBOARD_PATH, "w") as f:
        json.dump(data, f, indent=2)


def spending_breakdown_pending(data: dict) -> bool:
    """True if the required fields are done but the (optional) spending
    breakdown hasn't been offered yet — used by the top-level coordinator
    gate to route to financial_dashboard one more time so the offer
    actually gets a chance to run, even for a returning user whose
    required fields were already complete from a prior session."""
    if missing_required(data):
        return False  # required stage isn't done — not this gate's concern
    return not data.get(_SPENDING_BREAKDOWN_OFFERED_KEY)


def compute_summary(data: dict) -> dict:
    """Derive dashboard metrics (net worth, cash flow, etc.) from raw stored values."""
    summary = {}

    assets, liabilities = data.get("total_assets"), data.get("total_liabilities")
    if assets is not None and liabilities is not None:
        summary["net_worth"] = round(assets - liabilities, 2)

    income, expenses = data.get("monthly_income"), data.get("monthly_expenses")
    if income is not None and expenses is not None:
        cash_flow = income - expenses
        summary["monthly_cash_flow"] = round(cash_flow, 2)
        if income:
            summary["savings_rate_pct"] = round(cash_flow / income * 100, 1)

    emergency_fund = data.get("emergency_fund_balance")
    if emergency_fund is not None and expenses:
        summary["emergency_fund_months"] = round(emergency_fund / expenses, 1)

    if data.get("debt_breakdown"):
        summary["debt_breakdown"] = data["debt_breakdown"]

    if data.get("investment_balance") is not None:
        summary["investment_balance"] = data["investment_balance"]

    retirement_balance, retirement_goal = data.get("retirement_balance"), data.get("retirement_goal")
    if retirement_balance is not None:
        summary["retirement_balance"] = retirement_balance
        if retirement_goal:
            summary["retirement_progress_pct"] = round(retirement_balance / retirement_goal * 100, 1)

    if data.get("credit_score") is not None:
        summary["credit_score"] = data["credit_score"]

    return summary


# Standard, widely-cited personal-finance rules of thumb, not personalized
# advice: the 50/30/20 budget split (needs/wants/savings), and prioritizing
# a 6-month emergency fund before allocating the full "savings" share to
# investing. Computed in plain Python — see profile_store/dashboard_store
# module docstrings for why math like this isn't left to the LLM.
EMERGENCY_FUND_TARGET_MONTHS = 6
BUDGET_SPLIT = {"needs": 0.50, "wants": 0.30, "savings": 0.20}


def compute_guidance(data: dict) -> dict:
    """Rule-of-thumb budgeting/investing guidance derived from income and expenses."""
    income, expenses = data.get("monthly_income"), data.get("monthly_expenses")
    if not income or not expenses:
        return {}

    guidance = {
        "budget_needs": round(income * BUDGET_SPLIT["needs"], 2),
        "budget_wants": round(income * BUDGET_SPLIT["wants"], 2),
        "budget_savings_target": round(income * BUDGET_SPLIT["savings"], 2),
    }

    leftover = income - expenses
    # Never recommend saving/investing more than is actually left over
    # after expenses, even if the 50/30/20 target says otherwise.
    available = max(0.0, min(leftover, guidance["budget_savings_target"]))
    if leftover < guidance["budget_savings_target"]:
        guidance["savings_shortfall"] = round(guidance["budget_savings_target"] - max(leftover, 0), 2)

    emergency_fund = data.get("emergency_fund_balance") or 0
    emergency_target = expenses * EMERGENCY_FUND_TARGET_MONTHS
    guidance["emergency_fund_target"] = round(emergency_target, 2)

    if emergency_fund < emergency_target:
        # Not yet fully funded: split the available savings share roughly
        # evenly between topping up the emergency fund and investing.
        emergency_share = round(available * 0.5, 2)
        investing_share = round(available - emergency_share, 2)
        guidance["emergency_fund_monthly_contribution"] = emergency_share
        guidance["emergency_fund_pct_of_income"] = round(emergency_share / income * 100, 1)
        if emergency_share > 0:
            gap = emergency_target - emergency_fund
            guidance["emergency_fund_months_to_target"] = round(gap / emergency_share, 1)
    else:
        investing_share = available

    guidance["recommended_investing_amount"] = round(investing_share, 2)
    guidance["recommended_investing_pct"] = round(investing_share / income * 100, 1)

    return guidance


# Two more widely-cited underwriting/budgeting rules of thumb, not
# personalized advice: housing costs at or under ~30% of gross income
# (the long-standing "30% rule," tracing to US federal housing
# affordability standards), and total debt-to-income — housing plus other
# debt payments — at or under ~36% of gross income (the conventional
# mortgage-underwriting DTI ceiling).
HOUSING_PCT_GUIDELINE = 30.0
DTI_PCT_GUIDELINE = 36.0


def compute_spending_breakdown(data: dict) -> dict:
    """Category-level spending breakdown and how it stacks up against common guidelines.

    Only returns anything once at least one spending category field is
    saved — this is optional, proactively-offered enrichment on top of
    the single required monthly_expenses aggregate, not required data.
    """
    income = data.get("monthly_income")
    if not income:
        return {}

    categories = {f: data[f] for f in SPENDING_CATEGORY_FIELDS if data.get(f) is not None}
    if not categories:
        return {}

    total_categorized = sum(categories.values())
    breakdown = {
        "categories": {f: round(v, 2) for f, v in categories.items()},
        "total_categorized": round(total_categorized, 2),
        "categories_pct_of_income": {f: round(v / income * 100, 1) for f, v in categories.items()},
    }

    expenses = data.get("monthly_expenses")
    if expenses is not None:
        diff = round(expenses - total_categorized, 2)
        if abs(diff) > max(20.0, expenses * 0.05):
            breakdown["uncategorized_difference"] = diff

    housing = categories.get("housing_expense")
    if housing is not None:
        housing_pct = round(housing / income * 100, 1)
        breakdown["housing_pct_of_income"] = housing_pct
        breakdown["housing_over_guideline"] = housing_pct > HOUSING_PCT_GUIDELINE

    if "debt_payments_expense" in categories or housing is not None:
        dti_pct = round(((categories.get("debt_payments_expense", 0) + (housing or 0)) / income) * 100, 1)
        breakdown["debt_to_income_pct"] = dti_pct
        breakdown["debt_to_income_over_guideline"] = dti_pct > DTI_PCT_GUIDELINE

    return breakdown
