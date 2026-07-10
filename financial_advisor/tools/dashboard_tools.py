"""ADK tools exposing the local financial dashboard store to agents."""

from ..dashboard_store import compute_guidance, compute_summary, load_dashboard, missing_required, set_field


def _summary_with_guidance(data: dict) -> dict:
    summary = compute_summary(data)
    summary.update(compute_guidance(data))
    return summary


def get_dashboard_status() -> dict:
    """Get the user's saved financial dashboard inputs and computed summary.

    Returns:
        A dict with 'data' (raw saved values), 'missing_required' (required
        field keys not yet answered), and 'summary' (computed metrics such
        as net_worth, monthly_cash_flow, savings_rate_pct, and
        emergency_fund_months, plus budgeting/investing guidance —
        budget_needs, budget_wants, budget_savings_target,
        recommended_investing_amount, recommended_investing_pct,
        emergency_fund_target, emergency_fund_monthly_contribution — once
        monthly_income and monthly_expenses are both known).
    """
    data = load_dashboard()
    return {"data": data, "missing_required": missing_required(data), "summary": _summary_with_guidance(data)}


def update_dashboard_field(field: str, value: str) -> dict:
    """Save or update a single financial dashboard input field.

    Args:
        field: One of: total_assets, total_liabilities, monthly_income,
            monthly_expenses, emergency_fund_balance, debt_breakdown,
            investment_balance, retirement_balance, retirement_goal,
            credit_score. Dollar-amount fields should be plain numbers
            (e.g. "5000", not "$5,000/mo").
        value: The user's answer for that field.

    Returns:
        A dict with 'data', 'missing_required', and 'summary' (see
        get_dashboard_status), or a dict with an 'error' key if the field
        name isn't recognized or a numeric field couldn't be parsed.
    """
    result = set_field(field, value)
    if "error" in result:
        return result
    return {"data": result, "missing_required": missing_required(result), "summary": _summary_with_guidance(result)}
