"""ADK tools exposing the local user profile store to agents."""

from ..profile_store import load_profile, missing_required, set_field


def get_profile() -> dict:
    """Get the user's saved financial profile and which required fields are still missing.

    Returns:
        A dict with 'profile' (known fields so far, may be empty for a new
        user) and 'missing_required' (list of required field keys not yet
        answered, e.g. "age", "risk_tolerance").
    """
    profile = load_profile()
    return {"profile": profile, "missing_required": missing_required(profile)}


def update_profile_field(field: str, value: str) -> dict:
    """Save or update a single field in the user's financial profile.

    Args:
        field: One of: age, household_status, income, location,
            risk_tolerance, investment_experience, time_horizon, expenses,
            assets, liabilities, tax_bracket, employment_status,
            retirement_goals, major_life_goals, liquidity_needs.
        value: The user's answer for that field, as free text.

    Returns:
        A dict with 'profile' (all fields saved so far) and
        'missing_required' (required fields still missing), or a dict with
        an 'error' key if the field name isn't recognized.
    """
    result = set_field(field, value)
    if "error" in result:
        return result
    return {"profile": result, "missing_required": missing_required(result)}
