"""Local JSON persistence for the user's financial profile.

Single-local-user demo storage: everything lives in one profile.json under
financial_advisor/data/, which is gitignored since it holds personal
financial information.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
PROFILE_PATH = DATA_DIR / "profile.json"

# label: shown to the agent as guidance on what the field means.
# required: whether the coordinator gates other specialists on this field.
# Dict order = the order fields should be asked in; "name" comes first
# since it's also used as the identity key for saved conversation history.
PROFILE_SCHEMA = {
    "name": {"label": "First name (used to greet you and save your conversation history)", "required": True},
    "age": {"label": "Age", "required": True},
    "household_status": {"label": "Household status (single, married, dependents, etc.)", "required": True},
    "income": {"label": "Annual income and how stable it is", "required": True},
    "location": {"label": "State/country of residence", "required": True},
    "risk_tolerance": {"label": "Risk tolerance", "required": True},
    "investment_experience": {"label": "Investment experience", "required": True},
    "time_horizon": {"label": "Investing time horizon", "required": True},
    "expenses": {"label": "Typical monthly expenses", "required": False},
    "assets": {"label": "Current assets (cash, investments, property, etc.)", "required": False},
    "liabilities": {"label": "Current debts/liabilities", "required": False},
    "tax_bracket": {"label": "Approximate tax bracket", "required": False},
    "employment_status": {"label": "Employment status", "required": False},
    "retirement_goals": {"label": "Retirement goals", "required": False},
    "major_life_goals": {"label": "Major life goals (house, education, travel, etc.)", "required": False},
    "liquidity_needs": {"label": "Liquidity needs", "required": False},
}

REQUIRED_FIELDS = [key for key, spec in PROFILE_SCHEMA.items() if spec["required"]]


def load_profile() -> dict:
    if not PROFILE_PATH.exists():
        return {}
    with open(PROFILE_PATH) as f:
        return json.load(f)


def set_field(field: str, value: str) -> dict:
    """Save one field and return the full updated profile, or {'error': ...}."""
    if field not in PROFILE_SCHEMA:
        return {"error": f"Unknown field '{field}'. Valid fields: {', '.join(PROFILE_SCHEMA)}"}
    profile = load_profile()
    profile[field] = value
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=2)
    return profile


def missing_required(profile: dict) -> list:
    return [field for field in REQUIRED_FIELDS if not profile.get(field)]


def is_complete(profile: dict) -> bool:
    return not missing_required(profile)
