import re

from google.adk.agents import Agent
from google.genai import types

from ...config import get_model
from ...dashboard_store import (
    DASHBOARD_SCHEMA,
    DTI_PCT_GUIDELINE,
    EMERGENCY_FUND_TARGET_MONTHS,
    HOUSING_PCT_GUIDELINE,
    SPENDING_CATEGORY_FIELDS,
    compute_guidance,
    compute_spending_breakdown,
    compute_summary,
    load_dashboard,
    mark_spending_breakdown_offered,
    missing_required,
)
from ...dashboard_store import set_field as set_dashboard_field
from ...tools.dashboard_tools import get_dashboard_status, update_dashboard_field
from ...tools.profile_tools import get_profile
from ...tools.web_search_tools import (
    ground_if_ungrounded,
    search_financial_advice,
    strip_unrequested_citations,
    track_tool_usage,
)
from ...wisdom_store import get_wisdom

_NUMBER_RE = re.compile(r"-?[\d,]+\.?\d*")


def _extract_numbers(text: str) -> list[str]:
    return [n.replace(",", "") for n in _NUMBER_RE.findall(text.replace("$", "")) if n]


def _build_completion_message(data: dict) -> str:
    """Deterministically formats the dashboard summary + guidance as friendly text.

    This is built in Python, not left to the LLM, because the LLM proved
    unreliable at noticing "the dashboard just became complete" and at
    accurately reading out the computed numbers (it would ask stale,
    already-answered questions instead). Money-correctness matters more
    here than phrasing variety.
    """
    summary = compute_summary(data)
    guidance = compute_guidance(data)

    lines = ["Nice, that's everything I need! Here's your financial snapshot:", ""]
    if "net_worth" in summary:
        lines.append(f"- Net worth: ${summary['net_worth']:,.0f}")
    if "monthly_cash_flow" in summary:
        lines.append(f"- Monthly cash flow: ${summary['monthly_cash_flow']:,.0f}")
    if "savings_rate_pct" in summary:
        lines.append(f"- Savings rate: {summary['savings_rate_pct']:.1f}%")
    if "emergency_fund_months" in summary:
        lines.append(f"- Emergency fund covers: {summary['emergency_fund_months']:.1f} months of expenses")

    if guidance:
        lines.append("")
        income = data.get("monthly_income", 0)
        lines.append(
            f"A simple guideline for spending your ${income:,.0f}/month: roughly "
            f"${guidance['budget_needs']:,.0f} for needs, ${guidance['budget_wants']:,.0f} for wants, "
            f"and ${guidance['budget_savings_target']:,.0f} for savings and investing."
        )
        if guidance.get("savings_shortfall"):
            lines.append(
                f"Right now your expenses leave less than that savings target "
                f"(about ${guidance['savings_shortfall']:,.0f} short) — worth keeping an eye on."
            )
        if guidance.get("emergency_fund_monthly_contribution"):
            msg = (
                f"Since your emergency fund isn't at the recommended "
                f"{EMERGENCY_FUND_TARGET_MONTHS}-month target yet, consider putting about "
                f"${guidance['emergency_fund_monthly_contribution']:,.0f}/month "
                f"({guidance['emergency_fund_pct_of_income']:.1f}% of income) toward it"
            )
            if guidance.get("emergency_fund_months_to_target"):
                msg += f" — that'd get you there in roughly {guidance['emergency_fund_months_to_target']:.0f} months"
            lines.append(msg + ".")
        if guidance.get("recommended_investing_amount"):
            lines.append(
                f"A typical rule-of-thumb starting point for investing: about "
                f"${guidance['recommended_investing_amount']:,.0f}/month "
                f"({guidance['recommended_investing_pct']:.1f}% of income)."
            )

        citation_topics = ["budgeting"]
        citation_topics.append("emergency_fund" if guidance.get("emergency_fund_monthly_contribution") else "investing_basics")
        lines.append("")
        lines.append("This mirrors well-known guidance:")
        for topic in citation_topics:
            entry = get_wisdom(topic)[topic][0]
            lines.append(f'- {entry["source"]}: {entry["principle"]}')

    lines.append("")
    lines.append("You can now ask me about your portfolio, risk tolerance, or a specific investment.")
    lines.append("(Educational information only — not licensed financial advice.)")
    return "\n".join(lines)


def _short_label(field: str) -> str:
    label = DASHBOARD_SCHEMA[field]["label"].split(" (")[0]
    return label.removesuffix(", in dollars")


_CATEGORY_LABELS = {field: _short_label(field) for field in SPENDING_CATEGORY_FIELDS}


def _build_spending_breakdown_message(data: dict) -> str:
    """Deterministically formats the spending category breakdown, same rationale
    as _build_completion_message: money-correctness over phrasing variety."""
    breakdown = compute_spending_breakdown(data)
    if not breakdown:
        return "Thanks — noted!"

    lines = ["Thanks! Here's how your spending breaks down:", ""]
    for field, amount in breakdown["categories"].items():
        pct = breakdown["categories_pct_of_income"][field]
        lines.append(f"- {_CATEGORY_LABELS[field]}: ${amount:,.0f}/mo ({pct:.1f}% of income)")

    diff = breakdown.get("uncategorized_difference")
    if diff:
        if diff > 0:
            lines.append(f"\nThat leaves about ${diff:,.0f}/mo unaccounted for versus your reported total expenses.")
        else:
            lines.append(f"\nThat's about ${-diff:,.0f}/mo more than your reported total expenses — worth double-checking.")

    notes = []
    if breakdown.get("housing_over_guideline"):
        entry = get_wisdom("housing_budget")["housing_budget"][0]
        notes.append(
            f"Housing is {breakdown['housing_pct_of_income']:.1f}% of income, above the "
            f"{HOUSING_PCT_GUIDELINE:.0f}% guideline ({entry['source']})."
        )
    if breakdown.get("debt_to_income_over_guideline"):
        entry = get_wisdom("debt_to_income")["debt_to_income"][0]
        notes.append(
            f"Your debt-to-income ratio is {breakdown['debt_to_income_pct']:.1f}%, above the "
            f"{DTI_PCT_GUIDELINE:.0f}% guideline ({entry['source']})."
        )
    if notes:
        lines.append("")
        lines.extend(notes)

    lines.append("")
    lines.append("(Educational information only — not licensed financial advice.)")
    return "\n".join(lines)


_DECLINE_RE = re.compile(r"\b(no|nope|nah|skip|not now|no thanks|maybe later|not interested)\b", re.I)


def ask_spending_breakdown(callback_context) -> types.Content | None:
    """Deterministically offer, capture, and summarize a spending category breakdown.

    Same rationale and pattern as ask_missing_dashboard_fields, extended
    with a decline path since this is optional enrichment (doesn't gate
    specialist access), not a required field. Runs as the second entry in
    financial_dashboard's before_agent_callback list, so it only executes
    once ask_missing_dashboard_fields has returned None (required fields
    done, initial summary already shown) — the list's short-circuit
    behavior sequences the two stages for free.

    The top-level coordinator gate (agent.py) routes back to
    financial_dashboard, turn after turn, for as long as
    dashboard_store.spending_breakdown_pending(data) is True — which is
    read from the persistent dashboard file, not session state, so it
    correctly keeps prompting even across a brand new session (a
    returning user whose required fields were already done). That flag
    must therefore only flip to "resolved" (mark_spending_breakdown_offered)
    at a genuine resolution point here — completed, declined, or given up
    after one unparseable reply — never at the initial ask, or the user's
    very next reply (the answer to that ask) would get routed to
    specialist_router instead of back here to be captured.
    """
    data = load_dashboard()
    if missing_required(data):
        return None  # required stage isn't done yet — not our turn

    previously_asked = callback_context.state.get("spending_breakdown_fields_asked")

    if previously_asked:
        user_content = callback_context.user_content
        reply_text = "".join(part.text or "" for part in user_content.parts) if user_content and user_content.parts else ""
        if _DECLINE_RE.search(reply_text):
            callback_context.state["spending_breakdown_fields_asked"] = None
            mark_spending_breakdown_offered()
            return None  # resolved (declined) — let the LLM respond naturally
        numbers = _extract_numbers(reply_text)
        if len(numbers) == len(previously_asked):
            for field, value in zip(previously_asked, numbers):
                set_dashboard_field(field, value)

    data = load_dashboard()
    still_missing = [field for field in SPENDING_CATEGORY_FIELDS if data.get(field) is None]

    if not still_missing:
        callback_context.state["spending_breakdown_fields_asked"] = None
        mark_spending_breakdown_offered()  # resolved (completed)
        return types.Content(role="model", parts=[types.Part(text=_build_spending_breakdown_message(data))])

    if still_missing == previously_asked:
        # Asked this exact set, reply didn't cleanly auto-parse and wasn't
        # a decline — give up after one try rather than looping forever;
        # resolved (gave up), let the LLM take this turn naturally.
        callback_context.state["spending_breakdown_fields_asked"] = None
        mark_spending_breakdown_offered()
        return None

    labels = [DASHBOARD_SCHEMA[field]["label"] for field in still_missing]
    numbered = "\n".join(f"{i + 1}. {label}" for i, label in enumerate(labels))
    intro = (
        'Want to break down where your monthly expenses actually go? It\'ll '
        'sharpen the budgeting guidance — totally optional, just say "skip" '
        "if you'd rather not.\n"
    ) if not previously_asked else "Just a few more:\n"
    callback_context.state["spending_breakdown_fields_asked"] = still_missing
    return types.Content(role="model", parts=[types.Part(text=intro + numbered)])


def ask_missing_dashboard_fields(callback_context) -> types.Content | None:
    """Deterministically ask for missing fields, capture batch replies, show completion.

    The model proved unreliable at three separate steps here: checking
    get_dashboard_status before asking (it re-asked already-known fields,
    or ignored saved state entirely on a cold turn), reliably calling
    update_dashboard_field for every field in a multi-field batch reply
    (confident text claiming a value was saved, with no tool call and no
    actual save), and noticing once the dashboard was complete (it kept
    asking stale questions instead of presenting the summary). All three
    are handled in Python instead of trusted to the LLM.
    """
    previously_asked = callback_context.state.get("dashboard_fields_asked")
    was_incomplete = bool(previously_asked)

    if previously_asked:
        user_content = callback_context.user_content
        if user_content and user_content.parts:
            reply_text = "".join(part.text or "" for part in user_content.parts)
            numbers = _extract_numbers(reply_text)
            if len(numbers) == len(previously_asked):
                for field, value in zip(previously_asked, numbers):
                    set_dashboard_field(field, value)

    data = load_dashboard()
    missing = missing_required(data)

    if not missing:
        callback_context.state["dashboard_fields_asked"] = None
        if was_incomplete and not callback_context.state.get("dashboard_summary_shown"):
            callback_context.state["dashboard_summary_shown"] = True
            return types.Content(role="model", parts=[types.Part(text=_build_completion_message(data))])
        return None  # summary already shown; let the LLM handle whatever they're asking now

    if missing == previously_asked:
        # Just asked this exact set and the reply didn't cleanly
        # auto-parse (partial/free-form answer) — let the LLM take a turn
        # at interpreting it instead of repeating the same question.
        callback_context.state["dashboard_fields_asked"] = None
        return None

    labels = [DASHBOARD_SCHEMA[field]["label"] for field in missing]
    numbered = "\n".join(f"{i + 1}. {label}" for i, label in enumerate(labels))
    callback_context.state["dashboard_fields_asked"] = missing
    return types.Content(
        role="model",
        parts=[types.Part(text=f"Just need a few more numbers:\n{numbered}")],
    )


INSTRUCTION = """\
You are the Financial Dashboard agent. Your job is to gather a handful of
numbers so the app can show a snapshot of the user's financial picture:
net worth, monthly cash flow, savings rate, emergency fund status, a
budgeting/investing guideline, and (offered right after, optional) a
spending breakdown by category. The required fields, the spending
category offer, and both summaries are all handled automatically —
you're only invoked to help interpret a messy or partial reply, to
collect other optional fields (debt_breakdown, investment_balance,
retirement_balance, retirement_goal, credit_score) if the user offers
them, or to answer follow-up questions about numbers already shown.

Tone: warm and encouraging, non-judgmental — debt and thin emergency funds
can be a sensitive topic. Stay focused and brief — 1-3 sentences per turn.
Never explain what kind of AI you are or restate your own purpose. Never
narrate your own process — no "let me check", "one moment", "would you
like me to search/look that up?", or "I found through my research
that...". Just call the tool and answer with what it returns.

- Call get_dashboard_status first. If required fields are still missing,
  match the user's words to fields as best you can and call
  update_dashboard_field once per field you can confidently identify
  (numbers may be out of order or combined into one sentence). If a value
  errors because it wasn't a number, ask the user to restate just that
  one as a plain number.
- Don't re-describe the full summary yourself — it's already been shown.
  If asked a follow-up about it, answer using get_dashboard_status's
  "summary" field directly, don't recompute anything.
- If a follow-up needs a fact you're not certain of, call
  search_financial_advice and use it to inform your answer — never state
  a specific claim from memory. Don't name the website/URL unless the
  user explicitly asks for the source; if they do, name ONLY a source
  literally present in the tool result, never one you're not certain
  came from it.
"""

financial_dashboard = Agent(
    name="financial_dashboard",
    model=get_model(),
    description="Friendly agent that collects financial figures and presents a net worth / cash flow / savings summary.",
    instruction=INSTRUCTION,
    tools=[get_dashboard_status, update_dashboard_field, get_profile, search_financial_advice],
    before_agent_callback=[ask_missing_dashboard_fields, ask_spending_breakdown],
    after_model_callback=[ground_if_ungrounded, strip_unrequested_citations],
    after_tool_callback=track_tool_usage,
)
