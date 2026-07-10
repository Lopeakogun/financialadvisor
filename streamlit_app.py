import asyncio
import os

import streamlit as st
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

load_dotenv("financial_advisor/.env")

from financial_advisor.agent import root_agent  # noqa: E402
from financial_advisor.dashboard_store import DASHBOARD_SCHEMA  # noqa: E402
from financial_advisor.dashboard_store import compute_guidance as compute_dashboard_guidance  # noqa: E402
from financial_advisor.dashboard_store import compute_spending_breakdown  # noqa: E402
from financial_advisor.dashboard_store import compute_summary as compute_dashboard_summary  # noqa: E402
from financial_advisor.dashboard_store import load_dashboard  # noqa: E402
from financial_advisor.dashboard_store import missing_required as dashboard_missing_required  # noqa: E402
from financial_advisor.profile_store import DATA_DIR, PROFILE_SCHEMA, load_profile  # noqa: E402
from financial_advisor.profile_store import missing_required as profile_missing_required  # noqa: E402

APP_NAME = "financial_advisor"
DB_PATH = DATA_DIR / "sessions.db"

st.set_page_config(page_title="Financial Advisor", page_icon="\U0001f4b0")
st.title("\U0001f4b0 Financial Advisor")
st.caption(
    "Multi-agent demo (onboarding, financial dashboard, risk profiler, "
    "portfolio analyst, market research) running on a local Gemma model "
    "via Ollama. Educational use only — not licensed financial advice."
)


def _completion_widget(label: str, schema: dict, missing: list) -> None:
    total_required = sum(1 for spec in schema.values() if spec["required"])
    done = total_required - len(missing)
    st.subheader(label)
    st.progress(done / total_required if total_required else 1.0)
    if missing:
        st.caption(f"{done}/{total_required} required fields collected")
        st.caption("Still needed: " + ", ".join(schema[f]["label"] for f in missing))
    else:
        st.caption("Complete.")


profile = load_profile()
profile_missing = profile_missing_required(profile)
# The saved profile's name doubles as the identity key for persisted chat
# history, so returning users (same name) automatically resume their last
# conversation. Before a name is known, everything lives under "guest".
USER_ID = profile.get("name") or "guest"

with st.sidebar:
    _completion_widget("Your profile", PROFILE_SCHEMA, profile_missing)

    if not profile_missing:
        st.divider()
        dashboard = load_dashboard()
        dash_missing = dashboard_missing_required(dashboard)
        _completion_widget("Financial dashboard", DASHBOARD_SCHEMA, dash_missing)

        if not dash_missing:
            summary = compute_dashboard_summary(dashboard)
            guidance = compute_dashboard_guidance(dashboard)
            col1, col2 = st.columns(2)
            if "net_worth" in summary:
                col1.metric("Net worth", f"${summary['net_worth']:,.0f}")
            if "monthly_cash_flow" in summary:
                col2.metric("Monthly cash flow", f"${summary['monthly_cash_flow']:,.0f}")
            if "savings_rate_pct" in summary:
                col1.metric("Savings rate", f"{summary['savings_rate_pct']:.1f}%")
            if "emergency_fund_months" in summary:
                col2.metric("Emergency fund", f"{summary['emergency_fund_months']:.1f} mo")
            if "recommended_investing_amount" in guidance:
                col1.metric(
                    "Suggested investing",
                    f"${guidance['recommended_investing_amount']:,.0f}/mo",
                    help=f"~{guidance['recommended_investing_pct']:.1f}% of monthly income",
                )
            if "emergency_fund_monthly_contribution" in guidance:
                col2.metric(
                    "Suggested to emergency fund",
                    f"${guidance['emergency_fund_monthly_contribution']:,.0f}/mo",
                    help=f"~{guidance['emergency_fund_pct_of_income']:.1f}% of monthly income",
                )

            breakdown = compute_spending_breakdown(dashboard)
            if breakdown:
                with st.expander("Spending breakdown", expanded=False):
                    for field, amount in breakdown["categories"].items():
                        label = DASHBOARD_SCHEMA[field]["label"].split(" (")[0]
                        pct = breakdown["categories_pct_of_income"][field]
                        st.caption(f"{label}: ${amount:,.0f}/mo ({pct:.1f}% of income)")
                    if breakdown.get("housing_over_guideline"):
                        st.caption(f"⚠️ Housing is {breakdown['housing_pct_of_income']:.1f}% of income (guideline: ≤30%)")
                    if breakdown.get("debt_to_income_over_guideline"):
                        st.caption(f"⚠️ Debt-to-income is {breakdown['debt_to_income_pct']:.1f}% (guideline: ≤36%)")


@st.cache_resource
def get_session_service() -> DatabaseSessionService:
    os.makedirs(DATA_DIR, exist_ok=True)
    return DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{DB_PATH}")


session_service = get_session_service()
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)


async def _resolve_session():
    """Resume the most recent saved session for USER_ID, or start a new one."""
    existing = await session_service.list_sessions(app_name=APP_NAME, user_id=USER_ID)
    if existing.sessions:
        # list_sessions returns lightweight summaries with no events; fetch
        # the full session (with event history) for the one we're resuming.
        latest = max(existing.sessions, key=lambda s: s.last_update_time)
        return await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=latest.id
        )
    return await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)


if st.session_state.get("user_id") != USER_ID:
    session = asyncio.run(_resolve_session())
    st.session_state.user_id = USER_ID
    st.session_state.session_id = session.id
    st.session_state.messages = []
    for event in session.events:
        if event.content and event.content.parts and event.author:
            text = "".join(part.text or "" for part in event.content.parts)
            if text:
                role = "user" if event.author == "user" else "assistant"
                st.session_state.messages.append({"role": role, "content": text})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


async def run_agent(text: str) -> str:
    content = types.Content(role="user", parts=[types.Part(text=text)])
    final_text = "(no response)"
    async for event in runner.run_async(
        user_id=USER_ID, session_id=st.session_state.session_id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text or final_text
    return final_text


if prompt := st.chat_input("Ask about investing, risk, or a stock..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response_text = asyncio.run(run_agent(prompt))
            except Exception as e:
                response_text = (
                    f"Error talking to the agent: {e}\n\n"
                    "Make sure Ollama is running locally (`ollama serve`) "
                    "and the model in financial_advisor/.env has been pulled."
                )
        st.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
