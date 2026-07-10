from typing import AsyncGenerator

from google.adk.agents import Agent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.tools.agent_tool import AgentTool
from pydantic import ConfigDict

from .config import get_coordinator_model
from .dashboard_store import load_dashboard, spending_breakdown_pending
from .dashboard_store import missing_required as dashboard_missing_required
from .profile_store import load_profile
from .profile_store import missing_required as profile_missing_required
from .sub_agents.financial_dashboard.financial_dashboard_agent import financial_dashboard
from .sub_agents.market_research.market_research_agent import market_research
from .sub_agents.portfolio_analyst.portfolio_analyst_agent import portfolio_analyst
from .sub_agents.risk_profiler.risk_profiler_agent import risk_profiler
from .sub_agents.user_profile.user_profile_agent import user_profile

ROUTER_INSTRUCTION = """\
You are the Financial Coordinator, the warm and welcoming entry point for
a small team of specialist agents that help users think about personal
finance and investing. You are not a licensed financial advisor and must
never claim to be one.

Route each request to whichever specialist fits it, then use its returned
result to write your reply:
- risk_profiler: when the user wants help figuring out their risk
  tolerance, time horizon, or general investing profile.
- portfolio_analyst: when the user wants help with asset allocation,
  diversification, or evaluating a specific set of holdings.
- market_research: when the user asks about a specific company, ticker,
  price, or market trend — AND also for anything about the broader market
  with no specific ticker: recaps, overviews, "how's the market doing",
  or current/past trends in major indices.

NEVER answer a market-data or market-trend question — ticker-specific or
broad/index-level — from your own memory, even if you think you know the
number. Market data changes constantly and your training data is stale;
always call market_research for it. Always call the relevant specialist
tool rather than answering from memory when the request matches one of
the areas above. If a request spans
multiple areas, call them in this order — risk_profiler, then
portfolio_analyst, then market_research — since risk tolerance should
inform allocation advice, and let the user ask follow-ups to reach the
rest rather than calling all three in one turn.

Tone: warm and approachable, but stay focused and concise — get to the
point, don't pad with generic AI-assistant preamble, and don't restate
your own purpose. Handle greetings yourself, briefly. Always keep
responses grounded, avoid guaranteeing returns, and remind the user this
is educational information, not licensed financial advice.
"""

specialist_router = Agent(
    name="specialist_router",
    model=get_coordinator_model(),
    description="Routes investing, risk, and portfolio questions to the right specialist.",
    instruction=ROUTER_INSTRUCTION,
    tools=[
        AgentTool(agent=risk_profiler),
        AgentTool(agent=portfolio_analyst),
        AgentTool(agent=market_research),
    ],
)


class FinancialCoordinator(BaseAgent):
    """Deterministically gates on profile, then dashboard, completeness.

    Both checks are done in plain Python (load_profile/load_dashboard +
    missing_required) rather than left to the LLM to remember to check —
    the earlier instruction-based version of the profile gate ("call
    get_profile first, then branch") turned out unreliable on these small
    local models, which skipped straight to the obviously-matching
    specialist tool instead of following the procedure.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    profile_agent: Agent
    dashboard_agent: Agent
    router_agent: Agent

    def __init__(
        self,
        name: str,
        profile_agent: Agent,
        dashboard_agent: Agent,
        router_agent: Agent,
        **kwargs,
    ):
        super().__init__(
            name=name,
            profile_agent=profile_agent,
            dashboard_agent=dashboard_agent,
            router_agent=router_agent,
            sub_agents=[profile_agent, dashboard_agent, router_agent],
            **kwargs,
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        dashboard = load_dashboard()
        if profile_missing_required(load_profile()):
            chosen_agent = self.profile_agent
        elif dashboard_missing_required(dashboard) or spending_breakdown_pending(dashboard):
            # spending_breakdown_pending covers the optional, proactively-
            # offered spending-category follow-up: required fields alone
            # being done isn't enough to move on to specialist_router yet,
            # or a returning user whose required fields were already
            # complete from a prior session would skip financial_dashboard
            # entirely and the offer would never get a chance to run.
            chosen_agent = self.dashboard_agent
        else:
            chosen_agent = self.router_agent
        async for event in chosen_agent.run_async(ctx):
            yield event


root_agent = FinancialCoordinator(
    name="financial_coordinator",
    description="Coordinates onboarding, the financial dashboard, and specialist agents.",
    profile_agent=user_profile,
    dashboard_agent=financial_dashboard,
    router_agent=specialist_router,
)
