"""Curated, attributed personal-finance principles from well-known books/figures.

Hand-authored and paraphrased (not verbatim quotes) so agents can ground
advice in real, named sources instead of relying on the LLM's own
training-data recall — which this project has repeatedly found fabricates
specifics (numbers, quotes, attributions) when not given real content
directly. See dashboard_store.py's compute_guidance for the same pattern
applied to budgeting math.
"""

FINANCIAL_WISDOM = {
    "emergency_fund": [
        {
            "source": "Dave Ramsey, The Total Money Makeover",
            "principle": (
                "Start with a small $1,000 starter emergency fund before aggressively "
                "paying off debt, then build it to 3-6 months of expenses once debt-free."
            ),
        },
        {
            "source": "Suze Orman",
            "principle": (
                "Prioritize an emergency fund and being debt-free before taking on big "
                "purchases — \"people first, then money, then things.\""
            ),
        },
    ],
    "debt": [
        {
            "source": "Dave Ramsey, The Total Money Makeover",
            "principle": (
                "The \"debt snowball\" method: pay off debts smallest balance first for "
                "psychological wins and momentum, regardless of interest rate."
            ),
        },
    ],
    "budgeting": [
        {
            "source": "Elizabeth Warren & Amelia Warren Tyagi, All Your Worth",
            "principle": (
                "The 50/30/20 rule: roughly 50% of income to needs, 30% to wants, "
                "20% to savings and debt repayment."
            ),
        },
        {
            "source": "George S. Clason, The Richest Man in Babylon",
            "principle": "\"Pay yourself first\" — save at least 10% of income before spending on anything else.",
        },
        {
            "source": "Ramit Sethi, I Will Teach You To Be Rich",
            "principle": (
                "Automate savings and investing so good habits happen by default, "
                "rather than relying on willpower every month."
            ),
        },
    ],
    "investing_basics": [
        {
            "source": "John C. Bogle, founder of Vanguard",
            "principle": (
                "For most investors, low-cost, broad-market index funds beat trying to "
                "pick individual winning stocks over the long run."
            ),
        },
        {
            "source": "Warren Buffett",
            "principle": (
                "Has repeatedly recommended that most non-professional investors put "
                "long-term savings into a low-cost S&P 500 index fund."
            ),
        },
        {
            "source": "Benjamin Graham, The Intelligent Investor",
            "principle": (
                "Favor a \"margin of safety\" and long-term discipline over speculation "
                "or trying to time the market."
            ),
        },
    ],
    "diversification": [
        {
            "source": "Bogleheads philosophy (inspired by John C. Bogle)",
            "principle": (
                "A simple, diversified \"three-fund portfolio\" (total US stock, total "
                "international stock, total bonds) is a common low-maintenance approach."
            ),
        },
    ],
    "retirement": [
        {
            "source": "The 4% rule (William Bengen's research / the Trinity Study)",
            "principle": (
                "Historically, withdrawing about 4% of a diversified portfolio in the "
                "first year of retirement, then adjusting for inflation, has had a high "
                "likelihood of lasting 30 years."
            ),
        },
        {
            "source": "Dave Ramsey, The Total Money Makeover",
            "principle": (
                "Once debt-free with a full emergency fund, aim to invest around 15% of "
                "household income for retirement."
            ),
        },
    ],
    "assets_vs_liabilities": [
        {
            "source": "Robert Kiyosaki, Rich Dad Poor Dad",
            "principle": (
                "Distinguishes assets (put money in your pocket) from liabilities (take "
                "money out), and emphasizes acquiring income-generating assets."
            ),
        },
    ],
    "housing_budget": [
        {
            "source": "The 30% rule (U.S. federal housing affordability standard)",
            "principle": (
                "Keeping housing costs (rent or mortgage) at or under about 30% of gross "
                "income is a long-standing affordability benchmark."
            ),
        },
    ],
    "debt_to_income": [
        {
            "source": "Conventional mortgage underwriting guidelines",
            "principle": (
                "Lenders commonly look for a debt-to-income ratio — housing plus other "
                "debt payments, divided by gross income — at or under about 36%."
            ),
        },
    ],
}


def get_wisdom(topic: str | None = None) -> dict[str, list[dict[str, str]]]:
    """Return curated wisdom entries for a topic, or all topics if none given."""
    if topic is None:
        return FINANCIAL_WISDOM
    return {topic: FINANCIAL_WISDOM[topic]} if topic in FINANCIAL_WISDOM else {}
