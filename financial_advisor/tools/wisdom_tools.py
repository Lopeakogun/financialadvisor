"""ADK tool exposing the curated financial-wisdom store to agents."""

from ..wisdom_store import FINANCIAL_WISDOM, get_wisdom


def get_financial_wisdom(topic: str) -> dict:
    """Get curated, attributed personal-finance principles for a topic.

    Use this to ground advice in real, named sources (books/well-known
    figures) instead of citing anything from memory.

    Args:
        topic: One of: emergency_fund, debt, budgeting, investing_basics,
            diversification, retirement, assets_vs_liabilities.

    Returns:
        A dict with 'topic' and 'wisdom' (list of {source, principle}
        entries), or a dict with an 'error' key and the list of valid
        topics if the topic isn't recognized.
    """
    result = get_wisdom(topic)
    if not result:
        return {"error": f"Unknown topic '{topic}'. Valid topics: {', '.join(FINANCIAL_WISDOM)}"}
    return {"topic": topic, "wisdom": result[topic]}
