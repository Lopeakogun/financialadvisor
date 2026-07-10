"""yfinance-backed tools exposed to ADK agents.

Each function's type hints and docstring are used by ADK to auto-generate
the tool schema the LLM sees, so keep both accurate.
"""

import yfinance as yf

MARKET_INDICES = {
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "Nasdaq": "^IXIC",
    "Volatility Index (VIX)": "^VIX",
}


def get_stock_price(ticker: str) -> dict:
    """Get the latest trading price and basic quote info for a stock ticker.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL", "MSFT", "VTI".

    Returns:
        A dict with price, currency, day_high, day_low, volume and
        market_cap, or a dict with an "error" key if the ticker is invalid
        or the lookup fails.
    """
    try:
        info = yf.Ticker(ticker).fast_info
        return {
            "ticker": ticker.upper(),
            "price": info.last_price,
            "currency": info.currency,
            "day_high": info.day_high,
            "day_low": info.day_low,
            "volume": info.last_volume,
            "market_cap": info.market_cap,
        }
    except Exception as e:
        return {"error": f"Could not fetch price for '{ticker}': {e}"}


def get_stock_fundamentals(ticker: str) -> dict:
    """Get key fundamental metrics for a stock ticker.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL", "MSFT", "VTI".

    Returns:
        A dict with sector, industry, trailing_pe, forward_pe,
        dividend_yield, beta, and 52-week high/low, or a dict with an
        "error" key if the lookup fails.
    """
    try:
        info = yf.Ticker(ticker).get_info()
        return {
            "ticker": ticker.upper(),
            "name": info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        }
    except Exception as e:
        return {"error": f"Could not fetch fundamentals for '{ticker}': {e}"}


def get_historical_performance(ticker: str, period: str = "6mo") -> dict:
    """Get historical price performance and trend for a stock over a period.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL", "MSFT", "VTI".
        period: One of "1mo", "3mo", "6mo", "1y", "5y", "max".

    Returns:
        A dict with start_price, end_price, pct_change, period_high,
        period_low, and a "trend" label ("uptrend", "downtrend", or
        "flat", based on pct_change), or a dict with an "error" key if
        the lookup fails.
    """
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return {"error": f"No historical data found for '{ticker}'."}
        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])
        pct_change = (end_price - start_price) / start_price * 100
        trend = "uptrend" if pct_change > 3 else "downtrend" if pct_change < -3 else "flat"
        return {
            "ticker": ticker.upper(),
            "period": period,
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "pct_change": round(pct_change, 2),
            "period_high": round(float(hist["High"].max()), 2),
            "period_low": round(float(hist["Low"].min()), 2),
            "trend": trend,
        }
    except Exception as e:
        return {"error": f"Could not fetch history for '{ticker}': {e}"}


def get_market_overview(period: str = "1d") -> dict:
    """Get a snapshot of major U.S. market indices — for recaps, overviews, and trend questions.

    Covers the S&P 500, Dow Jones, Nasdaq, and the VIX (volatility index).
    Use period="1d" for a same-day/latest-close recap, or a longer period
    for broader trend context.

    Args:
        period: One of "1d", "5d", "1mo", "3mo", "6mo", "1y".

    Returns:
        A dict with 'period' and 'indices' (each index name mapped to its
        latest level and pct_change over the period), or a dict with an
        "error" key if no index data could be fetched.
    """
    try:
        fetch_period = "5d" if period == "1d" else period
        indices = {}
        for name, ticker in MARKET_INDICES.items():
            hist = yf.Ticker(ticker).history(period=fetch_period)
            if hist.empty:
                continue
            if period == "1d" and len(hist) > 1:
                start_price = float(hist["Close"].iloc[-2])
            else:
                start_price = float(hist["Close"].iloc[0])
            end_price = float(hist["Close"].iloc[-1])
            pct_change = (end_price - start_price) / start_price * 100
            indices[name] = {"level": round(end_price, 2), "pct_change": round(pct_change, 2)}
        if not indices:
            return {"error": "Could not fetch market index data."}
        return {"period": period, "indices": indices}
    except Exception as e:
        return {"error": f"Could not fetch market overview: {e}"}
