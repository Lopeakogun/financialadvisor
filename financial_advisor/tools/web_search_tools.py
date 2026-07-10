"""Web search for financial questions the app can't otherwise ground.

No API key (uses DuckDuckGo via ddgs, consistent with the rest of this
project's "no credentials anywhere" design), restricted to a curated
allowlist of reputable financial publishers/institutions so "trusted
reliable financial sources" is a concrete, auditable filter rather than
trusting whatever a search engine happens to return.
"""

import re
import uuid
from urllib.parse import urlparse

from ddgs import DDGS
from google.adk.models.llm_response import LlmResponse
from google.genai import types

TRUSTED_FINANCIAL_DOMAINS = {
    "investopedia.com",
    "nerdwallet.com",
    "fool.com",
    "morningstar.com",
    "bogleheads.org",
    "kiplinger.com",
    "forbes.com",
    "cnbc.com",
    "marketwatch.com",
    "vanguard.com",
    "fidelity.com",
    "schwab.com",
    "ramseysolutions.com",
    "irs.gov",
    "sec.gov",
    "consumerfinance.gov",
    "usa.gov",
}


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def search_financial_advice(query: str) -> dict:
    """Search reputable financial websites for information to ground an answer in.

    Use this when a question needs a real fact, statistic, or named
    source you're not certain of — never state a specific claim or
    attribution from memory alone. Returns short snippets and URLs to
    cite, not full articles.

    Args:
        query: A focused search query, e.g. "recommended emergency fund
            months of expenses" or "Dave Ramsey debt snowball method".

    Returns:
        A dict with 'results' (list of {title, url, snippet}, from
        trusted financial sites when available) and 'filtered' (True if
        results were restricted to trusted domains, False if no trusted
        results were found and these are unfiltered general results
        instead — treat those more cautiously). Returns a dict with an
        'error' key if the search itself fails.
    """
    try:
        raw_results = DDGS().text(query, max_results=8)
    except Exception as e:
        return {"error": f"Search failed: {e}"}

    if not raw_results:
        return {"results": [], "filtered": True}

    def to_entry(r: dict) -> dict:
        return {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}

    trusted = [to_entry(r) for r in raw_results if _domain(r.get("href", "")) in TRUSTED_FINANCIAL_DOMAINS]
    if trusted:
        return {"results": trusted[:5], "filtered": True}

    return {"results": [to_entry(r) for r in raw_results[:5]], "filtered": False}


_MIN_DECLARATIVE_CHARS = 60


def _declarative_length(text: str) -> int:
    """Total length of sentences that aren't questions.

    Used instead of a naive `text.endswith("?")` check, which lets a long
    ungrounded claim through un-forced whenever the model tacks on a
    trailing offer like "Would you like me to look that up?" — observed
    live: a multi-paragraph, fully ungrounded answer about portfolio
    rebalancing slipped through this way. Measuring declarative content
    specifically catches that regardless of how the response ends.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return sum(len(s) for s in sentences if s and not s.rstrip().endswith("?"))


def ground_if_ungrounded(callback_context, llm_response: LlmResponse) -> LlmResponse | None:
    """Force a real search when the model answers substantively with no tool call.

    This model reliably calls tools when a single obvious action is
    needed, and reliably synthesizes a correct answer FROM a given tool
    result — but often skips calling a tool at all when the easier move
    is to just answer from memory, even when explicitly instructed to
    always ground claims first (see README "Known limitations" for the
    evidence: e.g. asked "how often should I rebalance my portfolio?" it
    answered fluently and plausibly with zero tool calls).

    Rather than keep trusting that instruction, this detects the skip
    after the fact and forces it: inject a synthetic function_call to
    search_financial_advice using the user's own question as the query.
    ADK treats a callback-returned LlmResponse exactly like a real model
    response, so this gets executed and looped back to the model with
    real results — the same reliable "present the numbers I was given"
    behavior this project has seen work throughout, just triggered by
    Python instead of left to the model's judgment.

    Use as an agent's `after_model_callback`.
    """
    if not llm_response.content or not llm_response.content.parts:
        return None

    called_key = f"tool_called_{callback_context.invocation_id}"
    forced_key = f"forced_search_{callback_context.invocation_id}"

    if any(part.function_call for part in llm_response.content.parts):
        callback_context.state[called_key] = True
        return None  # this response is itself a real tool/delegation call

    # No function call in this response. If a tool was already called
    # earlier in this same turn (e.g. get_stock_price, then this response
    # presents the result), this is a legitimately grounded final answer,
    # not a skip — leave it alone.
    if callback_context.state.get(called_key):
        return None

    # Only force one search per user turn — this callback also fires on
    # the follow-up response after the forced search returns, and that
    # one (presenting the real results) must be allowed through.
    if callback_context.state.get(forced_key):
        return None

    text = "".join(part.text or "" for part in llm_response.content.parts).strip()
    if _declarative_length(text) < _MIN_DECLARATIVE_CHARS:
        return None  # mostly/only a question — not a substantive claim to ground

    user_content = callback_context.user_content
    query = "".join(part.text or "" for part in user_content.parts) if user_content and user_content.parts else ""
    if not query:
        return None

    callback_context.state[forced_key] = True
    callback_context.state[called_key] = True

    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id=f"forced_search_{uuid.uuid4().hex[:8]}",
                        name="search_financial_advice",
                        args={"query": query},
                    )
                )
            ],
        )
    )


def track_tool_usage(tool, args, tool_context, tool_response) -> dict | None:
    """Record which tools ran this turn, for strip_unrequested_citations to check.

    Use as an agent's `after_tool_callback`.
    """
    if tool.name == "search_financial_advice":
        tool_context.state[f"search_used_{tool_context.invocation_id}"] = True
    elif tool.name == "get_financial_wisdom":
        tool_context.state[f"wisdom_used_{tool_context.invocation_id}"] = True
    return None  # don't modify the actual tool response


_ASKING_FOR_SOURCE_RE = re.compile(
    r"\b(source|sources|cite|citation|link|reference|says who|proof|evidence"
    r"|where.{0,20}(?:from|find|get)|who says)\b",
    re.IGNORECASE,
)

# Matches a citation lead-in naming a proper-noun-ish source at the start of
# a sentence — the two patterns actually observed live: "According to
# Vanguard, ..." / "Based on Vanguard's research, ..." and "Retirement
# Researcher suggests that ...".
_CITATION_LEADIN_RE = re.compile(
    r"^(?:According to|Based on|Per|As (?:reported|noted|suggested) by)\s+"
    r"[A-Z][A-Za-z0-9.&'-]*(?:\s+[A-Z][A-Za-z0-9.&'-]*){0,3}"
    r"(?:'s (?:research|data|analysis|study|findings|philosophy|approach))?"
    r"[,:]\s+"
)
_CITATION_SUBJECT_RE = re.compile(
    r"^[A-Z][A-Za-z0-9.&'-]*(?:\s+[A-Z][A-Za-z0-9.&'-]*){0,3}\s+"
    r"(?:suggests?|indicates?|recommends?|notes?|reports?|shows?)\s+(?:that\s+)?"
)


def _strip_citations(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    for sentence in sentences:
        new_sentence = _CITATION_LEADIN_RE.sub("", sentence, count=1)
        if new_sentence == sentence:
            new_sentence = _CITATION_SUBJECT_RE.sub("", sentence, count=1)
        if new_sentence != sentence and new_sentence:
            new_sentence = new_sentence[0].upper() + new_sentence[1:]
        cleaned.append(new_sentence)
    return " ".join(cleaned)


def strip_unrequested_citations(callback_context, llm_response: LlmResponse) -> LlmResponse | None:
    """Remove source-name citations the user didn't ask for.

    Instructing the model not to name sources unless asked helps a lot but
    isn't 100% reliable (observed live: it named "Vanguard" once despite
    the instruction, even though the more dangerous multi-source
    fabrication was gone). This is a best-effort text-pattern safety net,
    not a full NLP solution — it only catches the specific lead-in
    phrasings actually observed ("According to X, ...", "Based on X's
    research, ...", "X suggests that ..."), not every possible phrasing.
    Only applies when search_financial_advice (not get_financial_wisdom,
    whose citations are always wanted) ran this turn, and the user's own
    message doesn't look like it's asking for a source.

    Use as an agent's `after_model_callback`, alongside `ground_if_ungrounded`.
    """
    if not llm_response.content or not llm_response.content.parts:
        return None
    if any(part.function_call for part in llm_response.content.parts):
        return None  # not a final text response

    invocation_id = callback_context.invocation_id
    if not callback_context.state.get(f"search_used_{invocation_id}"):
        return None  # no web search this turn — nothing to strip
    if callback_context.state.get(f"wisdom_used_{invocation_id}"):
        return None  # wisdom_store citations were also used this turn — don't risk touching those

    user_content = callback_context.user_content
    user_text = "".join(part.text or "" for part in user_content.parts) if user_content and user_content.parts else ""
    if _ASKING_FOR_SOURCE_RE.search(user_text):
        return None  # they asked for it — let it through

    text = "".join(part.text or "" for part in llm_response.content.parts)
    cleaned = _strip_citations(text)
    if cleaned == text:
        return None  # nothing matched, no change needed

    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=cleaned)]))
