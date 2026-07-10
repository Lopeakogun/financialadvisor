import re

from google.adk.agents import Agent
from google.genai import types

from ...config import get_model
from ...profile_store import load_profile, missing_required
from ...profile_store import set_field as set_profile_field
from ...tools.profile_tools import get_profile, update_profile_field

ASK_NAME_MESSAGE = "Hi! Before we get started, what's your first name?"

# Bounded, well-known vocabularies — reliably keyword-matchable, unlike
# genuinely open-ended fields. Order matters within each list: first
# match wins, so put more specific patterns first.
#
# Two tiers per field: STRICT patterns require enough surrounding context
# to be confident the text is actually answering that question (used in
# the fallback whole-message scan, run on every subsequent turn — bare
# words there are dangerous: "conservative estimate", "single source of
# income", and "I experienced a loss" all matched their field's LOOSE
# pattern in testing despite having nothing to do with risk tolerance,
# household status, or investing experience, silently locking in a wrong
# value and making the real question never get (re-)asked). LOOSE
# patterns (STRICT + bare words) are only used in positional mode, where
# the segment is already confirmed to correspond to that specific field
# by position, so a bare word is safe there.
_EXPERIENCE_KEYWORDS_STRICT = [
    (re.compile(r"\bnever\s+(?:have\s+)?invest", re.I), "No experience — never invested before"),
    (re.compile(r"\bno\s+(?:investing\s+)?experience\b", re.I), "No experience"),
    (re.compile(r"\bbeginner\s*investor\b|\bnew\s+to\s+invest", re.I), "Beginner"),
    (re.compile(r"\bsome\s+(?:investing\s+)?experience\b|\bintermediate\s+investor\b", re.I), "Some experience"),
    (re.compile(r"\bexperienced\s+investor\b|\binvesting\s+veteran\b|\bseasoned\s+investor\b", re.I), "Experienced"),
]
_EXPERIENCE_KEYWORDS_LOOSE = _EXPERIENCE_KEYWORDS_STRICT + [
    (re.compile(r"\bbeginner\b", re.I), "Beginner"),
    (re.compile(r"\bintermediate\b", re.I), "Some experience"),
    (re.compile(r"\bexperienced\b|\bveteran\b|\bseasoned\b|\badvanced\b", re.I), "Experienced"),
]

_RISK_TOLERANCE_KEYWORDS_STRICT = [
    (re.compile(r"\brisk[\s-]?avers\w*\b", re.I), "Conservative"),
    (
        re.compile(r"\b(?:i'?m|i am)\s+(?:pretty\s+|very\s+|quite\s+)?conservative\b|\bconservative\s+(?:investor|risk|approach)", re.I),
        "Conservative",
    ),
    (
        re.compile(r"\bhigh[\s-]?risk\b|\b(?:i'?m|i am)\s+(?:pretty\s+|very\s+|quite\s+)?aggressive\b|\baggressive\s+(?:investor|risk|approach)", re.I),
        "Aggressive",
    ),
    (
        re.compile(r"\bmedium[\s-]?risk\b|\b(?:i'?m|i am)\s+(?:pretty\s+|somewhat\s+)?moderate\b|\bmoderate\s+(?:investor|risk)|\bbalanced\s+(?:approach|risk)", re.I),
        "Moderate",
    ),
]
_RISK_TOLERANCE_KEYWORDS_LOOSE = _RISK_TOLERANCE_KEYWORDS_STRICT + [
    (re.compile(r"\bconservative\b", re.I), "Conservative"),
    (re.compile(r"\baggressive\b", re.I), "Aggressive"),
    (re.compile(r"\bmoderate\b|\bbalanced\b", re.I), "Moderate"),
]

_HOUSEHOLD_KEYWORDS_STRICT = [
    (re.compile(r"\bdivorced\b", re.I), "Divorced"),
    (re.compile(r"\bwidow", re.I), "Widowed"),
    (re.compile(r"\bmarried\b", re.I), "Married"),
    (re.compile(r"\bi\s+have\s+(?:kids|children|dependents)\b", re.I), "Has dependents"),
    (re.compile(r"\b(?:i'?m|i am)\s+single\b|\bsingle\s+(?:person|individual|adult)\b", re.I), "Single"),
]
_HOUSEHOLD_KEYWORDS_LOOSE = _HOUSEHOLD_KEYWORDS_STRICT + [
    (re.compile(r"\bsingle\b", re.I), "Single"),
    (re.compile(r"\b(?:with\s+)?(?:kids|children|dependents)\b", re.I), "Has dependents"),
]

_US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina",
    "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
    "Washington DC", "District of Columbia",
]
_COUNTRIES = [
    "United States", "USA", "U.S.", "Canada", "United Kingdom", "UK", "Australia",
    "India", "Germany", "France", "Ireland", "New Zealand", "Singapore", "Mexico",
]
_LOCATION_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _US_STATES + _COUNTRIES) + r")\b", re.I
)

_TIME_HORIZON_RE = re.compile(r"\b(\d{1,3})[\s-]*(?:\+\s*)?(?:years?|yrs?)\b", re.I)
_TIME_HORIZON_WORDS_RE = re.compile(r"\b(long[\s-]?term|short[\s-]?term|retirement)\b", re.I)

_INCOME_AMOUNT_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?\s?[kK]?|\b\d{2,3}[kK]\b")
_INCOME_STABILITY_RE = re.compile(
    r"\b(stable|unstable|variable|fluctuat\w*|secure|steady|volatile|irregular)\b", re.I
)

_AGE_STRICT_RE = re.compile(
    r"\b(?:i'?m|i am|age[d]?)\s*:?\s*(\d{1,3})\b|\b(\d{1,3})\s*(?:years?\s*old|y\.?o\.?)\b", re.I
)
_AGE_LOOSE_RE = re.compile(r"^\D{0,10}(\d{1,3})\b")


def _keyword_match(patterns: list[tuple[re.Pattern, str]], text: str) -> str | None:
    for pattern, value in patterns:
        if pattern.search(text):
            return value
    return None


def _extract_age(text: str, *, positional: bool) -> str | None:
    m = _AGE_STRICT_RE.search(text)
    if m:
        n = int(m.group(1) or m.group(2))
        if 13 <= n <= 110:
            return str(n)
    if positional:
        m = _AGE_LOOSE_RE.match(text.strip())
        if m and 13 <= int(m.group(1)) <= 110:
            return m.group(1)
    return None


def _extract_income(text: str, *, positional: bool) -> str | None:
    amount_m = _INCOME_AMOUNT_RE.search(text)
    stability_m = _INCOME_STABILITY_RE.search(text)
    if amount_m and stability_m:
        return f"{amount_m.group(0).strip()}, {stability_m.group(1).lower()}"
    if amount_m:
        return amount_m.group(0).strip()
    if positional and stability_m:
        return text.strip()  # couldn't find a clean amount, but this segment was clearly meant as income
    return None


def _extract_time_horizon(text: str) -> str | None:
    m = _TIME_HORIZON_RE.search(text)
    if m:
        return f"{m.group(1)} years"
    m = _TIME_HORIZON_WORDS_RE.search(text)
    if m:
        return m.group(1)
    return None


def _extract_location(text: str) -> str | None:
    m = _LOCATION_RE.search(text)
    return m.group(1) if m else None


# field -> extractor. Extractors that don't need to know if they're
# looking at an isolated (positional) segment vs. the whole free-form
# reply take just `text`; the two that benefit from knowing (age, income)
# take `positional` too, since a bare number is only safe to trust as the
# answer when it's already isolated to a known field slot.
_EXTRACTORS = {
    "age": lambda text, positional: _extract_age(text, positional=positional),
    "household_status": lambda text, positional: _keyword_match(
        _HOUSEHOLD_KEYWORDS_LOOSE if positional else _HOUSEHOLD_KEYWORDS_STRICT, text
    ),
    "income": lambda text, positional: _extract_income(text, positional=positional),
    "location": lambda text, positional: _extract_location(text),
    "risk_tolerance": lambda text, positional: _keyword_match(
        _RISK_TOLERANCE_KEYWORDS_LOOSE if positional else _RISK_TOLERANCE_KEYWORDS_STRICT, text
    ),
    "investment_experience": lambda text, positional: _keyword_match(
        _EXPERIENCE_KEYWORDS_LOOSE if positional else _EXPERIENCE_KEYWORDS_STRICT, text
    ),
    "time_horizon": lambda text, positional: _extract_time_horizon(text),
}


def capture_profile_fields(callback_context) -> None:
    """Deterministically save as many required fields as can be reliably parsed.

    Live testing (and a user report) showed the model's own tool-calling
    silently drops fields from a batch reply — it references them
    correctly in its own response text ("your age as 30... income of
    $95,000... moderate risk tolerance...") without ever calling
    update_profile_field, so they stay empty and get re-asked later. This
    runs as a side effect before the LLM turn, in addition to (not
    instead of) the LLM's own attempt:

    1. Split the reply on commas/semicolons/newlines. If the segment count
       exactly matches the number of currently-missing fields (in the
       order they were asked), map segments to fields positionally and
       extract each — this is the common case, since the agent always
       asks in a fixed, numbered order and users tend to answer in the
       same order. A segment that doesn't parse cleanly is still saved
       as-is (better than losing it) since positional matching already
       gives high confidence about which field it answers.
    2. Otherwise (reply doesn't cleanly segment — combined into flowing
       prose, or only some fields answered), fall back to scanning the
       whole reply for each still-missing field's pattern. Only fields
       with a clean, context-anchored match get saved here — no bare
       numbers or other low-confidence guesses, since a wrong silent save
       is worse than an occasional re-ask.

    Best-effort, not full NLU: still won't catch every phrasing.
    """
    profile = load_profile()
    if not profile.get("name"):
        return None  # ask_name_first handles this turn instead

    missing = [f for f in missing_required(profile) if f != "name"]
    if not missing:
        return None

    user_content = callback_context.user_content
    if not user_content or not user_content.parts:
        return None
    reply_text = "".join(part.text or "" for part in user_content.parts).strip()
    if not reply_text:
        return None

    segments = [s.strip() for s in re.split(r"[,;\n]+", reply_text) if s.strip()]
    if len(segments) == len(missing):
        for field, segment in zip(missing, segments):
            value = _EXTRACTORS[field](segment, True) or segment
            set_profile_field(field, value)
        return None

    for field in missing:
        value = _EXTRACTORS[field](reply_text, False)
        if value:
            set_profile_field(field, value)

    return None


def ask_name_first(callback_context) -> types.Content | None:
    """Deterministically ask for, then capture, the user's name.

    Small local models proved unreliable here in two ways: the conditional
    "if name is missing, ask ONLY for it; otherwise do X" branch (they'd
    hallucinate a name or skip straight past the question), and separately,
    actually calling update_profile_field to save the name once given (it
    would use the name fluently in its reply text without ever calling the
    tool). Both are handled in plain Python instead of trusted to the LLM:
    ask with a canned question the first time, then treat the very next
    user message as the name directly (it's a direct reply to a name
    question, so no NLU is needed) and save it before the LLM even runs.
    """
    if load_profile().get("name"):
        return None
    if not callback_context.state.get("asked_for_name"):
        callback_context.state["asked_for_name"] = True
        return types.Content(role="model", parts=[types.Part(text=ASK_NAME_MESSAGE)])

    user_content = callback_context.user_content
    if user_content and user_content.parts:
        reply_text = "".join(part.text or "" for part in user_content.parts).strip()
        if reply_text:
            set_profile_field("name", reply_text.split("\n")[0][:50])
    return None


INSTRUCTION = """\
You are the friendly onboarding guide for a personal finance app. Your job
is to get to know the user a little before the rest of the team can give
them tailored advice.

Tone: warm and conversational, like a friendly chat, not a form. Stay
focused and brief — 1-3 sentences per turn. Never explain what kind of AI
you are or restate your own purpose; just ask the next question.

- Start by calling get_profile to see what's already known — their name
  is always already saved by the time you're asked anything, so never ask
  for it yourself. Never re-ask a question whose answer is already saved.
- Greet them using their saved name, then ask all other still-missing
  required fields in ONE message, as a short numbered list (so they can
  answer everything at once instead of a slow back-and-forth). Required
  fields, in this order: age, household_status, income, location,
  risk_tolerance, investment_experience, time_horizon. Format that list
  like:
  1. How old are you?
  2. What's your household situation (single, married, dependents)?
  3. What's your income and how stable is it?
  ...(continue for the rest of the still-missing required fields)
- When the user replies, match their answers to the fields as best you
  can (they may answer out of order, skip some, or combine answers into
  one sentence) and call update_profile_field once per field you can
  confidently identify. Then call get_profile again — if anything is
  still missing, ask again for just those remaining items as a short list,
  don't restart the whole list.
- If it's not obvious why you're asking, briefly explain in a few words
  (e.g. "so I can gauge how much risk makes sense for you").
- Stay focused on onboarding. If the user asks something else first (a
  stock price, investment advice, etc.), give a one-sentence friendly
  acknowledgment and steer back to finishing their profile — don't try to
  answer it yourself and don't apologize at length.
- Once every required field is saved, tell the user their profile is set
  up and they can now move on to the next step. End that message (only
  that one) with a short disclaimer that this is educational information,
  not licensed financial advice — no need to repeat it every turn before
  that.
- You can also collect the optional fields (expenses, assets, liabilities,
  tax_bracket, employment_status, retirement_goals, major_life_goals,
  liquidity_needs) once the required ones are done, but don't block on
  them or make the user feel like it's mandatory.
"""

user_profile = Agent(
    name="user_profile",
    model=get_model(),
    description="Friendly onboarding agent that conversationally collects and stores the user's financial profile.",
    instruction=INSTRUCTION,
    tools=[get_profile, update_profile_field],
    before_agent_callback=[ask_name_first, capture_profile_fields],
)
