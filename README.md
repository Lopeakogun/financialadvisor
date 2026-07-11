# Financial Advisor (multi-agent, local Gemma)

A multi-agent financial/investment advisor built on Google's [Agent
Development Kit](https://github.com/google/adk-python) (ADK), running
entirely on a local open-source LLM (Gemma via [Ollama](https://ollama.com))
with a Streamlit chat front end.

**Educational demo only — not licensed financial advice.**

## Architecture

```
financial_advisor/
  agent.py                     root_agent: "financial_coordinator" (deterministic gate) ->
                                user_profile | financial_dashboard | specialist_router
  config.py                    builds the LiteLLM/Ollama models (coordinator + sub-agent tiers, tuned num_ctx/num_predict)
  profile_store.py             local JSON persistence + schema for the user's financial profile (name asked first)
  dashboard_store.py           local JSON persistence + schema + computed metrics (net worth, etc.) +
                                budgeting/investing guidance (50/30/20 split, emergency-fund-first investing) +
                                spending-category breakdown (housing/transportation/food/etc., with the
                                30%-housing and 36%-debt-to-income guideline checks)
  wisdom_store.py               curated, attributed personal-finance principles (Dave Ramsey, Bogle,
                                 Buffett, the 4% rule, etc.) — see "Known limitations" for why this
                                 is hand-authored content, not the LLM's own recall
  tools/
    market_data.py             yfinance-backed tools: price, fundamentals, history, market overview
    profile_tools.py           get_profile / update_profile_field tools backed by profile_store.py
    dashboard_tools.py         get_dashboard_status / update_dashboard_field, backed by dashboard_store.py
    wisdom_tools.py            get_financial_wisdom, backed by wisdom_store.py
    web_search_tools.py        search_financial_advice — no-API-key DuckDuckGo search (ddgs),
                                filtered to a trusted-domain allowlist; plus ground_if_ungrounded,
                                strip_unrequested_citations, and track_tool_usage — callbacks that
                                force real grounding and clean up citations (see "Known limitations")
  sub_agents/
    user_profile/
      user_profile_agent.py     friendly onboarding agent + deterministic "ask/capture name" callback
    financial_dashboard/
      financial_dashboard_agent.py  deterministic "ask fields / capture batch replies / show
                                     summary+guidance" callback, then a second deterministic
                                     "offer/capture/decline spending breakdown" stage, wrapping a
                                     friendly LLM for follow-ups
    risk_profiler/
      risk_profiler_agent.py    assesses risk tolerance via a few questions (can read the profile)
    portfolio_analyst/
      portfolio_analyst_agent.py  allocation / diversification analysis (uses market + profile + dashboard tools)
    market_research/
      market_research_agent.py  ticker + broad-market lookups (uses market tools)
streamlit_app.py                chat UI wrapping an ADK Runner + DatabaseSessionService (persistent, SQLite-
                                 backed, keyed by the saved profile name). Sidebar stays empty during
                                 onboarding (the chat itself asks the questions — no point spoiling them
                                 in a "still needed" list on the side) and only appears once a stage
                                 completes, showing computed dashboard metrics
```

Note the sub-agent files are named `<name>_agent.py`, not `agent.py` — `adk web`'s
dropdown discovery matches on the literal filename `agent.py` (see "Known
limitations"), so any new sub-agent module should follow the same
`<name>_agent.py` convention to avoid reappearing there.

`root_agent` is a small custom `BaseAgent` (`FinancialCoordinator`), not a
plain LLM agent. Each turn, it checks completeness directly in Python
(`profile_store.missing_required`, `dashboard_store.missing_required`, and
`dashboard_store.spending_breakdown_pending`) and deterministically hands
the whole turn to the first stage that isn't done yet:
1. `user_profile` — if required profile fields are still missing.
2. `financial_dashboard` — if the profile is done but required dashboard
   inputs (assets, liabilities, income, expenses, emergency fund) aren't,
   **or** if those are done but the optional spending-category breakdown
   hasn't been offered yet (`spending_breakdown_pending`, tracked
   persistently in `dashboard.json`, not session state — see "Known
   limitations" for why that distinction mattered).
3. `specialist_router` — once all of the above are resolved (completed or,
   for the optional breakdown, explicitly declined): an LLM agent that
   calls one of the three specialists **as a tool** (ADK's `AgentTool`),
   passing just the relevant question and getting back a finished answer.

All three of these design choices — the Python-level onboarding gates, and
`AgentTool` over ADK's native `sub_agents` conversational transfer —
replaced earlier LLM-instruction-based versions that turned out unreliable
on these small local models (they'd skip the check, or stall mid-handoff,
instead of following the intended procedure). See "Known limitations"
below for the specific reliability numbers and what's still LLM-decided
(specialist routing) vs. deterministic (the two onboarding gates).

## Setup

1. **Install Ollama and pull a tool-calling-capable Gemma model.** Stock
   `gemma3` tags in Ollama do **not** support tool calling at any size (the
   model's Ollama template has no tool-call format), which breaks both the
   market-data tools and the coordinator's specialist-calling used here. Use
   [`orieg/gemma3-tools`](https://ollama.com/orieg/gemma3-tools), a
   community QLoRA fine-tune of Gemma 3 built specifically to add reliable
   tool calling (not stock Google weights — worth knowing since it's a
   third-party fine-tune):

   ```bash
   brew install ollama
   brew services start ollama     # or: ollama serve, in its own terminal
   ollama pull orieg/gemma3-tools:4b-ft
   ollama pull orieg/gemma3-tools:12b-ft
   ```

   Two sizes on purpose: `4b-ft` reliably calls tools directly, so it's used
   by the three sub-agents. The root coordinator's job is deciding *which*
   specialist to call, which needs slightly more reliable reasoning, so it
   runs on `12b-ft`. `27b-ft` is also available if you want to push accuracy
   further.

2. **Create a virtualenv and install dependencies:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **(Optional) Customize the models.** `config.py`'s defaults already
   match the setup above (`4b-ft` for sub-agents, `12b-ft` for the
   coordinator, `http://localhost:11434` for Ollama) — no `.env` file is
   required to run. To use different sizes, create
   `financial_advisor/.env` with any of:

   ```bash
   ADK_MODEL=ollama_chat/orieg/gemma3-tools:4b-ft
   ADK_COORDINATOR_MODEL=ollama_chat/orieg/gemma3-tools:12b-ft
   OLLAMA_API_BASE=http://localhost:11434
   ADK_NUM_CTX=4096
   ADK_NUM_PREDICT=512
   ```

## Running locally

**Option A — ADK's own dev UI** (best for inspecting agent transfers/tool
calls while iterating):

```bash
adk web
```

Open the printed local URL and pick `financial_advisor` — the only entry
in the list, by design (see "Known limitations").

**Option B — Streamlit app** (the eventual deployment target):

```bash
streamlit run streamlit_app.py
```

## Notes

- `risk_profiler`, `portfolio_analyst`, and `market_research` append a
  disclaimer reminding the user this isn't licensed financial advice —
  keep that instruction if you extend the agents. `user_profile` and
  `financial_dashboard` only add it once, on the message where onboarding
  completes, so routine intake questions don't feel like a wall of legal
  text every turn.
- `get_stock_price` / `get_stock_fundamentals` / `get_historical_performance`
  / `get_market_overview` in `financial_advisor/tools/market_data.py` are
  plain Python functions; ADK derives their tool schema from the type
  hints and docstrings, so keep those accurate if you modify them.
  `get_market_overview` covers the S&P 500, Dow, Nasdaq, and VIX for
  recap/overview/trend questions that aren't about one specific ticker.
  `market_research` is explicitly told never to answer market-data
  questions from its own memory — see "Known limitations" for why that
  instruction exists.
- **Budgeting/investing guidance** (`dashboard_store.compute_guidance`) is
  derived from `monthly_income` and `monthly_expenses` using two standard,
  widely-cited rules of thumb — not personalized advice: a 50/30/20
  needs/wants/savings budget split, and prioritizing a 6-month emergency
  fund before recommending the full "savings" share go to investing (if
  the fund isn't there yet, that share splits evenly between topping it
  up and investing). It never recommends saving/investing more than
  `monthly_income - monthly_expenses` actually leaves over — if that's
  below the 20% target, `savings_shortfall` surfaces the gap instead.
  Shown in the dashboard-complete chat message and as extra sidebar
  tiles in `streamlit_app.py`.
- **Financial-guru/book citations** (`wisdom_store.py`) are hand-authored,
  paraphrased (not verbatim quoted), attributed summaries of well-known
  personal-finance frameworks — Dave Ramsey's Baby Steps, the 50/30/20
  rule, Bogle/Buffett on index investing, the Bogleheads three-fund
  portfolio, the 4% retirement rule, "pay yourself first," etc. 1-2
  relevant citations are guaranteed to appear in the dashboard-completion
  message (deterministically selected, not LLM-chosen); `portfolio_analyst`
  and `risk_profiler` can also call `get_financial_wisdom` on demand for
  relevant follow-up questions, best-effort (see "Known limitations").
- **Web search for questions outside the curated wisdom store**
  (`web_search_tools.search_financial_advice`) uses `ddgs` (DuckDuckGo,
  no API key — preserves this project's "no credentials anywhere"
  property) restricted to a `TRUSTED_FINANCIAL_DOMAINS` allowlist
  (Investopedia, NerdWallet, Bogleheads, IRS.gov, SEC.gov, etc.), falling
  back to unfiltered results (flagged `filtered: false`) only if nothing
  from the allowlist comes back. Returns short `{title, url, snippet}`
  entries for the agent to cite, never full page content. Wired into
  `portfolio_analyst`, `risk_profiler`, `market_research`, and
  `financial_dashboard`'s follow-up path — best-effort, see "Known
  limitations" for how reliably it actually gets used.
- For Streamlit Community Cloud deployment, note that it can't reach a
  `localhost` Ollama server — you'd need a hosted Ollama instance (or swap
  `financial_advisor/config.py` to point at a different LiteLLM-supported
  backend) before deploying there.
- **Speed on CPU-only hardware.** These fine-tunes default to a 32768-token
  context window and unbounded generation length, which is far more than a
  short instruction + a few tool schemas + a handful of turns ever needs —
  wasted memory/compute on CPU-only inference (no GPU). `config.py` trims
  both down (`ADK_NUM_CTX=4096`, `ADK_NUM_PREDICT=512` by default,
  overridable via env vars) for noticeably faster responses. If you're on
  a GPU-accelerated setup you can likely raise these back up.
- **The saved profile's `name` field doubles as the persistent-session
  identity key.** `streamlit_app.py` uses `DatabaseSessionService`
  (SQLite, `financial_advisor/data/sessions.db`) instead of an in-memory
  session service, and resumes the most recent conversation for whatever
  name is currently saved in the profile (falling back to a generic
  `"guest"` identity before onboarding gives a name). This is a
  single-local-user app, not multi-account switching — it's for surviving
  restarts, not for different people sharing one instance with separate
  histories.
- **If the app suddenly fails to import at all** (`NameError` or similar
  on startup), check `dashboard_store.py`'s `DATA_DIR` constant first —
  it was found renamed to `DATA1_DIR` at one point (line 12 defined it,
  line 13 referenced the old name), breaking every import of
  `financial_advisor.agent` transitively. Caught and fixed, but if a
  future manual edit reintroduces a similar typo, the whole app — `adk
  web`, Streamlit, everything — goes down at once, so it's worth knowing
  where to look first.
- **No API keys anywhere** — the whole stack (ADK, Ollama, yfinance) runs
  without credentials. The only local secrets-shaped file is
  `financial_advisor/.env`, which just holds model names and a localhost
  URL, no actual secrets.
- **Personal data stays local and gitignored.** `financial_advisor/data/`
  (profile.json, dashboard.json) and `financial_advisor/.env` are excluded
  in `.gitignore`, and are also caught by the parent repo's broader
  `*.json` / `*.db` / `.env` rules. `adk web`'s own session database
  (`financial_advisor/.adk/session.db`, contains full conversation history
  including anything shared during onboarding) is excluded the same way.

## Known limitations

This is a local demo, not production-grade agent routing. In testing
against the live local Ollama server (coordinator/router on `12b-ft`,
sub-agents on `4b-ft`):

- **Profile-completeness gating (user_profile vs. specialist_router) is
  deterministic Python, not an LLM decision** — it was tried as an LLM
  instruction first ("call get_profile, then branch") and that failed
  every time; the coordinator just skipped straight to whatever specialist
  the question obviously matched. The current `FinancialCoordinator`
  `BaseAgent` checks profile completeness itself before invoking any LLM,
  so this part is reliable by construction, not just by testing.
- **Specialist routing (risk_profiler / portfolio_analyst / market_research)
  is still an LLM decision** (`specialist_router`'s `AgentTool` calls), and
  is not 100% reliable: across repeated trials it correctly calls the right
  specialist most of the time, but occasionally answers a
  portfolio-concentration-style question directly from its own knowledge
  instead of calling `portfolio_analyst` — a reasonable-sounding but
  ungrounded answer, not a crash. Direct tool calls once a specialist *is*
  invoked (e.g. `get_stock_price`) have been reliable across every trial.
- If you see the router answering directly when you expected it to consult
  a specialist, that's this known failure mode, not a bug. Trying `27b-ft`
  for the coordinator/router (edit `ADK_COORDINATOR_MODEL` in
  `financial_advisor/.env`) is the first thing to try if this happens often
  enough to matter for your use case.
- **This "answer directly instead of calling a specialist" failure mode is
  worse than it sounds for market data specifically: it was observed to
  fabricate plausible-but-wrong index numbers** (a market recap question
  got confident, specific S&P/Dow/Nasdaq figures that didn't match reality
  — stale/hallucinated, not fetched) instead of declining or calling
  `market_research`. The fix applied was tightening `specialist_router`'s
  own instruction to explicitly cover broad/index-level questions (not
  just specific tickers) and to explicitly forbid answering market
  questions from memory — retested and confirmed fixed for that case, but
  treat any market figure from `specialist_router` itself (as opposed to
  `market_research`) with suspicion, and prefer rephrasing the question if
  you're unsure whether a lookup actually happened.
- **`adk web`'s dropdown recursively lists every directory containing a
  file literally named `agent.py`** (a plain filename match, not a check
  for a real agent inside it — confirmed by reading the installed
  `google-adk` source). With sub-agents living in their own packages, this
  used to clutter the dropdown with entries for each specialist that
  didn't even work if selected (`ValueError: No root_agent found`, since
  they don't export a symbol literally named `root_agent`). Fixed by
  renaming every sub-agent module to `<name>_agent.py` — keep that
  convention for any new sub-agent, and never let a sub-package's
  `__init__.py` contain the literal substring `"root_agent"`.
- **The model can produce fluent, correct-looking text that describes an
  action it never actually took.** Asking `user_profile` to "extract the
  user's name from their reply and call `update_profile_field`" produced
  replies that used the name naturally ("Nice to meet you, Sarah!") but
  never actually called the tool — the profile stayed empty. This is a
  different failure mode from the earlier "skips the tool entirely and
  declines to help" ones: here the text output looks completely correct,
  which makes it easy to miss without checking the underlying saved data,
  not just eyeballing the chat transcript. The fix follows the same
  pattern as the profile/dashboard completeness gates: don't ask the LLM
  to make a call it's proven unreliable at — `user_profile_agent.py`'s
  `ask_name_first` is a `before_agent_callback` that deterministically
  asks for the name (skipping the LLM for that turn) and then
  deterministically saves the next raw user reply as the name (skipping
  LLM extraction, since a direct reply to "what's your first name?"
  needs no NLU). If you add fields where correctness of *storage*, not
  just conversational tone, matters, consider whether it needs the same
  treatment before trusting the LLM's tool-calling for it.
- **The same gap showed up for every other required profile field too —
  user-reported, reproduced, and now fixed for all of them.** Reported
  live: the user said "I've never invested before" and was later asked
  about investing experience again. Reproduced with a batch reply
  combining all remaining profile fields in one message ("30, single,
  $95k a year stable job, California, moderate risk, ive never invested
  before, 10 year horizon") — the agent's own reply text correctly
  referenced the info, but `load_profile()` afterward showed only `name`
  saved; everything else was silently dropped. A first, narrower fix
  covered just `investment_experience`/`risk_tolerance` (small, keyword-
  matchable vocabularies), but the same retest showed the other five
  fields still silently failing, plus confusing re-asks of fields that
  actually *had* saved. Generalized into `user_profile_agent.py`'s
  `capture_profile_fields` (`before_agent_callback`, alongside
  `ask_name_first`), which runs before every LLM turn once `name` is
  known:
  1. **Positional pass**: split the reply on commas/semicolons/newlines.
     If the segment count exactly matches the number of currently-missing
     fields (the common case, since the agent always asks in a fixed
     order and users tend to answer in the same order), map segments to
     fields positionally and extract each with a per-field parser (age:
     leading number or "I'm X"/"X years old"; income: dollar amount +
     stability word; location: matched against a US state/country list;
     time_horizon: "X years" or long/short-term/retirement; household
     status/risk tolerance/investment experience: keyword vocabularies as
     before). A segment that doesn't parse cleanly is still saved as-is
     — positional matching already gives high confidence which field it
     answers, so a raw save beats losing it.
  2. **Fallback pass**: if the reply doesn't cleanly segment (flowing
     prose, or only some fields answered), scan the whole reply for each
     still-missing field's *context-anchored* pattern only — no bare
     numbers or low-confidence guesses here, since a wrong silent save is
     worse than an occasional re-ask.
  Retested three ways: the original 7-field batch reply now saves all 7
  correctly (`{'name': 'Jordan', 'age': '30', 'household_status':
  'Single', 'income': '$95k, stable', 'location': 'California',
  'risk_tolerance': 'Moderate', 'investment_experience': 'No experience —
  never invested before', 'time_horizon': '10 years'}`), with the agent's
  own follow-up correctly summarizing everything back for confirmation;
  and a messy, partial, non-segmented reply ("I'm 25 and I live in Texas.
  I've been investing for a few years so I'd say I have some experience
  with it.") correctly captured age/location/investment_experience via
  the fallback pass while correctly leaving the genuinely-unanswered
  fields (household_status, income, risk_tolerance, time_horizon) open
  — and the agent's next question asked only about those, no redundant
  re-asks. Still best-effort, not full NLU — unusual phrasings can still
  slip through the fallback pass, but the common cases (clean batch
  answers and reasonably clear prose) are now covered.
- **Fixed: the fallback pass's own bare-word keyword matching had false
  positives.** User-reported: household status ("living situation") had
  stopped being asked at all. Root cause, found by code review and
  confirmed with test phrasings: the fallback pass (see above) scans
  *every* subsequent message for *every* still-missing field's keywords,
  and several patterns were bare, common English words — "single,"
  "conservative," "aggressive," "experienced" — that show up constantly
  outside their onboarding meaning. Confirmed all four of these
  incidentally matched and silently saved a wrong value with zero
  relation to the actual question: "Do I have a **single** source of
  income?" → `household_status: 'Single'`; "Give me a **conservative**
  estimate of my expenses" → `risk_tolerance: 'Conservative'`; "I want an
  **aggressive** savings plan" → `risk_tolerance: 'Aggressive'`; "I
  **experienced** a big loss last year" → `investment_experience:
  'Experienced'`. Once wrongly saved, the field silently vanishes from
  "missing" and is never asked about again — a data-*accuracy* bug that
  manifests identically to the earlier data-*loss* bugs (field never
  gets a real answer), just via the opposite mechanism. Fixed by giving
  `household_status`, `risk_tolerance`, and `investment_experience` two
  pattern tiers: STRICT (requires enough surrounding context — "I'm
  single," "conservative investor," "risk-averse" — to be confident the
  message is actually answering that question) for the fallback
  whole-message scan, and LOOSE (STRICT plus bare words) only for
  positional mode, where the segment is already confirmed to correspond
  to that field by position, so a bare word is safe there. Retested: all
  four false-positive phrasings now correctly extract nothing, the
  original legitimate phrasings ("I'm single and live alone," "I'd say
  I'm pretty conservative...") still extract correctly, and the full
  7-field batch-reply regression test still saves everything correctly —
  no loss of the fix from earlier, just removal of the false-positive
  risk it introduced.
- **`financial_dashboard` needed the same treatment, more broadly.**
  Testing with a *partially* pre-filled dashboard (unlike earlier tests
  that always started from empty, which happened to mask this) exposed
  three separate reliability failures in one flow: on a cold turn it
  sometimes never called `get_dashboard_status` at all and just
  re-asked/invented questions ignoring already-saved fields; on a batch
  reply (e.g. "my emergency fund has 5000 dollars") it produced a
  confident, specific-sounding confirmation ("I've set that to $5000")
  with zero tool calls and no actual save; and once every field *was*
  saved, it kept asking stale already-answered questions instead of
  noticing completion and presenting the summary. `financial_dashboard_agent.py`'s
  `ask_missing_dashboard_fields` (`before_agent_callback`) now handles
  all three deterministically: compute the true missing-field list and
  ask for exactly those; if a reply contains exactly as many numbers as
  fields asked, save them positionally without depending on a tool call;
  and build the completion summary + guidance message directly in Python
  (`_build_completion_message`) rather than asking the LLM to notice
  completion and read the numbers out correctly. The LLM is only invoked
  for genuinely messy/partial replies and post-completion follow-ups —
  retested against both a partial-dashboard and a fresh-empty-dashboard
  scenario, matching hand-calculated expected values exactly in both.
- **Fixed: the model was skipping `search_financial_advice` instead of
  calling it.** Live testing initially found the model reliably chose
  **not** to call it, even on questions squarely outside
  `wisdom_store.py`'s topics — e.g. asked `portfolio_analyst` directly
  "how often should I rebalance my portfolio?" it answered fluently and
  plausibly entirely from memory, zero tool calls. Unlike the
  profile/dashboard fields, "does this question need a web lookup" is
  inherent LLM judgment, not a yes/no state check, so it couldn't get the
  same before-the-fact deterministic treatment. Instead,
  `web_search_tools.ground_if_ungrounded` (an `after_model_callback` on
  `portfolio_analyst`, `risk_profiler`, `market_research`, and
  `financial_dashboard`) catches it **after** the fact: if a response has
  no tool call, no tool was already called earlier that turn, and its
  non-question content is substantial (`_declarative_length` — sentence-
  aware, so a long ungrounded answer can't dodge detection just by
  tacking on a trailing "want me to look that up?"), it discards that
  response and injects a synthetic `search_financial_advice` function
  call instead. ADK executes a callback-returned response exactly like a
  real model response, so this gets run for real and looped back to the
  model with actual results — reusing the "synthesize from a given tool
  result" behavior this project has seen work reliably throughout,
  instead of trusting the model to decide to fetch one. Retested against
  the exact rebalancing question that exposed this: it now calls
  `search_financial_advice`, gets a real Vanguard-sourced result, and
  answers "according to Vanguard, an annual rebalance would typically
  yield the best results" — a real citation, not a recital from memory.
  Confirmed no regression on legitimately-grounded single-tool-call
  answers (e.g. `market_research` presenting a real fetched stock price
  isn't second-guessed or re-triggered).
- **Fixed: process narration and citation fabrication.** Two follow-on
  issues surfaced once grounding started working:
  1. The model would narrate its own process instead of just acting —
     "would you like me to search/look that up?", "let me check", "I
     found through my research that...". All agent instructions now
     explicitly forbid this: call the tool silently, answer with what it
     returns.
  2. More seriously: even when genuinely grounded, the model sometimes
     **fabricated additional sources beyond what was actually
     retrieved**. Asked the rebalancing question, `search_financial_advice`
     returned 5 real results (SoFi, Farm Bureau, Zynergy Retirement,
     optimizedportfolio.com, and one Vanguard-content mirror) — but the
     model's answer cited "Vanguard," "Retirement Researcher,"
     "Investor.gov," "Farther.com," "The Conversation," and a YouTube
     channel, none of which correspond to the 5 actual URLs returned. It
     blended one real mention (a snippet did discuss Vanguard's research)
     with several invented ones, presented with equal confidence — a
     more dangerous failure than not searching at all, since the citation
     *looks* verified. The fix: agents no longer name the specific
     website/source by default at all (only the *information* is used to
     ground the answer) — only citing a source if the user explicitly
     asks, and even then only one "literally present in the tool result."
     Named principles/frameworks from `wisdom_store.py` (Dave Ramsey,
     Bogle, the 4% rule, etc.) are unaffected and still always cited by
     name, since that content is hand-authored and carries no fabrication
     risk. Retested: the same rebalancing question now produces one
     citation ("Based on Vanguard's research...", traceable to real
     snippet content) instead of six mixed real/invented ones — a real
     improvement, though not a perfect one: the model still named that
     one source despite not being asked, showing the same imperfect
     instruction-following seen elsewhere in this project.
  3. **Fixed further with a deterministic safety net.** Since instruction
     compliance alone wasn't 100%, `web_search_tools.strip_unrequested_citations`
     (an additional `after_model_callback`) catches what the instruction
     missed: if `search_financial_advice` ran this turn (tracked via a new
     `track_tool_usage` `after_tool_callback`, distinct from
     `get_financial_wisdom` so wanted wisdom citations are never touched)
     and the user's own message doesn't look like it's asking for a
     source (regex on "source", "cite", "link", "where did you find",
     etc.), it pattern-matches and strips citation lead-ins ("According to
     X, ...", "Based on X's research, ...", "X suggests that ...") from
     the response text. This is a best-effort text filter for the
     specific phrasings actually observed live, not a general NLP
     solution — it won't catch every possible citation phrasing, but it
     meaningfully reduces the residual gap instruction-following alone
     left open. Retested: the same rebalancing question now sometimes
     produces no named source at all ("the consensus among financial
     advisors suggests...") and the tool call still grounds the
     substance of the answer either way.
- **This does not fix `specialist_router` skipping delegation to a
  specialist in the first place** (the router-level version of the same
  failure mode) — `ground_if_ungrounded` lives on the specialists
  themselves, so it only helps once one is actually reached. That gap is
  harder to close the same way, since forcing a *specific* specialist
  delegation requires guessing which of three is right, not injecting one
  well-defined tool call. Still open; `27b-ft` for the router is the
  first thing to try if it matters for your use case.
- **Spending category breakdown, and a real bug in wiring it up.**
  `financial_dashboard` now offers a second, optional stage after the
  required-fields summary: a breakdown of monthly expenses by category
  (housing, transportation, food, utilities, insurance, debt payments,
  entertainment, other), computed against two more widely-cited
  underwriting guidelines — housing at or under ~30% of income, and total
  debt-to-income (housing + debt payments) at or under ~36% — with
  `wisdom_store.py` citations for both, the same "hand-authored, not
  LLM-recalled" pattern as the rest of the guidance. It's skippable (says
  so explicitly, recognizes "skip"/"no thanks"/etc.) since it's
  enrichment, not required.

  Building this exposed a real architecture bug, caught before shipping:
  the first version tracked "has this been offered" in ADK session
  state, and marked it at the moment the offer was *shown*. Two problems,
  both found via live testing rather than assumed: (1) the top-level
  `FinancialCoordinator` gate only ever checked the 5 *required* dashboard
  fields before routing straight to `specialist_router` — for a session
  where those were already complete (e.g. a returning user, or simply a
  fresh test with pre-seeded data), `financial_dashboard` was never
  invoked at all, so the new stage never got a chance to run; and (2)
  marking "offered" at ask-time (rather than at actual resolution) meant
  that *if* the gate did route there, the user's own reply — the answer
  to the question just asked — would get routed to `specialist_router`
  on the very next turn instead of back to `financial_dashboard` to
  actually be captured. Fixed by moving the tracking to persistent
  storage (`dashboard_store.spending_breakdown_pending`, read from
  `dashboard.json`, not session state, so it survives a fresh session
  the same way required-field completeness already does) and only
  marking it resolved at genuine resolution points (completed, declined,
  or given up after one unparseable reply) rather than at the initial
  ask. Retested three ways: full 8-category batch reply → all captured
  correctly with accurate math (verified by hand: housing 36.7% of
  income → correctly flagged over the 30% guideline; DTI 46.7% →
  correctly flagged over 36%; categorized total vs. reported
  monthly_expenses discrepancy correctly signed and reported); decline
  ("no thanks") → correctly marked resolved with zero spending fields
  saved, and the *next* turn correctly reached `specialist_router`
  instead of getting stuck. One remaining cosmetic-only quirk: the LLM's
  own free-text reply immediately after a decline was once observed
  acknowledging the decline and then still asking about housing spend in
  the same breath — confusing phrasing, but no data was incorrectly
  saved, so it's a tone issue, not a functional one.
