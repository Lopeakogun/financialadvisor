import os

from google.adk.models.lite_llm import LiteLlm

# Stock gemma3 tags in Ollama do not support tool calling at any size, which
# breaks both the market-data tools and agent delegation. Default to
# orieg/gemma3-tools, a community fine-tune of Gemma 3 that adds it.
SUB_AGENT_MODEL = os.getenv("ADK_MODEL", "ollama_chat/orieg/gemma3-tools:4b-ft")

# The root coordinator's job is deciding which sub-agent to transfer to, not
# just calling a tool directly. In testing, the 4b-ft fine-tune reliably
# calls tools when it's the one doing the work, but frequently *describes*
# delegating without emitting the actual transfer_to_agent call. 12b-ft
# routes correctly, so the coordinator gets the bigger model while
# sub-agents (whose direct tool calls already work fine at 4b) stay small.
COORDINATOR_MODEL = os.getenv(
    "ADK_COORDINATOR_MODEL", "ollama_chat/orieg/gemma3-tools:12b-ft"
)

# These fine-tunes bake in a 32768-token context window by default, far more
# than a short instruction + a handful of tool schemas + a few conversation
# turns ever needs. On CPU-only inference (no GPU acceleration) that's a lot
# of wasted memory/compute per call. Trim both way down for speed; bump via
# env var if a conversation ever gets long enough to truncate.
NUM_CTX = int(os.getenv("ADK_NUM_CTX", "4096"))
# Cap generation length too (Modelfile default is unlimited) so a rare
# runaway response can't blow up latency further.
NUM_PREDICT = int(os.getenv("ADK_NUM_PREDICT", "512"))


def get_model() -> LiteLlm:
    """Build the LiteLLM-wrapped local Ollama model used by sub-agents."""
    return LiteLlm(model=SUB_AGENT_MODEL, num_ctx=NUM_CTX, num_predict=NUM_PREDICT)


def get_coordinator_model() -> LiteLlm:
    """Build the LiteLLM-wrapped local Ollama model used by the root coordinator."""
    return LiteLlm(model=COORDINATOR_MODEL, num_ctx=NUM_CTX, num_predict=NUM_PREDICT)
