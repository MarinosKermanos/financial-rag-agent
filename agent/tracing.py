import os
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

load_dotenv()

# Langfuse client, for manual logging if needed
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)


def get_langfuse_handler() -> CallbackHandler:
    """
    Returns a LangChain callback handler that automatically traces:
    - Every LLM call (prompt + response + token usage + latency)
    - Every tool call (which tool, what input, what output)
    - The full agent reasoning chain (ReAct thought → action → observation loop)
    
    You just pass this handler into any LangChain chain/agent and 
    everything gets logged to LangFuse automatically.
    """
    return CallbackHandler()