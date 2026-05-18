import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.prompts import PromptTemplate
from agent.tools import TOOLS
from agent.tracing import get_langfuse_handler

load_dotenv()

# ─── System Prompt ────────────────────────────────────────────────────────────
# This is the ReAct prompt template. ReAct = Reason + Act.
# The agent loops through: Thought → Action → Observation → Thought → ...
# until it reaches a Final Answer.
#
# The {tools} and {tool_names} placeholders are filled automatically by LangChain.
# You MUST keep the exact format (Thought/Action/Action Input/Observation) —
# the agent parser depends on these exact strings.

REACT_PROMPT = PromptTemplate.from_template("""
You are a financial markets assistant with access to real-time forex prices 
and recent financial news. You help users understand market conditions, 
recent economic events, and currency movements.

Be concise, factual, and cite your sources when using news articles.
If you don't have enough information, say so rather than guessing.

You have access to the following tools:
{tools}

Use this format strictly:

Question: the input question you must answer
Thought: reason about what you need to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have enough information to answer
Final Answer: your final answer to the original question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")


# ─── LLM ─────────────────────────────────────────────────────────────────────

def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",       # cheap and smart enough for ReAct
        temperature=0,             # 0 = deterministic, important for agents
                                   # higher temp causes the agent to hallucinate tool names
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ─── Agent Factory ────────────────────────────────────────────────────────────

def create_agent_executor() -> AgentExecutor:
    llm = get_llm()

    # create_react_agent wires together: LLM + tools + prompt
    # It handles the ReAct loop logic for you
    agent = create_react_agent(
        llm=llm,
        tools=TOOLS,
        prompt=REACT_PROMPT,
    )

    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,          # prints the full Thought/Action/Observation chain to terminal
        max_iterations=6,      # safety limit — prevents infinite loops if agent gets confused
        handle_parsing_errors=True,  # if LLM returns malformed output, retry instead of crash
    )


# ─── Main Query Function ──────────────────────────────────────────────────────

def query_agent(question: str) -> str:
    """
    Run the agent on a question. Traces everything to LangFuse automatically.
    """
    executor = create_agent_executor()
    langfuse_handler = get_langfuse_handler()

    result = executor.invoke(
        {"input": question},
        config={"callbacks": [langfuse_handler]},  # this single line enables full tracing
    )

    return result.get("output", "No answer generated.")


# ─── Manual Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        "What is the current EUR/USD rate and what recent news might be affecting it?",
        "Summarize the latest news about oil prices.",
        "What's happening with the Federal Reserve and how might it affect the dollar?",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print('='*60)
        answer = query_agent(question)
        print(f"\nFINAL ANSWER:\n{answer}")
        print()