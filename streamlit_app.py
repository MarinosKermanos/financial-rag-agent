import streamlit as st
import requests
import os

# Config

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Financial RAG Agent",
    page_icon="📈",
    layout="centered",
)

# Header

st.title("📈 Financial RAG Agent")
st.caption(
    "Powered by GPT-4o-mini · Qdrant vector search · LangFuse observability"
)
st.divider()

# Example Questions

st.markdown("**Try asking:**")
col1, col2 = st.columns(2)

example_questions = [
    "What is the current EUR/USD rate and what's affecting it?",
    "Summarize the latest Federal Reserve news",
    "What's happening with oil prices?",
    "How is the US dollar performing lately?",
]

# Clicking an example pre-fills the input
if col1.button(f"💬 {example_questions[0]}", use_container_width=True):
    st.session_state.prefill = example_questions[0]
if col1.button(f"💬 {example_questions[1]}", use_container_width=True):
    st.session_state.prefill = example_questions[1]
if col2.button(f"💬 {example_questions[2]}", use_container_width=True):
    st.session_state.prefill = example_questions[2]
if col2.button(f"💬 {example_questions[3]}", use_container_width=True):
    st.session_state.prefill = example_questions[3]

st.divider()

# Chat History

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input

prefill = st.session_state.pop("prefill", "")
question = st.chat_input("Ask about forex, markets, or financial news...")

# Use prefill if a button was clicked, otherwise use typed input
active_question = prefill or question

if active_question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": active_question})
    with st.chat_message("user"):
        st.markdown(active_question)

    # Call the API
    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"question": active_question},
                    timeout=60,  # agent can take time — give it room
                )
                response.raise_for_status()
                answer = response.json()["answer"]
            except requests.exceptions.Timeout:
                answer = "⚠️ The request timed out. The agent may be overloaded — try again."
            except Exception as e:
                answer = f"⚠️ Error: {str(e)}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})