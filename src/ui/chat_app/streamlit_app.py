"""Streamlit chat UI for HDB resale price prediction.

Users describe a flat in natural language. Each message is run through the
LangGraph orchestration graph (:mod:`ui.chat_app.graph`), which parses the
fields, resolves any postal code, predicts and explains via the in-process MCP
tools, and narrates the reply — all behind one ``ainvoke`` call. The model never
runs in this process.

Each user message starts a fresh graph invocation: the graph is stateless per
turn. Streamlit's session state holds the visible conversation history, but the
parse node only ever sees the latest message. Multi-turn awareness is a
deliberate non-goal for this phase.
"""

import asyncio
import logging

import streamlit as st
from pydantic import ValidationError

from ui.chat_app.config import ChatConfig
from ui.chat_app.graph import GraphState, build_graph

logger = logging.getLogger(__name__)

# Load configuration through ChatConfig so the key resolves from either the
# environment or a .env file at the project root — the whole point of Pydantic
# Settings. Reading os.environ directly here would defeat that and report the
# key as missing whenever it lives only in .env.
_KEY_HELP = (
    "Set `ANTHROPIC_API_KEY` in your environment, or add it to a `.env` "
    "file in the project root:\n\n"
    "```\nANTHROPIC_API_KEY=sk-ant-...\n```"
)

try:
    _config = ChatConfig()
except ValidationError as exc:
    st.error(f"**Could not load chat configuration.**\n\n```\n{exc}\n```\n\n{_KEY_HELP}")
    st.stop()

if not _config.anthropic_api_key:
    st.error(f"**ANTHROPIC_API_KEY is not set.** {_KEY_HELP}")
    st.stop()

st.set_page_config(
    page_title="HDB Resale Price Chat",
    layout="centered",
)

st.title("HDB Resale Price Chat")
st.caption(
    "Describe a flat in plain English — e.g. *'3-room in Tampines, 90 sqm, lease started 1992'*"
)

# Compile the graph once per session. Streamlit reruns the script top to bottom on
# every interaction, so caching the compiled graph in session state avoids
# rebuilding it on each message.
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

# messages: (role, text) tuples for rendering the conversation.
if "messages" not in st.session_state:
    st.session_state.messages = []

for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(text)


def _run_turn(message: str) -> str:
    """Invoke the graph for one user message and return the narrated reply.

    Streamlit runs synchronously, so the async graph is driven via
    ``asyncio.run``. Per Day 1's note, ``ainvoke`` returns a plain dict, which is
    reconstructed into a :class:`GraphState` to read ``response_text`` with type
    safety.
    """
    result = asyncio.run(st.session_state.graph.ainvoke(GraphState(user_message=message)))
    return GraphState(**result).response_text


if prompt := st.chat_input("Ask about an HDB flat..."):
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = _run_turn(prompt)
            except Exception as exc:
                logger.exception("graph invocation failed")
                reply = f"Sorry, something went wrong: `{exc}`"
        st.markdown(reply)

    st.session_state.messages.append(("assistant", reply))

with st.sidebar:
    st.subheader("About")
    st.write(
        "GradientBoostingRegressor trained on HDB resale transactions.\n\n"
        "- MAE: ~S$20,000\n"
        "- R²: 0.975\n\n"
        "Predictions are estimates; actual prices may vary."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
