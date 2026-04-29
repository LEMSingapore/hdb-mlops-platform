"""Streamlit chat UI for HDB resale price prediction.

Users describe a flat in natural language; Claude Haiku extracts the required
fields via tool use and calls the FastAPI prediction service through
:mod:`ui.chat_app.chat_agent`. The model never runs in this process.
"""

import logging

import streamlit as st

from ui.chat_app.chat_agent import chat_turn
from ui.chat_app.config import ChatConfig

logger = logging.getLogger(__name__)

_config = ChatConfig()
if not _config.anthropic_api_key:
    st.error(
        "**ANTHROPIC_API_KEY is not set.** "
        "Export it in the same shell before launching:\n\n"
        "```\nexport ANTHROPIC_API_KEY=sk-ant-...\n"
        "streamlit run src/ui/chat_app/streamlit_app.py\n```"
    )
    st.stop()

st.set_page_config(
    page_title="HDB Resale Price Chat",
    layout="centered",
)

st.title("HDB Resale Price Chat")
st.caption(
    "Describe a flat in plain English — e.g. *'3-room in Tampines, 90 sqm, lease started 1992'*"
)

# history: raw Anthropic-format messages (sent to the API each turn)
# display: (role, text) tuples for rendering the conversation
if "history" not in st.session_state:
    st.session_state.history = []
if "display" not in st.session_state:
    st.session_state.display = []

for role, text in st.session_state.display:
    with st.chat_message(role):
        st.markdown(text)

if prompt := st.chat_input("Ask about an HDB flat..."):
    st.session_state.display.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                st.session_state.history, reply = chat_turn(st.session_state.history)
            except Exception as exc:
                logger.exception("chat_turn failed")
                reply = f"Sorry, something went wrong: `{exc}`"
        st.markdown(reply)

    st.session_state.display.append(("assistant", reply))

with st.sidebar:
    st.subheader("About")
    st.write(
        "GradientBoostingRegressor trained on HDB resale transactions.\n\n"
        "- MAE: ~S$20,000\n"
        "- R²: 0.975\n\n"
        "Predictions are estimates; actual prices may vary."
    )
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.session_state.display = []
        st.rerun()
