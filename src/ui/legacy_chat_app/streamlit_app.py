# mypy: ignore-errors
# ruff: noqa
"""Streamlit chat UI for the HDB resale price predictor.

Users describe a flat in natural language; Claude (Haiku) extracts the five
required fields via tool use and calls the local XGBoost model through
predictor.predict_price().
"""

import streamlit as st

from chat_agent import chat_turn

st.set_page_config(
    page_title="HDB Resale Price Chat",
    page_icon="🏠",
    layout="centered",
)

st.title("HDB Resale Price Chat")
st.caption(
    "Describe a flat in plain English — e.g. "
    "*'4-room in Tampines, 90 sqm, postal 520329, lease started 1992'*"
)

# history: raw Anthropic-format messages (what we send to the API)
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
            except Exception as e:
                reply = f"Sorry, something went wrong: `{e}`"
        st.markdown(reply)

    st.session_state.display.append(("assistant", reply))

with st.sidebar:
    st.subheader("About")
    st.write(
        "XGBoost regression model trained on HDB resale transactions.\n\n"
        "- MAE: ~$15,749\n"
        "- R²: 0.982\n\n"
        "Predictions are estimates; actual prices may vary."
    )
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.session_state.display = []
        st.rerun()
