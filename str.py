import json
from typing import Any, Dict, List, Union

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Chat", page_icon="💬", layout="centered")

# ---------- Styles ----------
st.markdown(
    """
    <style>
    /* chat area frame */
    .chat-window{
        height: 460px;               /* adjust as you like */
        overflow-y: auto;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px 14px 6px 14px;
        background: #fff;
    }
    /* simple bubbles */
    .msg{ 
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 10px 12px;
        margin-bottom: 10px;
        box-shadow: 0 1px 1px rgba(0,0,0,.02);
    }
    .msg-user{ background:#eef2ff; border-color:#c7d2fe; }
    .msg-bot{  background:#f8fafc; border-color:#e2e8f0; }
    .sender{ font-weight:600; font-size:0.85rem; color:#475569; margin-bottom:4px; }
    .content{ white-space:pre-wrap; font-size:0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
def get_bot_response(user_input: str) -> Union[str, List[Any], Dict[str, Any]]:
    """Replace this with your real backend call."""
    # demo: echo + structured examples to show pretty rendering
    if user_input.strip().lower() == "list":
        return ["apples", "bananas", "pears"]
    if user_input.strip().lower() == "table":
        return [
            {"sku": "A101", "qty": 12, "price": 6.5},
            {"sku": "B202", "qty": 3, "price": 12.0},
        ]
    if user_input.strip().lower() == "json":
        return {"kpi": "CTR", "value": 0.0423, "conf_int": [0.039, 0.045]}
    return f"You said: {user_input}"

def render_payload_nicely(payload: Any):
    """
    - list[dict] -> dataframe
    - dict/list -> pretty code block (JSON)
    - str/other -> markdown
    """
    if isinstance(payload, list) and payload and all(isinstance(x, dict) for x in payload):
        df = pd.DataFrame(payload)
        st.dataframe(df, use_container_width=True, hide_index=True)
    elif isinstance(payload, (list, dict)):
        st.code(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        st.markdown(str(payload))

def bubble(sender: str, content: Any):
    cls = "msg-user" if sender == "You" else "msg-bot"
    with st.container():
        st.markdown(f"<div class='msg {cls}'>"
                    f"<div class='sender'>{sender}</div>"
                    f"<div class='content'>", unsafe_allow_html=True)
        render_payload_nicely(content)
        st.markdown("</div></div>", unsafe_allow_html=True)

# ---------- State ----------
if "messages" not in st.session_state:
    # store as list of dicts: [{"role":"You"/"Bot","content":...}]
    st.session_state.messages: List[Dict[str, Any]] = []

st.title("💬 Chat")

# ---------- Chat window (scrollable) ----------
chat_container = st.container()
with chat_container:
    st.markdown("<div class='chat-window'>", unsafe_allow_html=True)

    # latest at TOP => iterate reversed
    for m in reversed(st.session_state.messages):
        bubble(m["role"], m["content"])

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Input (fixed below) ----------
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You", key="input", placeholder="Type a message…")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    st.session_state.messages.append({"role": "You", "content": user_input})
    bot = get_bot_response(user_input)
    st.session_state.messages.append({"role": "Bot", "content": bot})
    # Rerun to update the scroll area immediately with the new top message
    st.rerun()
