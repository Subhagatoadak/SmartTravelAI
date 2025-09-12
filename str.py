import time
import requests
import streamlit as st
from streamlit_chat import message

st.set_page_config(page_title="Streamlit Chat", page_icon="💬", layout="centered")

# ---- Sidebar ----
st.sidebar.title("Settings")
use_backend = st.sidebar.toggle("Use backend endpoint", value=False)
backend_url = st.sidebar.text_input(
    "Backend URL (POST JSON: {'message': str})",
    value="http://localhost:8000/chat",
    help="Your FastAPI (or other) endpoint that returns {'reply': '...'}",
)
temperature = st.sidebar.slider("Dummy response creativity", 0.0, 1.0, 0.2, 0.1)

# ---- Init session state ----
if "messages" not in st.session_state:
    # Each item: {"role": "user" | "assistant", "content": str}
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything to get started."}
    ]

# ---- Helper: call backend or local stub ----
def get_assistant_reply(user_text: str) -> str:
    """
    If use_backend is True: POST to backend_url with JSON {'message': user_text}
    expecting {'reply': '...'} response.
    Otherwise, return a local stubbed response.
    """
    if use_backend:
        try:
            resp = requests.post(
                backend_url,
                json={"message": user_text},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return str(data.get("reply", "⚠️ Backend did not return 'reply'."))
        except Exception as e:
            return f"⚠️ Backend error: {e}"
    # Local stub: simple reversible-echo with tiny delay to feel interactive
    time.sleep(0.3)
    return f"(local) You said: {user_text[::-1]}  \n(temp={temperature})"

# ---- Title ----
st.title("💬 Streamlit + streamlit-chat")

# ---- Chat container ----
chat_container = st.container()

with chat_container:
    for i, msg in enumerate(st.session_state.messages):
        is_user = msg["role"] == "user"
        message(
            msg["content"],
            is_user=is_user,
            key=f"msg-{i}",
            avatar_style="thumbs",  # fun default; remove if you want default avatars
        )

# ---- Input form (so enter submits nicely) ----
with st.form(key="chat-input", clear_on_submit=True):
    user_text = st.text_input("Your message", placeholder="Type here and press Enter…")
    submitted = st.form_submit_button("Send")

if submitted and user_text.strip():
    # 1) Add user message
    st.session_state.messages.append({"role": "user", "content": user_text.strip()})

    # 2) Get assistant reply
    reply = get_assistant_reply(user_text.strip())
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # 3) Rerender messages immediately
    st.experimental_rerun()

# ---- Utilities (optional) ----
col1, col2 = st.columns(2)
with col1:
    if st.button("🧹 Clear chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared. How can I help now?"}
        ]
        st.experimental_rerun()
with col2:
    if st.button("💾 Export history"):
        # Compact JSONL export for downstream analysis
        import json, io
        buf = io.StringIO()
        for m in st.session_state.messages:
            buf.write(json.dumps(m, ensure_ascii=False) + "\n")
        st.download_button(
            "Download messages.jsonl",
            data=buf.getvalue().encode("utf-8"),
            file_name="messages.jsonl",
            mime="application/json",
        )
