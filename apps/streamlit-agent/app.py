import streamlit as st
import requests

st.title("🤖 AI Agent (API Powered)")

API_URL = "http://127.0.0.1:8000/chat"

# Chat history (UI only)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask something...")

if user_input:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Call backend API
    response = requests.post(API_URL, json={"question": user_input})

    if response.status_code == 200:
        answer = response.json()["response"]
    else:
        answer = "Error: Backend not responding"

    # Show AI response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)