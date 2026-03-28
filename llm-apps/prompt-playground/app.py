import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.title("🎯 AI Prompt Playground")

prompt = st.text_area("Enter your prompt")

style = st.selectbox("Choose response style", [
    "Normal",
    "Explain like I'm 5",
    "Professional",
    "Funny"
])

def format_prompt(prompt, style):

    if style == "Explain like I'm 5":
        return f"Explain this in very simple terms: {prompt}"

    elif style == "Professional":
        return f"Give a professional explanation: {prompt}"

    elif style == "Funny":
        return f"Explain this in a funny way: {prompt}"

    return prompt


if st.button("Generate"):

    final_prompt = format_prompt(prompt, style)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=final_prompt
    )

    st.subheader("Response")
    st.write(response.text)
