import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.title("🧠 AI Text Summarizer")

text = st.text_area("Enter text to summarize")

mode = st.selectbox("Select Summary Type", [
    "Short Summary",
    "Bullet Points",
    "Key Insights"
])

def build_prompt(text, mode):

    if mode == "Short Summary":
        return f"Summarize this text in a concise paragraph:\n{text}"

    elif mode == "Bullet Points":
        return f"Summarize this text into clear bullet points:\n{text}"

    elif mode == "Key Insights":
        return f"Extract key insights and important takeaways:\n{text}"

    return text


if st.button("Generate Summary"):

    prompt = build_prompt(text, mode)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    st.subheader("Result")
    st.write(response.text)