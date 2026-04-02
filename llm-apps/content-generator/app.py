import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.title("✍️ AI Content Generator")

topic = st.text_input("Enter topic (e.g. RAG, AI Agents, Web Dev)")

platform = st.selectbox("Choose Platform", [
    "LinkedIn",
    "Twitter"
])

tone = st.selectbox("Tone", [
    "Professional",
    "Casual",
    "Inspirational"
])

def build_prompt(topic, platform, tone):

    if platform == "LinkedIn":
        return f"""
Write a {tone} LinkedIn post about {topic}.

Make it engaging, structured, and insightful.
Add a strong hook at the beginning.
"""

    elif platform == "Twitter":
        return f"""
Write a {tone} Twitter post about {topic}.

Keep it short, punchy, and engaging.
Max 280 characters.
"""

if st.button("Generate Content"):

    prompt = build_prompt(topic, platform, tone)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    st.subheader("Generated Post")
    st.write(response.text)