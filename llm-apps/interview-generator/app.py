import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.title("🎯 AI Interview Question Generator")

role = st.text_input("Enter Job Role (e.g. Backend Developer, Data Analyst)")

level = st.selectbox("Select Experience Level", [
    "Beginner",
    "Intermediate",
    "Advanced"
])

if st.button("Generate Questions"):

    prompt = f"""
You are an expert interviewer.

Generate 5 interview questions for a {level} {role}.

Also provide clear and concise answers for each question.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    st.subheader("Interview Questions & Answers")
    st.write(response.text)