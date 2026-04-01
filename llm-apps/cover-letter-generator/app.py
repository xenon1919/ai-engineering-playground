import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.title("✉️ AI Cover Letter Generator")

job_description = st.text_area("Paste Job Description")

resume_text = st.text_area("Paste Your Resume")

tone = st.selectbox("Tone", [
    "Professional",
    "Confident",
    "Enthusiastic"
])

if st.button("Generate Cover Letter"):

    prompt = f"""
You are an expert career assistant.

Write a {tone} cover letter based on the following:

Job Description:
{job_description}

Candidate Resume:
{resume_text}

The cover letter should be well-structured and personalized.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    st.subheader("Generated Cover Letter")
    st.write(response.text)