import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.title("📄 AI Resume Analyzer")

uploaded_file = st.file_uploader("Upload your resume (TXT only)", type=["txt"])

if uploaded_file:

    resume_text = uploaded_file.read().decode("utf-8")

    st.subheader("Resume Preview")
    st.text(resume_text[:1000])  # preview

    if st.button("Analyze Resume"):

        prompt = f"""
You are an expert career coach.

Analyze the following resume and provide:

1. Strengths
2. Weaknesses
3. Suggestions for improvement
4. Missing skills

Resume:
{resume_text}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        st.subheader("Analysis")
        st.write(response.text)
