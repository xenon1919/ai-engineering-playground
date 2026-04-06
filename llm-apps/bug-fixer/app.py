import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.title("🐞 AI Bug Fixer")

code = st.text_area("Paste your buggy code", height=300)

language = st.selectbox("Programming Language", [
    "Python",
    "JavaScript",
    "Java",
    "C++"
])

if st.button("Fix Code"):

    prompt = f"""
You are an expert software engineer.

Fix the bugs in the following {language} code.

Return:
1. Corrected code
2. Explanation of what was wrong

Code:
{code}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    st.subheader("Fixed Code & Explanation")
    st.write(response.text)