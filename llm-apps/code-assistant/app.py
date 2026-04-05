import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.title("🤖 AI Code Assistant")

mode = st.selectbox("Choose Mode", [
    "Generate Code",
    "Review Code"
])

language = st.selectbox("Programming Language", [
    "Python",
    "JavaScript",
    "Java",
    "C++"
])

# -------- GENERATE MODE -------- #

if mode == "Generate Code":

    description = st.text_area("Describe what you want to build")

    if st.button("Generate"):

        prompt = f"""
You are an expert programmer.

Generate {language} code for the following requirement.

Also explain how the code works.

Requirement:
{description}
"""

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        st.subheader("Generated Code & Explanation")
        st.write(response.text)

# -------- REVIEW MODE -------- #

elif mode == "Review Code":

    code = st.text_area("Paste your code here", height=300)

    if st.button("Review"):

        prompt = f"""
You are an expert software engineer.

Review this {language} code and provide:

1. Code quality feedback
2. Bugs or issues
3. Improvements
4. Best practices

Code:
{code}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        st.subheader("Code Review")
        st.write(response.text)