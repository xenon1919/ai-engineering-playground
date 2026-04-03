import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.title("💻 AI Code Reviewer")

code = st.text_area("Paste your code here", height=300)

language = st.selectbox("Programming Language", [
    "Python",
    "JavaScript",
    "Java",
    "C++"
])

if st.button("Review Code"):

    prompt = f"""
You are an expert software engineer.

Review the following {language} code and provide:

1. Code quality feedback
2. Possible bugs or issues
3. Suggestions for improvement
4. Best practices

Code:
{code}
"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )

    st.subheader("Review")
    st.write(response.text)