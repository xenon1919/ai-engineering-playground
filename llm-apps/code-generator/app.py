import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.title("⚙️ AI Code Generator")

description = st.text_area("Describe what you want to build")

language = st.selectbox("Programming Language", [
    "Python",
    "JavaScript",
    "Java",
    "C++"
])

if st.button("Generate Code"):

    prompt = f"""
You are an expert programmer.

Generate {language} code based on the following requirement.

Also provide a clear explanation of how the code works.

Requirement:
{description}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    st.subheader("Generated Output")
    st.write(response.text)