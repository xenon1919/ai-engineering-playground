import os
import streamlit as st
import chromadb
from dotenv import load_dotenv
from google import genai
from utils import read_pdf, chunk_text

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.title("📄 Chat with Your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    text = read_pdf(uploaded_file)
    chunks = chunk_text(text)

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="docs")

    for i, chunk in enumerate(chunks):

        embedding = client.models.embed_content(
            model="text-embedding-004",
            contents=chunk
        )

        vector = embedding.embeddings[0].values

        collection.add(
            ids=[str(i)],
            embeddings=[vector],
            documents=[chunk]
        )

    query = st.text_input("Ask a question")

    if query:

        query_embedding = client.models.embed_content(
            model="text-embedding-004",
            contents=query
        )

        vector = query_embedding.embeddings[0].values

        results = collection.query(
            query_embeddings=[vector],
            n_results=1
        )

        context = results["documents"][0][0]

        prompt = f"""
Answer the question using this context.

Context:
{context}

Question:
{query}
"""

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        st.subheader("Answer")
        st.write(response.text)