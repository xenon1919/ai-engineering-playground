import os
from dotenv import load_dotenv
from google import genai
import chromadb
from utils import chunk_text

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Read document
with open("document.txt", "r") as f:
    text = f.read()

chunks = chunk_text(text)

# Create ChromaDB client
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="documents")

# Store embeddings
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


def retrieve(query):

    query_embedding = client.models.embed_content(
        model="text-embedding-004",
        contents=query
    )

    vector = query_embedding.embeddings[0].values

    results = collection.query(
        query_embeddings=[vector],
        n_results=1
    )

    return results["documents"][0][0]


def generate_answer(query, context):

    prompt = f"""
Answer the question using the context.

Context:
{context}

Question:
{query}
"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )

    return response.text


print("Chroma RAG Ready\n")

while True:

    query = input("Ask: ")

    if query.lower() == "exit":
        break

    context = retrieve(query)

    answer = generate_answer(query, context)

    print("\nRetrieved Context:", context)
    print("\nAnswer:", answer)
    print()