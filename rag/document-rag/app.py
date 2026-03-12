import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from utils import chunk_text

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Read document
with open("document.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = chunk_text(text)

# Embedding function
def get_embedding(text):

    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )

    return np.array(response.embeddings[0].values)

# Create embeddings for chunks
chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

def cosine_similarity(a, b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def retrieve(query):

    query_embedding = get_embedding(query)

    similarities = [
        cosine_similarity(query_embedding, emb)
        for emb in chunk_embeddings
    ]

    best_index = np.argmax(similarities)

    return chunks[best_index]

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


print("Document RAG Ready. Type exit to stop.\n")

while True:

    query = input("Ask: ")

    if query.lower() == "exit":
        break

    context = retrieve(query)

    answer = generate_answer(query, context)

    print("\nRetrieved Context:\n", context)
    print("\nAnswer:\n", answer)
    print()