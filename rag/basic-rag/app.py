import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from data import documents

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Embedding function
def get_embedding(text):
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return np.array(response.embeddings[0].values)

# Create document embeddings
doc_embeddings = [get_embedding(doc) for doc in documents]

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Retrieve relevant document
def retrieve(query):
    query_embedding = get_embedding(query)

    similarities = [
        cosine_similarity(query_embedding, emb)
        for emb in doc_embeddings
    ]

    best_index = np.argmax(similarities)

    return documents[best_index]

# Generate answer using context
def generate_answer(query, context):

    prompt = f"""
Use the following context to answer the question.

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


print("Mini RAG System Ready. Type 'exit' to stop.\n")

while True:

    query = input("Ask: ")

    if query.lower() == "exit":
        break

    context = retrieve(query)

    answer = generate_answer(query, context)

    print("\nRetrieved Context:", context)
    print("\nAnswer:", answer)
    print()