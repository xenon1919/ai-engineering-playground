import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from data import documents

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text):
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return np.array(response.embeddings[0].values)

# Create embeddings for documents
doc_embeddings = [get_embedding(doc) for doc in documents]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query):
    query_embedding = get_embedding(query)

    similarities = [
        cosine_similarity(query_embedding, emb)
        for emb in doc_embeddings
    ]

    best_match_index = np.argmax(similarities)

    return documents[best_match_index]

print("Semantic Search Ready. Type 'exit' to stop.\n")

while True:
    query = input("Ask something: ")

    if query.lower() == "exit":
        break

    result = search(query)

    print("\nMost relevant document:")
    print(result)
    print()