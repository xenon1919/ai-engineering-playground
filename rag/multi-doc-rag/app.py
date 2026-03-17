import os
import chromadb
from dotenv import load_dotenv
from google import genai
from utils import load_documents, chunk_text

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

documents = load_documents("documents")

chunks = []

for doc in documents:
    chunks.extend(chunk_text(doc))

# Create vector DB
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="knowledge")

# Store embeddings
for i, chunk in enumerate(chunks):

    response = client.models.embed_content(
        model="text-embedding-004",
        contents=chunk
    )

    vector = response.embeddings[0].values

    collection.add(
        ids=[str(i)],
        embeddings=[vector],
        documents=[chunk]
    )


def retrieve(query):

    response = client.models.embed_content(
        model="text-embedding-004",
        contents=query
    )

    vector = response.embeddings[0].values

    results = collection.query(
        query_embeddings=[vector],
        n_results=2
    )

    return " ".join(results["documents"][0])


def generate_answer(query, context):

    prompt = f"""
Answer using the context.

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


print("Multi-Document RAG Ready\n")

while True:

    query = input("Ask: ")

    if query.lower() == "exit":
        break

    context = retrieve(query)

    answer = generate_answer(query, context)

    print("\nAnswer:\n", answer)
    print()