import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

load_dotenv()

# Load document
loader = TextLoader("documents/doc1.txt")
documents = loader.load()

# Split text
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Vector DB
db = Chroma.from_documents(docs, embeddings)

# Retriever
retriever = db.as_retriever()

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# RAG Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

print("LangChain RAG Ready\n")

while True:

    query = input("Ask: ")

    if query.lower() == "exit":
        break

    result = qa.run(query)

    print("\nAnswer:\n", result)
    print()