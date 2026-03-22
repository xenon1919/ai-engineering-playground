import os
import json
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

MEMORY_FILE = "memory.json"

# -------- LOAD MEMORY -------- #

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

chat_history = load_memory()

# -------- DOCUMENT SETUP -------- #

loader = TextLoader("documents/doc1.txt")
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = Chroma.from_documents(docs, embeddings)
retriever = db.as_retriever()

# -------- TOOLS -------- #

@tool
def retrieve_info(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in docs])


# -------- LLM -------- #

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# -------- AGENT -------- #

agent = create_react_agent(
    model=llm,
    tools=[retrieve_info]
)

print("Persistent Memory Agent Ready\n")

while True:

    query = input("You: ")

    if query.lower() == "exit":
        break

    chat_history.append(("user", query))

    result = agent.invoke({
        "messages": chat_history
    })

    answer = result["messages"][-1].content

    print("\nAI:", answer, "\n")

    chat_history.append(("assistant", answer))

    # SAVE MEMORY AFTER EVERY MESSAGE
    save_memory(chat_history)