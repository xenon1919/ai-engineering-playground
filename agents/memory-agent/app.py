import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Load documents
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


@tool
def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"


# -------- LLM -------- #

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# -------- AGENT -------- #

agent = create_react_agent(
    model=llm,
    tools=[retrieve_info, calculator]
)

print("Memory Agent Ready\n")

# 👇 THIS is the memory
chat_history = []

while True:

    query = input("You: ")

    if query.lower() == "exit":
        break

    # Add user message
    chat_history.append(("user", query))

    result = agent.invoke({
        "messages": chat_history
    })

    answer = result["messages"][-1].content

    print("\nAI:", answer, "\n")

    # Store AI response
    chat_history.append(("assistant", answer))