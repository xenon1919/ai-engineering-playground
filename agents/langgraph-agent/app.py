import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Load docs
loader = TextLoader("documents/doc1.txt")
documents = loader.load()

# Split text
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Embeddings + DB
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = Chroma.from_documents(docs, embeddings)
retriever = db.as_retriever()

# Tool: Retriever
@tool
def retrieve_info(query: str) -> str:
    """Retrieve relevant info from documents."""
    docs = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in docs])


# LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Create agent
agent = create_react_agent(
    model=llm,
    tools=[retrieve_info]
)

print("LangGraph Agent Ready\n")

while True:

    query = input("Ask: ")

    if query.lower() == "exit":
        break

    result = agent.invoke({
        "messages": [("user", query)]
    })

    print("\nAnswer:\n", result["messages"][-1].content)
    print()