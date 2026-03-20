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

splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = Chroma.from_documents(docs, embeddings)
retriever = db.as_retriever()

# -------- TOOLS -------- #

@tool
def retrieve_info(query: str) -> str:
    """Use this tool to fetch information from documents."""
    docs = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in docs])


@tool
def simple_calculator(expression: str) -> str:
    """Evaluate simple math expressions like 2+2 or 10*5."""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"


@tool
def explain_concept(concept: str) -> str:
    """Explain a concept in simple terms."""
    return f"{concept} is explained as a simple concept in AI context."


# -------- LLM -------- #

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# -------- AGENT -------- #

agent = create_react_agent(
    model=llm,
    tools=[retrieve_info, simple_calculator, explain_concept]
)

print("Multi-Tool Agent Ready\n")

while True:

    query = input("Ask: ")

    if query.lower() == "exit":
        break

    result = agent.invoke({
        "messages": [("user", query)]
    })

    print("\nAnswer:\n", result["messages"][-1].content)
    print()