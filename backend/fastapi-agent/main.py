from fastapi import FastAPI
from pydantic import BaseModel
from agent import ask_agent

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    answer = ask_agent(query.question)
    return {"response": answer}