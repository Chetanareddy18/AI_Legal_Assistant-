from fastapi import FastAPI
from pydantic import BaseModel
from rag_watsonx import generate_answer

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.get("/")
def home():
    return {"message": "FastAPI RAG WatsonX is running!"}

@app.post("/ask")
def ask_question(data: QueryRequest):
    try:
        result = generate_answer(data.query, data.top_k)
        return {
            "answer": result["answer"],
            "passages": result["passages"]
        }
    except Exception as e:
        return {"error": str(e)}
