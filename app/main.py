import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from app.chain import create_rag_chain, ask

load_dotenv()

app = FastAPI(
    title="Telecom RAG Chatbot API",
    description="AI powered customer support using RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load RAG chain once at startup
print("Loading RAG chain at startup...")
rag_chain = create_rag_chain()
print("Chain ready!")


# ---- Models ----
class ChatRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How do I reset my router?"
            }
        }


class SourceDocument(BaseModel):
    content: str
    source: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    source_count: int


# ---- Endpoints ----
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Telecom RAG Chatbot is running"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send a question, get an answer from the telecom knowledge base"""

    # Validation
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    if len(request.question) > 500:
        raise HTTPException(
            status_code=400,
            detail="Question too long. Max 500 characters"
        )

    # Get answer from RAG chain
    result = ask(rag_chain, request.question)

    return ChatResponse(
        answer=result["answer"],
        sources=[SourceDocument(**s) for s in result["sources"]],
        source_count=result["source_count"]
    )