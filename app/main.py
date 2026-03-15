import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

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


# ---- Request Model ----
# This defines what the user SENDS to the API
class ChatRequest(BaseModel):
    question: str

    # Example shown in Swagger UI docs
    class Config:
        json_schema_extra = {
            "example": {
                "question": "How do I reset my router?"
            }
        }


# ---- Response Models ----
# This defines what the API SENDS BACK
class SourceDocument(BaseModel):
    content: str
    source: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    source_count: int


# Health check
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Telecom RAG Chatbot is running"
    }


# Test models endpoint
@app.post("/test-models", response_model=ChatResponse)
def test_models(request: ChatRequest):
    """Temporary endpoint to test models work correctly"""
    return ChatResponse(
        answer=f"You asked: {request.question}",
        sources=[
            SourceDocument(
                content="This is a test source",
                source="test.txt"
            )
        ],
        source_count=1
    )