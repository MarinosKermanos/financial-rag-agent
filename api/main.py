import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from agent.agent import query_agent

load_dotenv()

app = FastAPI(
    title="Financial RAG Agent API",
    description="LLM-powered agent for financial news and forex data",
    version="1.0.0",
)

# Allow Streamlit (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request / Response Models

class ChatRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the current EUR/USD rate and recent news affecting it?"
            }
        }


class ChatResponse(BaseModel):
    question: str
    answer: str


# Routes

@app.get("/")
def root():
    """Health check — useful for Railway to verify the app is running."""
    return {"status": "ok", "service": "financial-rag-agent"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ingest")
def ingest():
    """Trigger news ingestion into Qdrant. Run once after deployment."""
    try:
        from ingestion.fetcher import fetch_all_articles
        from ingestion.chunker import chunk_articles
        from ingestion.embedder import store_chunks
        articles = fetch_all_articles()
        chunks = chunk_articles(articles)
        store_chunks(chunks)
        return {"status": "ok", "chunks_stored": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main endpoint. Takes a question, runs it through the ReAct agent,
    returns the final answer. All steps are traced in LangFuse.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer = query_agent(request.question)
        return ChatResponse(question=request.question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))