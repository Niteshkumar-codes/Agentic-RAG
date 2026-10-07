from fastapi import FastAPI
from app.api.endpoints import router

app = FastAPI(
    title="Agentic-RAG Backend",
    description="Minimal FastAPI application for Agentic-RAG pipeline",
    version="0.1.0",
)

app.include_router(router)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Agentic-RAG backend is running"
    }
