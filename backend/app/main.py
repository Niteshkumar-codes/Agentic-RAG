from fastapi import FastAPI

app = FastAPI(
    title="Agentic-RAG Backend",
    description="Minimal FastAPI application for Agentic-RAG pipeline",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Agentic-RAG backend is running"
    }
