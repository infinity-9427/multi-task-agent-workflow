from fastapi import FastAPI
from routes import api_router

app = FastAPI(
    title="Automated Task Review Agent",
    description="pgvector-based RAG system for task review and approval",
    version="1.0.0"
)

app.include_router(api_router)