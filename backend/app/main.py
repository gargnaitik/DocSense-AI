from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from sqlalchemy import text

from app.database import engine, init_db
from app.services.embedding_service import EmbeddingService
from app.routers import auth, documents, chat, search, capture, collections
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables and pgvector extension
    await init_db()
    
    # Pre-warm embedding service
    print("Loading embedding model...")
    EmbeddingService()
    print("Embedding model loaded successfully.")
    
    yield
    # Cleanup if needed

app = FastAPI(
    title="AI Knowledge Search Engine",
    description="Backend API for personal knowledge search with RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(search.router)
app.include_router(capture.router)
app.include_router(collections.router)
@app.get("/")
async def root():
    return {"message": "Knowledge Engine API", "version": "1.0.0"}


# Health check — runs a real query so uptime monitors also count as database
# activity (Supabase pauses free projects after 7 days without any)
@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "ok", "database": "up"}
    except Exception:
        return JSONResponse(status_code=503, content={"status": "degraded", "database": "down"})
