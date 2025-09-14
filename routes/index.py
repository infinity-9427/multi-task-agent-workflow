from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from database.connection import get_database_session
from settings.settings import get_settings

router = APIRouter()

@router.get("/")
def root():
    return {"welcome": "Welcome to Automated Task Review API"}

@router.get("/health")
def health_check():
    """Health check that verifies DB connectivity and chunk data integrity"""
    try:
        settings = get_settings()
        
        with get_database_session() as db:
            # Check DB connectivity
            db.execute(text("SELECT 1"))
            
            # Check chunks table and model/dim consistency
            result = db.execute(
                text("""
                    SELECT COUNT(*) as chunk_count
                    FROM chunks 
                    WHERE model = :model AND dim = :dim
                """),
                {
                    "model": settings.EMBEDDING_MODEL,
                    "dim": settings.EMBEDDING_DIM
                }
            ).fetchone()
            
            chunk_count = result[0] if result else 0
            
            if chunk_count == 0:
                raise HTTPException(
                    status_code=503,
                    detail=f"No chunks found with model={settings.EMBEDDING_MODEL} and dim={settings.EMBEDDING_DIM}"
                )
            
            return {
                "status": "ok",
                "chunks": chunk_count,
                "model": settings.EMBEDDING_MODEL,
                "embedding_dim": settings.EMBEDDING_DIM
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")