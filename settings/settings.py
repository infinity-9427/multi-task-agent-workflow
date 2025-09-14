from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    # LLM Configuration
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "768"))
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    
    # Retrieval Configuration
    TOP_K: int = int(os.getenv("TOP_K", "4"))
    COVERAGE_THRESHOLD: float = float(os.getenv("COVERAGE_THRESHOLD", "0.35"))
    APPROVAL_COVERAGE_MIN: float = float(os.getenv("APPROVAL_COVERAGE_MIN", "0.45"))
    MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "5000"))
    
    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    
    def __init__(self):
        # Fail fast validation
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required")
        if not self.DATABASE_URL:
            raise ValueError("DATABASE_URL is required")
        
        # Validate model and dimension consistency
        expected_models = ["gemini-embedding-001", "text-embedding-004"]
        if self.EMBEDDING_MODEL not in expected_models:
            raise ValueError(f"EMBEDDING_MODEL must be one of {expected_models}")
        
        if self.EMBEDDING_MODEL == "gemini-embedding-001" and self.EMBEDDING_DIM != 768:
            raise ValueError("gemini-embedding-001 requires EMBEDDING_DIM=768")


def get_settings():
    """Get settings singleton"""
    if not hasattr(get_settings, "_settings"):
        get_settings._settings = Settings()
    return get_settings._settings