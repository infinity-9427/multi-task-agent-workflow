"""
Database connection and session management
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool
import logging

logger = logging.getLogger(__name__)

# Database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://task_agent_user:SecurePassword123!@localhost:5432/task_agent_db")

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=StaticPool,
    pool_pre_ping=True,
    echo=False  # Set to True for SQL debugging
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_database_session():
    """Get database session"""
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        db.close()
        raise


def ensure_pgvector_extension():
    """Ensure pgvector extension is enabled"""
    try:
        with engine.connect() as conn:
            # Check if extension exists
            result = conn.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            ).fetchone()
            
            if not result:
                # Create extension if it doesn't exist
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                logger.info("pgvector extension created successfully")
            else:
                logger.info("pgvector extension already exists")
                
    except Exception as e:
        logger.error(f"Failed to ensure pgvector extension: {str(e)}")
        raise


def create_tables():
    """Create all tables"""
    try:
        # First ensure pgvector extension exists
        ensure_pgvector_extension()
        
        # Then create tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise