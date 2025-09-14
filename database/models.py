"""
Database models for document storage with pgvector
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import uuid

from database.connection import Base


class DocumentEmbedding(Base):
    """Store document chunks with their vector embeddings"""
    
    __tablename__ = "document_embeddings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Document metadata
    source_document = Column(String(255), nullable=False, index=True)
    page_number = Column(Integer, nullable=True)
    chunk_index = Column(Integer, nullable=False)
    
    # Content
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False, unique=True, index=True)
    
    # Vector embedding (768 dimensions for Google's embedding models)
    embedding = Column(Vector(768), nullable=False)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Document statistics
    content_length = Column(Integer, nullable=False)
    
    def __repr__(self):
        return f"<DocumentEmbedding(source='{self.source_document}', page={self.page_number}, chunk={self.chunk_index})>"


class DocumentMetadata(Base):
    """Store document-level metadata"""
    
    __tablename__ = "document_metadata"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Document info
    filename = Column(String(255), nullable=False, unique=True, index=True)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_hash = Column(String(64), nullable=False, unique=True)
    
    # Processing info
    total_pages = Column(Integer, nullable=True)
    total_chunks = Column(Integer, nullable=False, default=0)
    processing_status = Column(String(50), nullable=False, default='pending')  # pending, processing, completed, failed
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_processed_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<DocumentMetadata(filename='{self.filename}', status='{self.processing_status}')>"