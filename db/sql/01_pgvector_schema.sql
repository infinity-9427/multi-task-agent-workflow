-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop existing tables if they exist (for clean migrations)
DROP TABLE IF EXISTS chunks CASCADE;
DROP TABLE IF EXISTS document_embeddings CASCADE;
DROP TABLE IF EXISTS document_metadata CASCADE;

-- Create chunks table with pgvector support
CREATE TABLE chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    title TEXT,
    section TEXT,
    text TEXT NOT NULL,
    embedding vector(768),
    model TEXT,
    dim INTEGER,
    task_type TEXT,
    sha256 TEXT,
    ingested_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create ANN index for cosine similarity
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_cos 
ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create additional indexes for common queries
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_model_dim ON chunks (model, dim);
CREATE INDEX IF NOT EXISTS idx_chunks_sha256 ON chunks (sha256);