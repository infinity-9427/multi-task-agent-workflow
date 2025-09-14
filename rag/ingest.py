"""
Offline ingestion script for processing PDFs and storing embeddings in pgvector
Run with: python -m rag.ingest or make ingest
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.connection import get_database_session
from settings.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIngestion:
    """Handles offline PDF ingestion with Gemini embeddings and pgvector storage"""
    
    def __init__(self):
        self.settings = get_settings()
        self._setup_gemini()
        
    def _setup_gemini(self):
        """Configure Gemini API"""
        genai.configure(api_key=self.settings.GEMINI_API_KEY)
        
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini embedding model"""
        try:
            result = genai.embed_content(
                model=self.settings.EMBEDDING_MODEL,
                content=text,
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.settings.EMBEDDING_DIM
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
            
    def _load_pdfs(self, data_dir: Path) -> List[Document]:
        """Load and chunk PDFs from data directory"""
        documents = []
        
        for pdf_file in data_dir.glob("*.pdf"):
            try:
                logger.info(f"Loading {pdf_file.name}...")
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                
                # Add metadata
                for i, doc in enumerate(docs):
                    doc.metadata.update({
                        "source": pdf_file.name,
                        "document_id": pdf_file.stem,
                        "page": i + 1,
                        "page_start": i + 1,
                        "page_end": i + 1,
                    })
                
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {pdf_file.name}")
                
            except Exception as e:
                logger.warning(f"Failed to load {pdf_file.name}: {e}")
                
        return documents
        
    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with overlap"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=175,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        return chunks
        
    def _compute_sha256(self, text: str) -> str:
        """Compute SHA256 hash of text content"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
        
    def _upsert_chunks(self, chunks: List[Document]) -> int:
        """Upsert chunks into pgvector database"""
        inserted_count = 0
        
        with get_database_session() as db:
            try:
                for chunk in chunks:
                    # Generate content hash for deduplication
                    content_hash = self._compute_sha256(chunk.page_content)
                    
                    # Check if chunk already exists
                    existing = db.execute(
                        text("SELECT id FROM chunks WHERE sha256 = :hash"),
                        {"hash": content_hash}
                    ).fetchone()
                    
                    if existing:
                        logger.debug(f"Chunk with hash {content_hash[:8]} already exists, skipping")
                        continue
                        
                    # Generate embedding
                    embedding = self._get_embedding(chunk.page_content)
                    
                    # Extract title/section from content if available
                    lines = chunk.page_content.strip().split('\n')
                    title = lines[0][:200] if lines else None
                    
                    # Insert chunk
                    db.execute(
                        text("""
                            INSERT INTO chunks (
                                document_id, page_start, page_end, title, section, text, 
                                embedding, model, dim, task_type, sha256, ingested_at
                            ) VALUES (
                                :document_id, :page_start, :page_end, :title, :section, :text,
                                :embedding, :model, :dim, :task_type, :sha256, :ingested_at
                            )
                        """),
                        {
                            "document_id": hash(chunk.metadata.get("document_id", "unknown")) % 2147483647,
                            "page_start": chunk.metadata.get("page_start"),
                            "page_end": chunk.metadata.get("page_end"),
                            "title": title,
                            "section": chunk.metadata.get("section"),
                            "text": chunk.page_content,
                            "embedding": embedding,
                            "model": self.settings.EMBEDDING_MODEL,
                            "dim": self.settings.EMBEDDING_DIM,
                            "task_type": "RETRIEVAL_DOCUMENT",
                            "sha256": content_hash,
                            "ingested_at": datetime.now()
                        }
                    )
                    inserted_count += 1
                    
                    if inserted_count % 10 == 0:
                        logger.info(f"Processed {inserted_count} chunks...")
                        
                db.commit()
                logger.info(f"Successfully inserted {inserted_count} new chunks")
                
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to upsert chunks: {e}")
                raise
                
        return inserted_count
        
    def ingest_documents(self) -> Dict[str, Any]:
        """Main ingestion workflow"""
        start_time = datetime.now()
        logger.info("Starting document ingestion...")
        
        try:
            # Load PDFs from data directory
            data_dir = Path(__file__).parent.parent / "data"
            if not data_dir.exists():
                raise FileNotFoundError(f"Data directory not found: {data_dir}")
                
            documents = self._load_pdfs(data_dir)
            if not documents:
                logger.warning("No documents found to process")
                return {"status": "no_documents", "processed": 0}
                
            # Chunk documents
            chunks = self._chunk_documents(documents)
            
            # Store in database
            inserted_count = self._upsert_chunks(chunks)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            result = {
                "status": "success",
                "documents_loaded": len(documents),
                "chunks_created": len(chunks),
                "chunks_inserted": inserted_count,
                "duration_seconds": round(duration, 2)
            }
            
            logger.info(f"Ingestion completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return {"status": "error", "error": str(e)}


def main():
    """CLI entrypoint for document ingestion"""
    ingestion = DocumentIngestion()
    result = ingestion.ingest_documents()
    
    if result["status"] == "error":
        exit(1)
        
    print(f"Ingestion completed successfully: {result}")


if __name__ == "__main__":
    main()