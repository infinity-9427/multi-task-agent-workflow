"""
Rate-limited ingestion script with retry logic for handling API quotas
Run with: python -m rag.ingest_with_retry
"""

import os
import hashlib
import logging
import time
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


class RateLimitedIngestion:
    """Rate-limited PDF ingestion with retry logic for API quotas"""
    
    def __init__(self):
        self.settings = get_settings()
        self._setup_gemini()
        self.delay_seconds = 2  # Delay between API calls
        self.max_retries = 3
        
    def _setup_gemini(self):
        """Configure Gemini API"""
        genai.configure(api_key=self.settings.GEMINI_API_KEY)
        
    def _get_embedding_with_retry(self, text: str, chunk_num: int = 0) -> List[float]:
        """Generate embedding with exponential backoff retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Add delay to respect rate limits
                if chunk_num > 0:
                    time.sleep(self.delay_seconds + (attempt * 2))
                
                result = genai.embed_content(
                    model=self.settings.EMBEDDING_MODEL,
                    content=text,
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self.settings.EMBEDDING_DIM
                )
                return result['embedding']
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    wait_time = (2 ** attempt) * 10  # Exponential backoff: 10s, 20s, 40s
                    logger.warning(f"Rate limit hit on chunk {chunk_num}, attempt {attempt + 1}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    
                    if attempt == self.max_retries - 1:
                        logger.error(f"Max retries exceeded for chunk {chunk_num}: {e}")
                        raise
                else:
                    logger.error(f"Non-rate-limit error for chunk {chunk_num}: {e}")
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
        """Split documents into smaller chunks to reduce API calls"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # Smaller chunks to reduce API usage
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        return chunks
        
    def _compute_sha256(self, text: str) -> str:
        """Compute SHA256 hash of text content"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
        
    def _upsert_chunks_batch(self, chunks: List[Document], batch_size: int = 20) -> int:
        """Upsert chunks in small batches with individual transaction commits"""
        total_inserted = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} ({len(batch)} chunks)...")
            
            batch_inserted = 0
            
            for j, chunk in enumerate(batch):
                try:
                    with get_database_session() as db:
                        # Generate content hash for deduplication
                        content_hash = self._compute_sha256(chunk.page_content)
                        
                        # Check if chunk already exists
                        existing = db.execute(
                            text("SELECT id FROM chunks WHERE sha256 = :hash"),
                            {"hash": content_hash}
                        ).fetchone()
                        
                        if existing:
                            logger.debug(f"Chunk {total_inserted + j + 1} already exists, skipping")
                            continue
                            
                        # Generate embedding with retry logic
                        chunk_num = total_inserted + j + 1
                        embedding = self._get_embedding_with_retry(chunk.page_content, chunk_num)
                        
                        # Extract title/section from content
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
                        db.commit()
                        batch_inserted += 1
                        total_inserted += 1
                        
                        logger.info(f"✅ Successfully processed chunk {chunk_num}/{len(chunks)}")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to process chunk {total_inserted + j + 1}: {e}")
                    # Continue with next chunk instead of failing entire batch
                    continue
            
            logger.info(f"Completed batch: {batch_inserted}/{len(batch)} chunks inserted")
            
            # Small break between batches
            if i + batch_size < len(chunks):
                logger.info("Taking a short break between batches...")
                time.sleep(5)
                
        return total_inserted
        
    def ingest_documents(self) -> Dict[str, Any]:
        """Main ingestion workflow with rate limiting"""
        start_time = datetime.now()
        logger.info("Starting rate-limited document ingestion...")
        
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
            
            # Store in database with rate limiting
            inserted_count = self._upsert_chunks_batch(chunks, batch_size=15)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            result = {
                "status": "success",
                "documents_loaded": len(documents),
                "chunks_created": len(chunks),
                "chunks_inserted": inserted_count,
                "duration_seconds": round(duration, 2),
                "success_rate": f"{(inserted_count/len(chunks)*100):.1f}%"
            }
            
            logger.info(f"✅ Ingestion completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Ingestion failed: {e}")
            return {"status": "error", "error": str(e)}


def main():
    """CLI entrypoint for rate-limited document ingestion"""
    ingestion = RateLimitedIngestion()
    result = ingestion.ingest_documents()
    
    if result["status"] == "error":
        exit(1)
        
    print(f"✅ Rate-limited ingestion completed: {result}")


if __name__ == "__main__":
    main()