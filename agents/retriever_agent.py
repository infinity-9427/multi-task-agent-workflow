"""
Retriever Agent - Handles semantic retrieval from pgvector database
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

import google.generativeai as genai
from sqlalchemy import text as sql_text

from agents.base_agent import BaseAgent
from database.connection import get_database_session
from schemas.review import RetrievalReport


logger = logging.getLogger(__name__)


class RetrieverAgent(BaseAgent):
    """Agent responsible for retrieving relevant documents via pgvector"""
    
    def __init__(self):
        super().__init__()
        self._setup_gemini()
        
    def _setup_gemini(self):
        """Configure Gemini API"""
        try:
            genai.configure(api_key=self.settings.GEMINI_API_KEY)
            logger.info("RetrieverAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RetrieverAgent: {e}")
            raise
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini"""
        try:
            result = genai.embed_content(
                model=self.settings.EMBEDDING_MODEL,
                content=text,
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.settings.EMBEDDING_DIM
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def process(self, input_data: Dict[str, Any]) -> RetrievalReport:
        """Process retrieval request and return relevant documents from pgvector"""
        try:
            task_details = input_data.get("details", "")
            if not task_details:
                raise ValueError("task details are required")
            
            # Generate query embedding
            query_embedding = self._get_embedding(task_details)
            
            # Query pgvector with Top-K=4
            with get_database_session() as db:
                # Convert list to pgvector format
                embedding_str = f"[{','.join(map(str, query_embedding))}]"
                
                results = db.execute(
                    sql_text("""
                        SELECT id, document_id, text, 
                               (1 - (embedding <=> :query_embedding::vector)) as similarity
                        FROM chunks 
                        WHERE model = :model AND dim = :dim
                        ORDER BY embedding <=> :query_embedding::vector
                        LIMIT :top_k
                    """),
                    {
                        "query_embedding": embedding_str,
                        "model": self.settings.EMBEDDING_MODEL,
                        "dim": self.settings.EMBEDDING_DIM,
                        "top_k": self.settings.TOP_K
                    }
                ).fetchall()
            
            # Process results
            passages = []
            tags = []
            doc_ids = []
            similarities = []
            
            for row in results:
                try:
                    chunk_id, document_id, text, similarity = row
                    
                    # Trim passage to max 1500 chars
                    trimmed_text = text[:1500] if len(text) > 1500 else text
                    passages.append(trimmed_text)
                    
                    # Create tag
                    tag = f"doc:{document_id}#chunk:{chunk_id}"
                    tags.append(tag)
                    
                    # Track doc IDs and similarities
                    if document_id not in doc_ids:
                        doc_ids.append(document_id)
                    similarities.append(float(similarity))
                except Exception as row_error:
                    logger.error(f"Error processing row {row}: {row_error}")
                    continue
            
            # Calculate coverage as mean similarity
            coverage = sum(similarities) / len(similarities) if similarities else 0.0
            
            # Validate lengths
            if len(passages) != len(tags):
                logger.error(f"Length mismatch: passages={len(passages)}, tags={len(tags)}")
                raise ValueError("Passages and tags length mismatch")
            
            return RetrievalReport(
                passages=passages,
                tags=tags,
                doc_ids=doc_ids,
                coverage=round(coverage, 3)
            )
            
        except Exception as e:
            logger.error(f"Error in RetrieverAgent.process: {e}")
            return RetrievalReport(
                passages=[],
                tags=[],
                doc_ids=[],
                coverage=0.0
            )
    
    def get_agent_type(self) -> str:
        """Return the agent type"""
        return "RetrieverAgent"