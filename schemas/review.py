from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import uuid
import re


class Decision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"


class ReviewRequest(BaseModel):
    task_id: str = Field(
        ..., 
        min_length=1, 
        max_length=255,
        strip_whitespace=True,
        description="Unique identifier for the task"
    )
    details: str = Field(
        ..., 
        min_length=1, 
        max_length=10000,
        strip_whitespace=True,
        description="Task details for review"
    )
    
    @validator('task_id')
    def validate_task_id(cls, v):
        if not v or v.isspace():
            raise ValueError('task_id cannot be empty or whitespace')
        # Check for HTML/binary content
        if re.search(r'<[^>]+>', v) or any(ord(c) > 127 for c in v if not c.isascii()):
            raise ValueError('task_id cannot contain HTML or binary data')
        return v
    
    @validator('details')
    def validate_details(cls, v):
        if not v or v.isspace():
            raise ValueError('details cannot be empty or whitespace')
        # Check for HTML/binary content
        if re.search(r'<[^>]+>', v):
            raise ValueError('details cannot contain HTML content')
        # Ensure UTF-8 compatibility
        try:
            v.encode('utf-8')
        except UnicodeEncodeError:
            raise ValueError('details must be valid UTF-8')
        return v


class RetrievedPassage(BaseModel):
    """Raw passage returned by Retriever specialist"""
    content: str = Field(..., max_length=2000, description="Raw passage text, trimmed to safe length")
    tag: str = Field(..., description="Exact tag like 'doc:2#chunk:15'")
    document_id: str = Field(..., description="Unique document identifier")


class RetrieverResult(BaseModel):
    """Output from Retriever specialist"""
    passages: List[RetrievedPassage] = Field(default_factory=list, description="Retrieved passages with exact tags")
    document_ids: List[str] = Field(default_factory=list, description="Unique document IDs involved")
    coverage_score: float = Field(..., ge=0.0, le=1.0, description="Coverage score reflecting how directly passages speak to task")
    analysis_summary: Optional[str] = Field(None, description="Optional analysis summary from Retriever")


class RetrievalReport(BaseModel):
    """Output from pgvector-based retrieval"""
    passages: List[str] = Field(default_factory=list, description="Raw chunk texts trimmed to max 1500 chars")
    tags: List[str] = Field(default_factory=list, description="Tags in format doc:<document_id>#chunk:<id>")
    doc_ids: List[int] = Field(default_factory=list, description="Unique list of document_id values")
    coverage: float = Field(..., ge=0.0, le=1.0, description="Mean cosine similarity of returned Top-K")


class RequiredAction(BaseModel):
    """Action required to proceed with rejected task"""
    action: str = Field(..., description="Specific action needed")
    description: str = Field(..., description="Detailed description of what needs to be done")


class DecisionResult(BaseModel):
    """Output from Decision agent"""
    decision: Decision = Field(..., description="Decision: approve, reject, or reject_due_to_insufficient_context")
    rationale: str = Field(..., description="Evidence-based rationale citing only provided tags")
    cited_tags: List[str] = Field(default_factory=list, description="Tags cited in the decision")
    required_actions: List[RequiredAction] = Field(default_factory=list, description="Actions needed for rejected tasks")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level in the decision")


class Citation(BaseModel):
    document_name: str = Field(..., description="Name of the source document")
    page_number: Optional[int] = Field(None, description="Page number in the document")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0.0-1.0)")
    excerpt: str = Field(..., max_length=500, description="Relevant text excerpt")
    tag: str = Field(..., description="Exact tag for this citation")
    
    @classmethod
    def from_retrieved_passage(cls, passage: "RetrievedPassage", document_name: str = None, page_number: int = None):
        """Create Citation from RetrievedPassage"""
        # Properly truncate excerpt to 500 characters max
        excerpt = passage.content[:497] + "..." if len(passage.content) > 500 else passage.content
        
        return cls(
            document_name=document_name or passage.document_id,
            page_number=page_number,
            relevance_score=0.8,  # Default relevance score
            excerpt=excerpt,
            tag=passage.tag
        )


class ReviewResponse(BaseModel):
    # Legacy field for backward compatibility
    uuid: str = Field(..., description="Unique identifier for the review response")
    
    task_id: str = Field(..., description="Original task ID")
    decision: Decision = Field(..., description="Final decision")
    timestamp: str = Field(..., description="ISO timestamp")
    rationale: str = Field(..., description="Evidence-based decision rationale")
    cited_tags: List[str] = Field(default_factory=list, description="Tags cited in decision")
    required_actions: List[RequiredAction] = Field(default_factory=list, description="Actions needed if rejected")
    document_ids: List[str] = Field(default_factory=list, description="Document IDs used in decision")
    coverage_score: float = Field(..., ge=0.0, le=1.0, description="Coverage score from retrieval")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Decision confidence")
    latency_ms: int = Field(..., description="Processing latency in milliseconds")
    
    # Keep citations for backward compatibility
    citations: List[Citation] = Field(default_factory=list, description="Supporting documents with tags")
    
    @classmethod
    def create_response(
        cls, 
        task_id: str,
        decision_result: DecisionResult,
        retriever_result: RetrieverResult,
        latency_ms: int,
        citations: List[Citation] = None
    ):
        import uuid
        
        return cls(
            uuid=str(uuid.uuid4()),  # Generate UUID for backward compatibility
            task_id=task_id,
            decision=decision_result.decision,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            rationale=decision_result.rationale,
            cited_tags=decision_result.cited_tags,
            required_actions=decision_result.required_actions,
            document_ids=retriever_result.document_ids,
            coverage_score=retriever_result.coverage_score,
            confidence=decision_result.confidence,
            latency_ms=latency_ms,
            citations=citations or []
        )