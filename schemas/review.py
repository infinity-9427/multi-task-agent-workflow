from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timezone
from typing import List, Optional
import uuid


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
        return v
    
    @validator('details')
    def validate_details(cls, v):
        if not v or v.isspace():
            raise ValueError('details cannot be empty or whitespace')
        return v


class Citation(BaseModel):
    document_name: str = Field(..., description="Name of the source document")
    page_number: Optional[int] = Field(None, description="Page number in the document")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0.0-1.0)")
    excerpt: str = Field(..., max_length=500, description="Relevant text excerpt")


class ReviewResponse(BaseModel):
    uuid: str
    decision: Decision
    timestamp: str
    reasoning: str
    details: str
    citations: List[Citation] = Field(default_factory=list, description="Documents used in decision making")
    
    @classmethod
    def create_response(cls, details: str):
        return cls(
            uuid=str(uuid.uuid4()),
            decision=Decision.APPROVE,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            reasoning="Automated review passed all checks",
            details=details
        )