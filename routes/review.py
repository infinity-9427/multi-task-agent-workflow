from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from schemas.review import ReviewRequest
from rag.orchestrator import RAGOrchestrator
import logging

router = APIRouter()
logger = logging.getLogger(__name__)
orchestrator = RAGOrchestrator()


@router.post("/review")
async def create_review(request: ReviewRequest):
    """
    Create a new review task using the RAG orchestrator.
    
    Returns envelope response: {"message": "review completed", "data": {...}}
    
    Request body validation:
    - task_id: Required string 
    - details: Required string 
    """
    try:
        logger.info(f"Received review request for task_id: {request.task_id}")
        
        # Use orchestrator for complete RAG pipeline
        response = await orchestrator.process_review(request.task_id, request.details)
        
        logger.info(f"Review completed for task_id: {request.task_id}, decision: {response['data']['decision']}")
        
        return response
    
    except ValidationError as e:
        logger.warning(f"Validation error for review request: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail=f"Request validation failed: {str(e)}"
        )
        
    except Exception as e:
        logger.error(f"Error processing review request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing review request"
        )