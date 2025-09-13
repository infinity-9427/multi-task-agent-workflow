from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from schemas.review import ReviewRequest, ReviewResponse
from repositories.review_repository import ReviewRepository
import logging

router = APIRouter()
logger = logging.getLogger(__name__)
review_repository = ReviewRepository()


@router.post("/review", response_model=ReviewResponse)
async def create_review(request: ReviewRequest):
    """
    Create a new review task.
    
    Validates the request and returns the response.
    
    Request body validation:
    - task_id: Required string 
    - details: Required string 
    """
    try:
        logger.info(f"Received review request for task_id: {request.task_id}")
        
        response = review_repository.process_review(request)
        
        logger.info(f"Generated review uuid: {response.uuid} for task_id: {request.task_id}")
        
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