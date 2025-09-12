from fastapi import APIRouter
from . import index, review

api_router = APIRouter()
api_router.include_router(index.router)   
api_router.include_router(review.router)

__all__ = ["api_router"]