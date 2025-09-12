from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def root():
    return {"welcome": "Welcome to Automated Task Review API"}