from fastapi import APIRouter

router = APIRouter()

@router.get("/review")
def generate_review():
    return { "Response": "OK"}
