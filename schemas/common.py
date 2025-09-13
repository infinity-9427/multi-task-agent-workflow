from typing import Generic, Optional, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class Envelope(BaseModel, Generic[T]):
    success: bool = True
    message: str
    data: Optional[T] = None
