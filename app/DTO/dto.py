from pydantic import BaseModel
from typing import Optional

class QuestionRequest(BaseModel):
    category: Optional[str] = None
    filename: str
    question: str


