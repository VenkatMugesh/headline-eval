from typing import Literal

from pydantic import BaseModel, Field


class Case(BaseModel):
    id: str
    product: str
    audience: str
    tone: Literal["playful", "premium", "urgent", "friendly"]
    max_words: int = Field(gt=0)

class Grade(BaseModel):
    criterion: str
    passed: bool
    detail: str = ""

class JudgeVerdict(BaseModel):
    passed: bool
    reason: str

class CaseResult(BaseModel):
    case_id: str
    version: str
    output: str | None = None
    grades: list[Grade] = []
    error: str | None = None