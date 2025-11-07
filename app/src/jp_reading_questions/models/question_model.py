from pydantic import BaseModel, Field
from typing import List


class Question(BaseModel):
    """A single reading comprehension question."""
    category: str = Field(description="The category of the question: 事実, メインポイント, 暗示されたメッセージ, or 文法や表現")
    question: str = Field(description="The question text in Japanese")
    options: List[str] = Field(description="List of answer options")
    answer: str = Field(description="The correct answer")


class QuestionSet(BaseModel):
    """A set of reading comprehension questions."""
    questions: List[Question] = Field(description="List of generated questions covering all three categories")
