import os
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# Get the directory where this file is located
CURRENT_DIR = Path(__file__).parent
PROMPTS_DIR = CURRENT_DIR / "prompts"


def load_prompt(filename: str) -> str:
    """Load a prompt from a markdown file."""
    prompt_path = PROMPTS_DIR / filename
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


# Load prompts from markdown files
SYSTEM_PROMPT = load_prompt("system.md")
USER_PROMPT = load_prompt("user.md")


# Define Pydantic models for structured output
class Question(BaseModel):
    """A single reading comprehension question."""
    category: str = Field(description="The category of the question: 事実, メインポイント, 暗示されたメッセージ, or 文法や表現")
    question: str = Field(description="The question text in Japanese")
    options: List[str] = Field(description="List of answer options")
    answer: str = Field(description="The correct answer")


class QuestionSet(BaseModel):
    """A set of reading comprehension questions."""
    questions: List[Question] = Field(description="List of generated questions covering all three categories")


# Create the LLM with structured output
llm = ChatOpenAI(
    model="gpt-5-mini",
    api_key=os.getenv('OPENAI_API_KEY'),
    temperature=1.0
)

# Use with_structured_output to enforce the schema
structured_llm = llm.with_structured_output(QuestionSet)

# Create the prompt template using loaded prompts
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", USER_PROMPT)
])

# Create the chain
chain = prompt_template | structured_llm


def predict_fn(jp_text: str) -> str:
    """
    Generate Japanese reading comprehension questions from the given text.

    Args:
        jp_text: The Japanese text to generate questions from

    Returns:
        JSON string with the generated questions.
    """
    result = chain.invoke({"jp_text": jp_text})
    # Convert the Pydantic model to dict and then to JSON string
    return result.model_dump_json()