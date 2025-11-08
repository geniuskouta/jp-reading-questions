import json
import os
from pathlib import Path
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback
from typing import Any, Dict, Union, List
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError
from jp_reading_questions.models.question_model import QuestionSet, Question

# Pydantic model for structured scorer output
class ScorerJudgment(BaseModel):
    """Structured output for LLM-based scorers."""
    passed: bool = Field(description="Whether the evaluation passed (True) or failed (False)")
    reason: str = Field(description="Explanation for the judgment")

# Helper function to load scorer prompts from markdown files
def load_scorer_prompt(prompt_name: str) -> str:
    """Load a scorer prompt from the prompts/scorers directory."""
    prompts_dir = Path(__file__).parent / "prompts" / "scorers"
    prompt_path = prompts_dir / f"{prompt_name}.md"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

# Check if LLM-based scorers should be enabled (optional, costs money)
ENABLE_LLM_SCORERS = os.getenv("ENABLE_LLM_SCORERS", "false").lower() == "true"

@scorer
def json_format_correct(outputs: Union[List[Dict], list, Dict, str]) -> Feedback:
    """
    Checks whether the LLM output follows the expected format using Pydantic validation.
    Validates against QuestionSet model (list of questions with category, question, options, answer).
    """
    try:
        # Validate using Pydantic model
        question_set = QuestionSet(questions=outputs)
        return Feedback(
            value="yes",
            rationale=f"Output is valid with {len(question_set.questions)} properly formatted questions."
        )
    except ValidationError as e:
        return Feedback(
            value="no",
            rationale=f"Output validation failed: {e}"
        )
    except Exception as e:
        return Feedback(
            value="no",
            rationale=f"Output is not valid: {e}"
        )

@scorer
def has_all_categories(outputs: Union[List[Dict], list, Dict, str]) -> Feedback:
    """
    Checks if the output contains questions from all three required categories:
    - 事実 (facts)
    - メインポイント/暗示されたメッセージ (main points/implied messages)
    - 文法や表現 (grammar and expressions)
    """
    try:
        # Validate using Pydantic model
        question_set = QuestionSet(questions=outputs)
        questions = question_set.questions
    except Exception as e:
        return Feedback(
            value="no",
            rationale=f"Output is not valid: {e}"
        )

    categories = set()
    for q in questions:
        categories.add(q.category)

    # Check if at least one category from each type is present
    has_fact = "事実" in categories
    has_message = any(cat in categories for cat in ["メインポイント", "暗示されたメッセージ"])
    has_grammar = any(cat in categories for cat in ["文法", "表現", "文法や表現"])

    if has_fact and has_message and has_grammar:
        return Feedback(
            value="yes",
            rationale=f"Output contains all required category types. Found categories: {categories}"
        )
    else:
        missing = []
        if not has_fact:
            missing.append("事実")
        if not has_message:
            missing.append("メインポイント/暗示されたメッセージ")
        if not has_grammar:
            missing.append("文法や表現")
        return Feedback(
            value="no",
            rationale=f"Missing categories: {missing}. Found categories: {categories}"
        )

@scorer
def options_are_unique(outputs: Union[List[Dict], list, Dict, str]) -> Feedback:
    """
    Checks if all answer options within each question are unique (no duplicates).
    """
    try:
        # Validate using Pydantic model
        question_set = QuestionSet(questions=outputs)
        questions = question_set.questions
    except Exception as e:
        return Feedback(
            value="no",
            rationale=f"Output is not valid: {e}"
        )

    issues = []
    for i, q in enumerate(questions):
        # Check for duplicate options
        if len(q.options) != len(set(q.options)):
            duplicates = [opt for opt in q.options if q.options.count(opt) > 1]
            issues.append(f"Question {i}: has duplicate options: {set(duplicates)}")

    if issues:
        return Feedback(
            value="no",
            rationale=f"Found issues with option uniqueness: {'; '.join(issues)}"
        )
    else:
        return Feedback(
            value="yes",
            rationale=f"All {len(questions)} questions have unique options."
        )

@scorer
def answer_is_valid(outputs: Union[List[Dict], list, Dict, str]) -> Feedback:
    """
    Checks if the answer field contains a valid option identifier (A, B, C, or D)
    and that it corresponds to an actual option in the list.
    """
    try:
        # Validate using Pydantic model
        question_set = QuestionSet(questions=outputs)
        questions = question_set.questions
    except Exception as e:
        return Feedback(
            value="no",
            rationale=f"Output is not valid: {e}"
        )

    issues = []
    valid_answers = {"A", "B", "C", "D"}

    for i, q in enumerate(questions):
        # Check if answer is in valid set
        if q.answer not in valid_answers:
            issues.append(f"Question {i}: answer '{q.answer}' is not A, B, C, or D")
            continue

        # Check if answer corresponds to an actual option
        answer_index = ord(q.answer) - ord('A')
        if answer_index >= len(q.options):
            issues.append(f"Question {i}: answer '{q.answer}' references option {answer_index + 1} but only {len(q.options)} options exist")

    if issues:
        return Feedback(
            value="no",
            rationale=f"Found invalid answers: {'; '.join(issues)}"
        )
    else:
        return Feedback(
            value="yes",
            rationale=f"All {len(questions)} questions have valid answers."
        )

@scorer
def has_sufficient_questions(outputs: Union[List[Dict], list, Dict, str]) -> Feedback:
    """
    Checks if the output contains a reasonable number of questions (at least 3).
    For longer texts, expects more questions to provide comprehensive coverage.
    """
    try:
        # Validate using Pydantic model
        question_set = QuestionSet(questions=outputs)
        questions = question_set.questions
    except Exception as e:
        return Feedback(
            value="no",
            rationale=f"Output is not valid: {e}"
        )

    num_questions = len(questions)

    # Expect at least 3 questions
    if num_questions < 3:
        return Feedback(
            value="no",
            rationale=f"Only {num_questions} questions generated. Expected at least 3 for adequate coverage."
        )

    return Feedback(
        value="yes",
        rationale=f"Generated {num_questions} questions, which provides good coverage."
    )

# ============================================================================
# LLM-based scorers (optional, enabled via ENABLE_LLM_SCORERS env variable)
# These scorers use GPT to judge semantic quality and cost money per evaluation
# ============================================================================

if ENABLE_LLM_SCORERS:
    @scorer
    def question_text_relevance(outputs: Union[List[Dict], list, Dict, str], inputs: Dict) -> Feedback:
        """
        Uses LLM-as-judge to evaluate whether the generated questions are relevant
        to the input Japanese text. Checks if questions actually relate to the content.

        Requires: ENABLE_LLM_SCORERS=true
        """
        try:
            # Validate using Pydantic model
            question_set = QuestionSet(questions=outputs)
            questions = question_set.questions
        except Exception as e:
            return Feedback(
                value="no",
                rationale=f"Output is not valid: {e}"
            )

        jp_text = inputs.get("jp_text", "")
        if not jp_text:
            return Feedback(
                value="no",
                rationale="No input text provided for relevance check."
            )

        # Prepare questions summary
        questions_summary = []
        for i, q in enumerate(questions):
            questions_summary.append(f"{i+1}. {q.question}")

        questions_str = "\n".join(questions_summary)

        # Load prompt template and format
        prompt_template = load_scorer_prompt("question_relevance")
        prompt = prompt_template.format(jp_text=jp_text, questions_str=questions_str)

        # Use LLM with structured output
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(ScorerJudgment)

        try:
            result: ScorerJudgment = llm.invoke(prompt)
            return Feedback(
                value="yes" if result.passed else "no",
                rationale=f"Question relevance: {result.reason}"
            )
        except Exception as e:
            return Feedback(
                value="no",
                rationale=f"Error during LLM evaluation: {e}"
            )

    @scorer
    def option_quality(outputs: Union[List[Dict], list, Dict, str]) -> Feedback:
        """
        Uses LLM-as-judge to evaluate the quality of answer options.
        Checks if options are plausible, distinct, and appropriate difficulty.

        Requires: ENABLE_LLM_SCORERS=true
        """
        try:
            # Validate using Pydantic model
            question_set = QuestionSet(questions=outputs)
            questions = question_set.questions
        except Exception as e:
            return Feedback(
                value="no",
                rationale=f"Output is not valid: {e}"
            )

        # Prepare questions with options
        questions_with_options = []
        for i, q in enumerate(questions):
            options_str = "\n".join(q.options)
            questions_with_options.append(f"問題{i+1}: {q.question}\n選択肢:\n{options_str}")

        questions_str = "\n\n".join(questions_with_options)

        # Load prompt template and format
        prompt_template = load_scorer_prompt("option_quality")
        prompt = prompt_template.format(questions_str=questions_str)

        # Use LLM with structured output
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(ScorerJudgment)

        try:
            result: ScorerJudgment = llm.invoke(prompt)
            return Feedback(
                value="yes" if result.passed else "no",
                rationale=f"Option quality: {result.reason}"
            )
        except Exception as e:
            return Feedback(
                value="no",
                rationale=f"Error during LLM evaluation: {e}"
            )

    @scorer
    def answer_correctness_check(outputs: Union[List[Dict], list, Dict, str], inputs: Dict) -> Feedback:
        """
        Uses LLM-as-judge to verify that the marked answer is actually correct
        based on the input text. Checks if the answer key points to the right option.

        Requires: ENABLE_LLM_SCORERS=true
        """
        try:
            # Validate using Pydantic model
            question_set = QuestionSet(questions=outputs)
            questions = question_set.questions
        except Exception as e:
            return Feedback(
                value="no",
                rationale=f"Output is not valid: {e}"
            )

        jp_text = inputs.get("jp_text", "")
        if not jp_text:
            return Feedback(
                value="no",
                rationale="No input text provided for answer verification."
            )

        # Prepare questions with answers
        questions_detail = []
        for i, q in enumerate(questions):
            options_str = "\n".join(q.options)
            questions_detail.append(
                f"問題{i+1}: {q.question}\n選択肢:\n{options_str}\n正解: {q.answer}"
            )

        questions_str = "\n\n".join(questions_detail)

        # Load prompt template and format
        prompt_template = load_scorer_prompt("answer_correctness")
        prompt = prompt_template.format(jp_text=jp_text, questions_str=questions_str)

        # Use LLM with structured output
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(ScorerJudgment)

        try:
            result: ScorerJudgment = llm.invoke(prompt)
            return Feedback(
                value="yes" if result.passed else "no",
                rationale=f"Answer correctness: {result.reason}"
            )
        except Exception as e:
            return Feedback(
                value="no",
                rationale=f"Error during LLM evaluation: {e}"
            )