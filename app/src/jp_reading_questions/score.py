import json
import os
from pathlib import Path
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback
from typing import Any, Dict, Union
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

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
def json_format_correct(outputs: Union[Dict, str]) -> Feedback:
    """
    Checks whether the LLM output follows the expected JSON format:
    - Must be a dictionary
    - Must contain 'questions' key with a list of questions
    - Each question must have: category, question, options, answer
    """
    try:
        if isinstance(outputs, dict):
            output_json = outputs
        elif isinstance(outputs, str):
            output_json = json.loads(outputs)
        else:
            raise ValueError(f"Unexpected type: {type(outputs)}")
    except Exception as e:
        return Feedback(
            value="no",
            rationale=f"Output is not valid JSON: {e}"
        )

    # Check if output has questions key
    if not isinstance(output_json, dict) or "questions" not in output_json:
        return Feedback(
            value="no",
            rationale="Output JSON does not contain the required 'questions' key."
        )

    # Check if questions is a list
    questions = output_json["questions"]
    if not isinstance(questions, list):
        return Feedback(
            value="no",
            rationale="'questions' must be a list."
        )

    # Check each question has required fields
    required_fields = ["category", "question", "options", "answer"]
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            return Feedback(
                value="no",
                rationale=f"Question {i} is not a dictionary."
            )
        missing = [f for f in required_fields if f not in q]
        if missing:
            return Feedback(
                value="no",
                rationale=f"Question {i} is missing fields: {missing}"
            )

    return Feedback(
        value="yes",
        rationale=f"Output is valid JSON with {len(questions)} properly formatted questions."
    )

@scorer
def has_all_categories(outputs: Union[Dict, str]) -> Feedback:
    """
    Checks if the output contains questions from all three required categories:
    - 事実 (facts)
    - メインポイント/暗示されたメッセージ (main points/implied messages)
    - 文法や表現 (grammar and expressions)
    """
    try:
        if isinstance(outputs, dict):
            output_json = outputs
        elif isinstance(outputs, str):
            output_json = json.loads(outputs)
        else:
            raise ValueError(f"Unexpected type: {type(outputs)}")
    except Exception as e:
        return Feedback(
            value="no",
            rationale=f"Output is not valid JSON: {e}"
        )

    if "questions" not in output_json or not isinstance(output_json["questions"], list):
        return Feedback(
            value="no",
            rationale="No questions found in output."
        )

    categories = set()
    for q in output_json["questions"]:
        if isinstance(q, dict) and "category" in q:
            categories.add(q["category"])

    expected_categories = {"事実", "メインポイント", "暗示されたメッセージ", "文法", "表現"}
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
def options_are_unique(outputs: Union[Dict, str]) -> Feedback:
    """
    Checks if all answer options within each question are unique (no duplicates).
    """
    try:
        if isinstance(outputs, dict):
            output_json = outputs
        elif isinstance(outputs, str):
            output_json = json.loads(outputs)
        else:
            raise ValueError(f"Unexpected type: {type(outputs)}")
    except Exception as e:
        return Feedback(
            value="no",
            rationale=f"Output is not valid JSON: {e}"
        )

    if "questions" not in output_json or not isinstance(output_json["questions"], list):
        return Feedback(
            value="no",
            rationale="No questions found in output."
        )

    issues = []
    for i, q in enumerate(output_json["questions"]):
        if not isinstance(q, dict) or "options" not in q:
            continue

        options = q.get("options", [])
        if not isinstance(options, list):
            issues.append(f"Question {i}: options is not a list")
            continue

        # Check for duplicate options
        if len(options) != len(set(options)):
            duplicates = [opt for opt in options if options.count(opt) > 1]
            issues.append(f"Question {i}: has duplicate options: {set(duplicates)}")

    if issues:
        return Feedback(
            value="no",
            rationale=f"Found issues with option uniqueness: {'; '.join(issues)}"
        )
    else:
        return Feedback(
            value="yes",
            rationale=f"All {len(output_json['questions'])} questions have unique options."
        )

@scorer
def answer_is_valid(outputs: Union[Dict, str]) -> Feedback:
    """
    Checks if the answer field contains a valid option identifier (A, B, C, or D)
    and that it corresponds to an actual option in the list.
    """
    try:
        if isinstance(outputs, dict):
            output_json = outputs
        elif isinstance(outputs, str):
            output_json = json.loads(outputs)
        else:
            raise ValueError(f"Unexpected type: {type(outputs)}")
    except Exception as e:
        return Feedback(
            value="no",
            rationale=f"Output is not valid JSON: {e}"
        )

    if "questions" not in output_json or not isinstance(output_json["questions"], list):
        return Feedback(
            value="no",
            rationale="No questions found in output."
        )

    issues = []
    valid_answers = {"A", "B", "C", "D"}

    for i, q in enumerate(output_json["questions"]):
        if not isinstance(q, dict):
            continue

        answer = q.get("answer", "")
        options = q.get("options", [])

        # Check if answer is in valid set
        if answer not in valid_answers:
            issues.append(f"Question {i}: answer '{answer}' is not A, B, C, or D")
            continue

        # Check if answer corresponds to an actual option
        if isinstance(options, list):
            answer_index = ord(answer) - ord('A')
            if answer_index >= len(options):
                issues.append(f"Question {i}: answer '{answer}' references option {answer_index + 1} but only {len(options)} options exist")

    if issues:
        return Feedback(
            value="no",
            rationale=f"Found invalid answers: {'; '.join(issues)}"
        )
    else:
        return Feedback(
            value="yes",
            rationale=f"All {len(output_json['questions'])} questions have valid answers."
        )

@scorer
def has_sufficient_questions(outputs: Union[Dict, str]) -> Feedback:
    """
    Checks if the output contains a reasonable number of questions (at least 3).
    For longer texts, expects more questions to provide comprehensive coverage.
    """
    try:
        if isinstance(outputs, dict):
            output_json = outputs
        elif isinstance(outputs, str):
            output_json = json.loads(outputs)
        else:
            raise ValueError(f"Unexpected type: {type(outputs)}")
    except Exception as e:
        return Feedback(
            value="no",
            rationale=f"Output is not valid JSON: {e}"
        )

    if "questions" not in output_json or not isinstance(output_json["questions"], list):
        return Feedback(
            value="no",
            rationale="No questions found in output."
        )

    num_questions = len(output_json["questions"])

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
    def question_text_relevance(outputs: Union[Dict, str], inputs: Dict) -> Feedback:
        """
        Uses LLM-as-judge to evaluate whether the generated questions are relevant
        to the input Japanese text. Checks if questions actually relate to the content.

        Requires: ENABLE_LLM_SCORERS=true
        """
        try:
            if isinstance(outputs, dict):
                output_json = outputs
            elif isinstance(outputs, str):
                output_json = json.loads(outputs)
            else:
                raise ValueError(f"Unexpected type: {type(outputs)}")
        except Exception as e:
            return Feedback(
                value="no",
                rationale=f"Output is not valid JSON: {e}"
            )

        if "questions" not in output_json or not isinstance(output_json["questions"], list):
            return Feedback(
                value="no",
                rationale="No questions found in output."
            )

        jp_text = inputs.get("jp_text", "")
        if not jp_text:
            return Feedback(
                value="no",
                rationale="No input text provided for relevance check."
            )

        # Prepare questions summary
        questions_summary = []
        for i, q in enumerate(output_json["questions"]):
            if isinstance(q, dict) and "question" in q:
                questions_summary.append(f"{i+1}. {q['question']}")

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
    def option_quality(outputs: Union[Dict, str]) -> Feedback:
        """
        Uses LLM-as-judge to evaluate the quality of answer options.
        Checks if options are plausible, distinct, and appropriate difficulty.

        Requires: ENABLE_LLM_SCORERS=true
        """
        try:
            if isinstance(outputs, dict):
                output_json = outputs
            elif isinstance(outputs, str):
                output_json = json.loads(outputs)
            else:
                raise ValueError(f"Unexpected type: {type(outputs)}")
        except Exception as e:
            return Feedback(
                value="no",
                rationale=f"Output is not valid JSON: {e}"
            )

        if "questions" not in output_json or not isinstance(output_json["questions"], list):
            return Feedback(
                value="no",
                rationale="No questions found in output."
            )

        # Prepare questions with options
        questions_with_options = []
        for i, q in enumerate(output_json["questions"]):
            if isinstance(q, dict) and "question" in q and "options" in q:
                options_str = "\n".join(q.get("options", []))
                questions_with_options.append(f"問題{i+1}: {q['question']}\n選択肢:\n{options_str}")

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
    def answer_correctness_check(outputs: Union[Dict, str], inputs: Dict) -> Feedback:
        """
        Uses LLM-as-judge to verify that the marked answer is actually correct
        based on the input text. Checks if the answer key points to the right option.

        Requires: ENABLE_LLM_SCORERS=true
        """
        try:
            if isinstance(outputs, dict):
                output_json = outputs
            elif isinstance(outputs, str):
                output_json = json.loads(outputs)
            else:
                raise ValueError(f"Unexpected type: {type(outputs)}")
        except Exception as e:
            return Feedback(
                value="no",
                rationale=f"Output is not valid JSON: {e}"
            )

        if "questions" not in output_json or not isinstance(output_json["questions"], list):
            return Feedback(
                value="no",
                rationale="No questions found in output."
            )

        jp_text = inputs.get("jp_text", "")
        if not jp_text:
            return Feedback(
                value="no",
                rationale="No input text provided for answer verification."
            )

        # Prepare questions with answers
        questions_detail = []
        for i, q in enumerate(output_json["questions"]):
            if isinstance(q, dict) and all(k in q for k in ["question", "options", "answer"]):
                options_str = "\n".join(q.get("options", []))
                questions_detail.append(
                    f"問題{i+1}: {q['question']}\n選択肢:\n{options_str}\n正解: {q['answer']}"
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