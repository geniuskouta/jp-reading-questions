import json
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback
from typing import Any, Dict, Union

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