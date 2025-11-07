import dspy
from jp_reading_questions.models.question_model import QuestionSet

class QuestionSignature(dspy.Signature):
    """Generate Japanese reading comprehension questions for JLPT test preparation.

    Create questions that test:
    1. 事実 (Facts): Explicit information from the text
    2. メインポイント/暗示されたメッセージ (Main points/Implied messages): Inference and understanding
    3. 文法や表現 (Grammar and expressions): Language usage

    Each question must have 4 plausible options (A, B, C, D) with one correct answer.
    Questions should be appropriate for Japanese language learners.
    """

    jp_text: str = dspy.InputField(desc="Japanese reading text (news article, story, etc.)")
    question_set: QuestionSet = dspy.OutputField(desc="JSON with questions array containing category, question, options, answer")

class QuestionGenerator(dspy.Module):
    """DSPy module for generating Japanese comprehension questions."""

    def __init__(self):
        super().__init__()
        # ChainOfThought adds reasoning steps before generating output
        self.generate = dspy.ChainOfThought(QuestionSignature)

    def forward(self, jp_text: str) -> str:
        """Generate questions for the given Japanese text.

        Args:
            jp_text: Japanese reading passage

        Returns:
            JSON string with questions array
        """
        # DSPy handles the prompt automatically
        result = self.generate(jp_text=jp_text)

        # DSPy returns a QuestionSet object directly
        try:
            question_set = result.question_set
            # Convert Pydantic model to JSON string
            return question_set.model_dump_json(indent=2, ensure_ascii=False)

        except Exception as e:
            # Return error-wrapped result
            return json.dumps({
                "error": "Failed to parse output",
                "raw_output": str(result.question_set) if hasattr(result, 'question_set') else str(result),
                "exception": str(e)
            }, ensure_ascii=False)
