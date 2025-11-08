"""
Unified prediction module that switches between LangChain and DSPy based on USE_DSPY env variable.
"""
import os
import dspy
from jp_reading_questions.agent import Agent
from jp_reading_questions.prompts.dspy.question_dspy import QuestionGenerator
from jp_reading_questions.models.question_model import QuestionSet

USE_DSPY = os.getenv('USE_DSPY', 'True').lower() in ('true')

def predict_fn(jp_text: str) -> list:
    """Prediction function compatible with MLflow evaluation.

    Args:
        jp_text: Japanese reading text

    Returns:
        List of question dicts (each with category, question, options, answer)
    """
    # Initialize on first call
    if not hasattr(predict_fn, '_initialized'):
        agent = Agent(model="gpt-5-mini", temperature=1.0)

        if USE_DSPY:
            lm = agent.get_dspy_agent()
            dspy.configure(lm=lm)
            predict_fn._generator = QuestionGenerator()
        else:
            predict_fn._chain = agent.get_agent(schema=QuestionSet)

        predict_fn._initialized = True

    # Use cached backend
    if USE_DSPY:
        return predict_fn._generator(jp_text)
    else:
        result = predict_fn._chain.invoke({"jp_text": jp_text})
        return result.model_dump()["questions"]
