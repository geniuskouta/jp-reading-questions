"""
Question generator agent supporting both LangChain and DSPy backends.
"""
import os
from typing import Optional
import dspy
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from jp_reading_questions.models.question_model import QuestionSet
from jp_reading_questions.prompts.dspy.question_dspy import QuestionGenerator
from jp_reading_questions.prompts.prompt_loader import load_prompt


class Agent:
    """Question generator supporting LangChain and DSPy backends."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Initialize the agent.

        Args:
            model: Model name (e.g., "gpt-5-mini")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (DSPy only, default 16000)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_agent(self, api_key: Optional[str] = None):
        """Get LangChain LLM instance.

        Args:
            api_key: OpenAI API key (defaults to env var)

        Returns:
            LangChain ChatOpenAI instance
        """
        system_prompt = load_prompt("system.md")
        user_prompt = load_prompt("user.md")

        llm = ChatOpenAI(
            model=self.model or "gpt-5-mini",
            api_key=api_key or os.getenv('OPENAI_API_KEY'),
            temperature=self.temperature or 1.0
        )

        structured_llm = llm.with_structured_output(QuestionSet)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])

        return prompt_template | structured_llm

    def get_dspy_agent(self):
        """Get DSPy LM instance.

        Returns:
            DSPy LM instance
        """
        model = self.model or "gpt-5-mini"
        lm = dspy.LM(
            model=f"openai/{model}" if not model.startswith("openai/") else model,
            temperature=self.temperature or 1.0,
            max_tokens=self.max_tokens or 16000,
        )
        return lm
