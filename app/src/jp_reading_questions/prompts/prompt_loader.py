from pathlib import Path

CURRENT_DIR = Path(__file__).parent
PROMPTS_DIR = CURRENT_DIR

def load_prompt(filename: str) -> str:
    """Load a prompt from a markdown file."""
    prompt_path = PROMPTS_DIR / filename
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()
