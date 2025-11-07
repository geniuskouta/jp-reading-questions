# jp-reading-questions

This is an app which generates comprehension questions for Japanese reading text such as news articles, novels, and interviews. (at least 1 min length reading is expected as text input)

The question output comes with:
- question statement
- 4 options with A, B, C, and D
- a single answer
- category chosen from 事実 (facts), メインポイントや暗示されたメッセージ (main points/implied messages), and 文法や表現 (grammar and expressions)

## Architecture

- **LLM**: OpenAI models with structured output (Pydantic schema validation)
- **Framework**: Unified Agent supporting both LangChain and DSPy backends
  - LangChain: Manual prompt management with markdown files
  - DSPy: Automatic prompt optimization with machine learning
- **Evaluation**: MLflow for experiment tracking and scoring
- **Infrastructure**: Docker Compose (MLflow server + App container)

## Approach

This project supports two approaches for generating questions:

### 1. Manual Prompts (LangChain)
- Manually craft prompts in markdown files
- Requires prompt engineering expertise
- Good for understanding and control
- Run with: `docker-compose exec app poetry run python src/main.py`

### 2. Automatic Prompts (DSPy)
- Automatically optimizes prompts using evaluation data
- Learns from examples to improve quality
- Reduces manual prompt maintenance
- Three-step workflow:
  1. **Setup** (`dspy_setup.py`) - Verify configuration
  2. **Optimize** (`dspy_optimize.py`) - Generate optimized prompts (expensive, run once)
  3. **Evaluate** (`dspy_evaluate.py`) - Use cached prompts (cheap, run many times)

## Evaluation Cycle

The evaluation pipeline uses MLflow to track and score LLM output with two types of scorers:

### Structural Scorers (Always Enabled, Free)

These scorers validate the format and structure of generated questions:

1. **json_format_correct** - Validates JSON structure and required fields
2. **has_all_categories** - Ensures all 3 question categories are present
3. **options_are_unique** - Checks for duplicate answer options
4. **answer_is_valid** - Validates answer is A/B/C/D and references valid option
5. **has_sufficient_questions** - Ensures at least 3 questions are generated

### Semantic Scorers (Optional, Uses GPT-4o-mini)

These scorers use LLM-as-judge to evaluate content quality. **Enable by setting `ENABLE_LLM_SCORERS=true`**:

1. **question_text_relevance** - Validates questions relate to the input text
2. **answer_correctness_check** - Verifies the correct answer is actually correct
3. **option_quality** - Evaluates if options are plausible and well-formed

Prompts for semantic scorers are managed in markdown files at `app/src/jp_reading_questions/prompts/scorers/`.

## Setup

### Prerequisites

- Docker and Docker Compose
- OpenAI API key

### Installation

1. Clone the repository and navigate to the project directory

2. Create `app/.env` file with your OpenAI API key:
```bash
OPENAI_API_KEY=your-api-key-here
# Optional: Enable LLM-based semantic scorers (costs money)
# ENABLE_LLM_SCORERS=true
```

3. Start the services:
```bash
docker-compose up -d
```

This will start:
- **MLflow server** at http://localhost:5001
- **App container** (stays running in background)

### Running Evaluation

Run evaluation inside the app container:

```bash
# Without LLM-based scorers (free, structural validation only)
docker-compose exec app poetry run python src/main.py

# With LLM-based scorers (costs money, includes semantic quality checks)
docker-compose exec app env ENABLE_LLM_SCORERS=true poetry run python src/main.py
```

### Viewing Results

1. Open MLflow UI: http://localhost:5001
2. Navigate to the "jp_reading_questions_evaluation" experiment
3. View metrics, parameters, and artifacts for each run

## DSPy Workflow (Automatic Prompt Optimization)

DSPy automatically generates and optimizes prompts based on your evaluation data, eliminating manual prompt engineering.

### Step 1: Setup (First Time Only)

Verify DSPy configuration and test the connection:

```bash
docker-compose exec app poetry run python src/dspy_setup.py
```

This checks:
- DSPy installation
- OpenAI API key
- Model connectivity

### Step 2: Optimize (Run When Data Changes)

Generate optimized prompts using your evaluation dataset:

```bash
# Without LLM-based scorers (structural validation only)
docker-compose exec app poetry run python src/dspy_optimize.py

# With LLM-based scorers (semantic quality checks - costs money)
docker-compose exec app env ENABLE_LLM_SCORERS=true poetry run python src/dspy_optimize.py
```

**IMPORTANT**: This is expensive (uses GPT API calls)! Only run when:
- You add new evaluation data
- You want to improve prompt quality
- First time setup

The optimized prompts are saved to `optimized_generator_*.json` and cached for reuse.

### Step 3: Evaluate (Run Anytime)

Use the cached optimized prompts to run evaluations:

```bash
# Without LLM-based scorers (free)
docker-compose exec app poetry run python src/dspy_evaluate.py

# With LLM-based scorers (costs money)
docker-compose exec app env ENABLE_LLM_SCORERS=true poetry run python src/dspy_evaluate.py
```

This is cheap because it uses the cached prompts from Step 2. Run as many times as you want.

### DSPy vs Manual Prompts

| Aspect | Manual (LangChain) | Automatic (DSPy) |
|--------|-------------------|------------------|
| **Prompt Creation** | Hand-written in markdown | Auto-generated from examples |
| **Maintenance** | Manual editing required | Automatically improves with data |
| **Expertise Needed** | Prompt engineering skills | Data collection and evaluation |
| **Cost** | Only inference | Optimization + inference |
| **Control** | Full control over wording | Control via examples and metrics |
| **Iteration Speed** | Slow (manual changes) | Fast (add data, re-optimize) |

**Recommendation**: Use Manual approach for understanding and initial setup. Switch to DSPy once you have 10+ quality evaluation examples.


## Evaluation Dataset

The evaluation dataset is defined in `app/src/jp_reading_questions/evaluation.py` and currently contains 4 test cases:

1. Urban agriculture in Tokyo (事実 question)
2. AI in healthcare (事実 question)
3. Remote work challenges (メインポイント question)
4. Plastic pollution in Japan (暗示されたメッセージ question)

Each test case includes:
- Input: Long-form Japanese text (news article style, ~500+ characters)
- Expected output: Sample questions with category, question text, options, and correct answer

## Adding More Test Data

To expand the evaluation dataset, edit `app/src/jp_reading_questions/evaluation.py`:

```python
evaluation_dataset.append({
    "inputs": {"jp_text": "Your Japanese text here..."},
    "expectations": {
        "expected_questions": [
            {
                "category": "事実",
                "question": "Sample question?",
                "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
                "answer": "A"
            }
        ]
    }
})
```

## Configuration

See `app/.env.example` for environment variables.

## Future Improvements

- [x] Integrate DSPy for automatic prompt optimization
- [x] Add LLM-as-judge scorers for semantic quality
- [ ] Expand evaluation dataset to 50+ samples
- [ ] Add scorer for question difficulty level
- [ ] Implement model registry for version tracking
- [ ] Add dashboard for metric visualization over time
- [ ] Compare multiple DSPy optimizers (BootstrapFewShot vs MIPROv2)