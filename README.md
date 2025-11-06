# jp-reading-questions

This is an app which generates comprehension questions for Japanese reading text such as news articles, novels, and interviews. (at least 1 min length reading is expected as text input)

The question output comes with:
- question statement
- 4 options with A, B, C, and D
- a single answer
- category chosen from 事実 (facts), メインポイントや暗示されたメッセージ (main points/implied messages), and 文法や表現 (grammar and expressions)

## Architecture

- **LLM**: GPT-4o-mini with structured output (Pydantic schema validation)
- **Framework**: LangChain for prompt management and LLM interaction
- **Evaluation**: MLflow for experiment tracking and scoring
- **Infrastructure**: Docker Compose (MLflow server + App container)

## Approach

- Prepare evaluation set data with Japanese text and comprehension questions
- Use DSPy to define expected input and output with the evaluation set data
- Use DSPy to brush up on the prompt by adding more evaluation set data

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

## Project Structure

```
jp-reading-questions/
├── app/
│   ├── src/
│   │   ├── main.py                    # Evaluation runner
│   │   └── jp_reading_questions/
│   │       ├── prediction.py          # LLM prediction logic
│   │       ├── evaluation.py          # Test dataset (4 samples)
│   │       ├── score.py               # MLflow scorers
│   │       └── prompts/
│   │           ├── system.md          # System prompt (JLPT expert)
│   │           ├── user.md            # User prompt template
│   │           └── scorers/           # Scorer prompts (LLM-as-judge)
│   │               ├── question_relevance.md
│   │               ├── option_quality.md
│   │               └── answer_correctness.md
│   ├── Dockerfile
│   └── pyproject.toml                 # Dependencies
├── mlflow_server/
│   ├── Dockerfile
│   ├── data/                          # SQLite database
│   └── mlruns/                        # Experiment artifacts
├── docker-compose.yml
└── README.md
```

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

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key for GPT models |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URL |
| `ENABLE_LLM_SCORERS` | `false` | Enable semantic quality scorers (costs money) |
| `MLFLOW_PORT` | `5001` | Port for MLflow UI |

## Future Improvements

- [ ] Integrate DSPy for automatic prompt optimization
- [ ] Expand evaluation dataset to 50+ samples
- [ ] Add scorer for question difficulty level
- [ ] Implement model registry for version tracking
- [ ] Add dashboard for metric visualization over time