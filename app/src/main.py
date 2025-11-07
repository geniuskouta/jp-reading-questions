from jp_reading_questions.evaluation import evaluation_dataset
from jp_reading_questions.prediction import predict_fn, USE_DSPY
from jp_reading_questions.score import (
    json_format_correct,
    has_all_categories,
    options_are_unique,
    answer_is_valid,
    has_sufficient_questions,
    ENABLE_LLM_SCORERS
)
import os
import mlflow
import json
from datetime import datetime


# Import backend-specific metadata
if USE_DSPY:
    llm_model = "gpt-5-mini"
    llm_temperature = 1.0
    SYSTEM_PROMPT = "DSPy-based generation (prompts handled internally)"
    USER_PROMPT = "DSPy-based generation (prompts handled internally)"
else:
    from jp_reading_questions.prediction import chain
    from jp_reading_questions.prompts.prompt_loader import load_prompt
    llm_model = "gpt-5-mini"
    llm_temperature = 1.0
    SYSTEM_PROMPT = load_prompt("system.md")
    USER_PROMPT = load_prompt("user.md")

# Set the tracking URI to point to your MLflow server
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
# Create a new MLflow experiment for this evaluation
mlflow.set_experiment("jp_reading_questions_evaluation")

# Build scorer list
scorers = [
    json_format_correct,
    has_all_categories,
    options_are_unique,
    answer_is_valid,
    has_sufficient_questions,
]

# Add LLM-based scorers if enabled
if ENABLE_LLM_SCORERS:
    from jp_reading_questions.score import (
        question_text_relevance,
        option_quality,
        answer_correctness_check
    )
    scorers.extend([question_text_relevance, option_quality, answer_correctness_check])
    print("LLM-based scorers enabled (question_text_relevance, option_quality, answer_correctness_check)")
else:
    print("LLM-based scorers disabled. Set ENABLE_LLM_SCORERS=true to enable.")

scorer_names = ", ".join([getattr(s, '__name__', str(s)) for s in scorers])

# Start an MLflow run to track this evaluation
with mlflow.start_run(run_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    # Log parameters
    mlflow.log_param("backend", "dspy" if USE_DSPY else "langchain")
    mlflow.log_param("model_name", llm_model)
    mlflow.log_param("temperature", llm_temperature)
    mlflow.log_param("num_eval_samples", len(evaluation_dataset))
    mlflow.log_param("enable_llm_scorers", ENABLE_LLM_SCORERS)
    mlflow.log_param("scorers", scorer_names)

    # Log prompts as text directly
    mlflow.log_text(SYSTEM_PROMPT, artifact_file="prompts/system_prompt.md")
    mlflow.log_text(USER_PROMPT, artifact_file="prompts/user_prompt.md")

    # Run the evaluation
    results = mlflow.genai.evaluate(
        data=evaluation_dataset,
        predict_fn=predict_fn,
        scorers=scorers,
    )

    # Log evaluation results summary as JSON
    results_summary = {
        "metrics": results.metrics,
        "num_samples": len(evaluation_dataset),
        "timestamp": datetime.now().isoformat()
    }
    mlflow.log_dict(results_summary, artifact_file="results/evaluation_results.json")

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Backend: {'DSPy' if USE_DSPY else 'LangChain'}")
    print(f"Model: {llm_model}")
    print(f"Temperature: {llm_temperature}")
    print(f"Samples evaluated: {len(evaluation_dataset)}")
    print(f"LLM Scorers enabled: {ENABLE_LLM_SCORERS}")
    print("\nMetrics:")
    for metric_name, metric_value in results.metrics.items():
        print(f"  {metric_name}: {metric_value}")
    print("="*50 + "\n")