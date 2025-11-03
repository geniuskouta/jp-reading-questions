from jp_reading_questions.evaluation import evaluation_dataset
from jp_reading_questions.prediction import predict_fn, llm, SYSTEM_PROMPT, USER_PROMPT
from jp_reading_questions.score import json_format_correct, has_all_categories, options_are_unique, answer_is_valid
import os
import mlflow
import json
from datetime import datetime


# Set the tracking URI to point to your MLflow server
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
# Create a new MLflow experiment for this evaluation
mlflow.set_experiment("jp_reading_questions_evaluation")

# Start an MLflow run to track this evaluation
with mlflow.start_run(run_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    # Log parameters
    mlflow.log_param("model_name", llm.model_name)
    mlflow.log_param("temperature", llm.temperature)
    mlflow.log_param("num_eval_samples", len(evaluation_dataset))
    mlflow.log_param("scorers", "json_format_correct, has_all_categories, options_are_unique, answer_is_valid")

    # Log prompts as text directly
    mlflow.log_text(SYSTEM_PROMPT, artifact_file="prompts/system_prompt.md")
    mlflow.log_text(USER_PROMPT, artifact_file="prompts/user_prompt.md")

    # Run the evaluation
    results = mlflow.genai.evaluate(
        data=evaluation_dataset,
        predict_fn=predict_fn,
        scorers=[json_format_correct, has_all_categories, options_are_unique, answer_is_valid],
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
    print(f"Model: {llm.model_name}")
    print(f"Temperature: {llm.temperature}")
    print(f"Samples evaluated: {len(evaluation_dataset)}")
    print("\nMetrics:")
    for metric_name, metric_value in results.metrics.items():
        print(f"  {metric_name}: {metric_value}")
    print("="*50 + "\n")