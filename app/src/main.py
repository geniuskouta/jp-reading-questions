from openai import OpenAI
import mlflow
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Set MLflow tracking URI
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
mlflow.set_experiment("traces-quickstart")

# Enable MLflow tracing for OpenAI
mlflow.openai.autolog()

# Example of loading and using the prompt
prompt = mlflow.genai.load_prompt("prompts:/jp-reading-questions/1")

# Wrap the LLM call in MLflow tracing
with mlflow.start_run():
    response = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": prompt.format(),
        }],
        model="gpt-4o-mini",
    )

    # Log the response for visibility
    mlflow.log_param("model", "gpt-4o-mini")
    mlflow.log_param("prompt_name", "jp-reading-questions")
    mlflow.log_param("prompt_version", "1")

    print(response.choices[0].message.content)