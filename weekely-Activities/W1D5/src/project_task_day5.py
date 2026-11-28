import os
import requests
import pandas as pd

from pathlib import Path

# Locate the W1D5 project directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Dataset location
DATA_PATH = PROJECT_DIR / "data" / "states.csv"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Prepare dataset information
def create_dataset_context(data):
    context = f"""
You are a helpful local data analysis assistant.

The user has loaded a dataset with:
- Rows: {data.shape[0]}
- Columns: {data.shape[1]}
- Column names: {list(data.columns)}

Dataset preview:
{data.head(5).to_string(index=False)}

Numerical summary:
{data.describe(include="all").to_string()}

Missing values:
{data.isnull().sum().to_string()}

Answer questions using the dataset information above.
Do not invent values that are not available.
If the dataset does not contain the required information, say so.
"""
    return context


SYSTEM_CONTEXT = create_dataset_context(df)


def ask_ollama(question):

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_CONTEXT
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=180
        )

        response.raise_for_status()

        result = response.json()

        return result["message"]["content"]

    except requests.exceptions.RequestException as error:
        return f"Ollama API Error: {error}"

    except (KeyError, ValueError) as error:
        return f"Invalid Ollama response: {error}"

    
def main():

    print(f"\nDataset loaded: {DATA_PATH}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Model: {MODEL}")

    print("\nAsk questions about your dataset.")
    print("Type 'exit' to close the assistant.\n")

    question_count = 0

    while question_count < 3:
        question = input("You: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            print("Please enter a question.\n")
            continue

        print("\nAI is thinking...\n")

        answer = ask_ollama(question)

        print("Assistant:", answer)
        print("\n" + "-" * 50 + "\n")

        question_count += 1

    print("Session completed.")
    print(f"Questions answered: {question_count}")


if __name__ == "__main__":
    main()
