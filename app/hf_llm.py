import requests
import os
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}"
}


def classify_intent(user_input: str) -> str:
    prompt = f"""
Classify the intent of the following user message into one of the categories:
- ask_preferences
- remember_preference
- reference_drink
- request_recommendation
- chat

User message: "{user_input}"

Intent:"""
    print(prompt)
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 10,
            "temperature": 0.0,
        },
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()

    result = response.json()
    text = result[0]["generated_text"]

    # витягуємо лише після "Intent:"
    return text.strip().split("Intent:")[-1].strip().split()[0]


def generate_answer(context: str, question: str) -> str:
    print("generate")
    prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""
    print(f"generate {prompt}")
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 150}
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()

    result = response.json()
    return result[0]["generated_text"].split("Answer:")[-1].strip()
