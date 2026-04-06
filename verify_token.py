import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

print(f"Testing connectivity to {API_BASE_URL} with model {MODEL_NAME}...")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print("SUCCESS: Token is valid and API responds.")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"FAILURE: {e}")
