import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-2-9b-it:free")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_TOKEN = OPENROUTER_API_KEY or os.getenv("HF_TOKEN")

print(f"Testing connectivity to {API_BASE_URL} with model {MODEL_NAME}...")

client = OpenAI(
    base_url=API_BASE_URL, 
    api_key=HF_TOKEN,
    default_headers={
        "HTTP-Referer": "https://localhost:7860",
        "X-Title": "ScalarXMeta Test Script"
    }
)

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
