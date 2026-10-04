from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os

load_dotenv()

token = os.getenv("HF_TOKEN")

if not token:
    raise RuntimeError("HF_TOKEN is missing")

client = InferenceClient(
    api_key=token
)

response = client.chat_completion(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant for Tinayu, a personal color analysis app."
        },
        {
            "role": "user",
            "content": "In one short sentence, explain what Warm Spring means in personal color analysis."
        }
    ],
    max_tokens=100,
)

print("\n--- LLAMA RESPONSE ---")
print(response.choices[0].message.content)
print("----------------------")
