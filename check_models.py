import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Fetch the list of all available models
models = client.models.list()

print("--- AVAILABLE GROQ MODELS ---")
for model in models.data:
    print(f"ID: {model.id}")