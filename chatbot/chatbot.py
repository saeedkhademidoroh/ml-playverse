import openai
from config import OPENAI_API_KEY, MODEL_NAME

openai.api_key = OPENAI_API_KEY

def ask_openai(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message["content"].strip()

# Print confirmation message
print("\n✅ chatbot.py successfully executed")
