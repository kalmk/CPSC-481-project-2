# test file to make sure that the API is working for Gemini

from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

print("Key:", os.getenv("GEMINI_API_KEY"))

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello"
)

print(response.text)