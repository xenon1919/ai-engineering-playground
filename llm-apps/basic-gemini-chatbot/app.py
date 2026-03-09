import os
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("AI Chatbot started. Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye.")
        break

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_input
    )

    print("AI:", response.text)
    
history = []

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    history.append({"role": "user", "parts": [user_input]})

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=history
    )

    ai_reply = response.text
    print("AI:", ai_reply)

    history.append({"role": "model", "parts": [ai_reply]})