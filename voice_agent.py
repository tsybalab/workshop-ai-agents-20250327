"""
Assignment 6: Voice Assistant with Vapi

TODO:
1. Implement your example
"""

import os

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    auth_token = os.getenv("VAPI_API_KEY")
    phone_number_id = os.getenv("VAPI_PHONE_NUMBER_ID")
    customer_number = os.getenv("VAPI_CUSTOMER_NUMBER")
    provider = os.getenv("LLM_PROVIDER")
    model = os.getenv("LLM_MODEL")

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }

    data = {
        "assistant": {
            "firstMessage": "Hi there! I'm a virtual assistant exploring how AI is changing the way we diagnose illnesses. Can I share something interesting with you?",
            "model": {
                "provider": provider,
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a curious and engaging virtual assistant calling in English. Your role is to start a light conversation about how artificial intelligence "
                            "is being used in modern medicine to support diagnostics. Begin by asking if the person has ever heard of AI helping doctors detect diseases like cancer or heart conditions. "
                            "Mention that AI can analyze scans, lab results, and even symptoms faster than ever. Share a quick example like 'AI tools can now detect early signs of lung cancer from X-rays with very high accuracy.' "
                            "Keep the conversation open and engaging—encourage the person to ask questions or share their opinion. If they’re interested, you can explain how AI is used ethically to assist—not replace—doctors."
                        ),
                    }
                ],
            },
            "voice": "echo-openai",
        },
        "phoneNumberId": phone_number_id,
        "customer": {
            "number": customer_number,
        },
    }

    print("📞 Initiating call...")
    response = requests.post(
        "https://api.vapi.ai/call/phone", headers=headers, json=data
    )

    if response.status_code == 201:
        print("✅ Call created successfully.")
    else:
        print("❌ Failed to create call")
        print(response.text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting gracefully. Bye!")
