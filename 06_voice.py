"""
Assignment 6: Voice Assistant with Vapi

TODO: 
1. Implement your example
"""


import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Your Vapi API Authorization token (API private key)
auth_token = os.getenv("VAPI_API_KEY")
# The Phone Number ID, and the Customer details for the call
phone_number_id = os.getenv("VAPI_PHONE_NUMBER_ID")
customer_number = os.getenv("VAPI_CUSTOMER_NUMBER")

# Create the header with Authorization token
headers = {
    'Authorization': f'Bearer {auth_token}',
    'Content-Type': 'application/json',
}
# Create the data payload for the API request
data = {
    'assistant': {
        "firstMessage": "Hey, what's up? Are you trying to learn AI agents today? Ask me anything!",
        "model": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an assistant."
                }
            ]
        },
        "voice": "jennifer-playht"
    },
    'phoneNumberId': phone_number_id,
    'customer': {
        'number': customer_number,
    },
}

# Make the POST request to Vapi to create the phone call
response = requests.post(
    'https://api.vapi.ai/call/phone', headers=headers, json=data)

# Check if the request was successful and print the response
if response.status_code == 201:
    print('Call created successfully')
    print(response.json())
else:
    print('Failed to create call')
    print(response.text)