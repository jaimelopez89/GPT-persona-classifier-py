"""Simple utility module for calling OpenAI Chat API.

This module provides a basic function to call OpenAI's chat completion API
with a simple prompt. Used by legacy scripts.

Author: Jaime López, 2025
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()


def ask_chatgpt(prompt: str) -> str | None:
    """Call OpenAI Chat API with a simple user prompt.

    Makes a single API call to OpenAI's chat completion endpoint with the
    provided prompt. Uses gpt-3.5-turbo model by default.

    Args:
        prompt: The user's prompt/message to send to the API.

    Returns:
        The assistant's response text as a string, or None if an error occurs.

    Note:
        This function uses the OPENAI_API_KEY environment variable for authentication.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not found.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "gpt-3.5-turbo",
        # Alternative models (commented out):
        # "model": "gpt-4o",
        # "model": "gpt-o1-mini"
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(
        url="https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=120
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        # Handle errors or unexpected response status codes
        print(f"Error: Received response code {response.status_code}")
        return None
