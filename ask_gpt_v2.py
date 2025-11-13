"""Enhanced GPT API client with system and user message support.

This module provides a more flexible interface to the OpenAI Chat API,
allowing separate system and user messages. This is useful for setting
context and instructions separately from the actual prompt.

Author: Jaime López, 2025
"""

import os
import json
import io
import requests
from dotenv import load_dotenv

load_dotenv()


def ask_gpt_v2(system_message: str | None = None, user_message: str | None = None, model: str = "gpt-4o-mini") -> str | None:
    """Call OpenAI Chat API with separate system and user messages.

    Calls the OpenAI Chat Completion API, allowing separate system and user messages
    as well as a customizable model. The system message sets high-level context,
    while the user message contains the actual prompt.

    Args:
        system_message: Content for the system role, which sets high-level
            instructions or context. Optional.
        user_message: The actual user prompt or data payload. Optional.
        model: The name of the OpenAI model (e.g., "gpt-3.5-turbo-16k", "gpt-4",
            "gpt-4o-mini"). Default: "gpt-4o-mini".

    Returns:
        The response from GPT as a string, or None if there's an error.

    Note:
        At least one of system_message or user_message should be provided.
        Uses OPENAI_API_KEY environment variable for authentication.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not found.")
        return None

    # Build the messages list dynamically
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "messages": messages
    }

    try:
        response = requests.post(
            url="https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            print(f"Error: Received response code {response.status_code}")
            print(response.text)  # Helpful to see any error details from OpenAI
            return None
    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"Error while calling OpenAI API: {e}")
        return None