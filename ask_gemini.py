"""Utility module for calling Google Gemini API.

This module provides a function to call Google's Gemini API. Note that the
implementation appears incomplete - it uses the google.generativeai library
for configuration but then uses requests for the actual call, which may not
be the correct approach for Gemini API.

Author: Jaime López, 2025
"""

import requests
from dotenv import load_dotenv
import os
import google.generativeai as genai


load_dotenv()


# Configure Gemini API (though this may not be used by ask_gemini function)
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    # The Gemini 1.5 models are versatile and work with both text-only and multimodal prompts
    model = genai.GenerativeModel('gemini-1.5-flash')
except (KeyError, TypeError):
    # API key not set or invalid
    pass


def ask_gemini(prompt: str) -> str | None:
    """Call Google Gemini API with a prompt.

    Note: This implementation appears to be incomplete or incorrect. It uses
    the google.generativeai library for configuration but then makes a direct
    HTTP request, which may not match the actual Gemini API structure.

    Args:
        prompt: The prompt/message to send to the API.

    Returns:
        The API response text as a string, or None if an error occurs.

    Note:
        This function uses the GEMINI_API_KEY environment variable for authentication.
        The API endpoint and request format may need adjustment based on actual
        Gemini API documentation.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not found.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "gemini-1.0",  # Adjust model name as per Gemini API documentation
        "prompt": prompt
    }

    response = requests.post(
        url="https://api.gemini.com/v1/chat/completions",  # Adjust the URL as per Gemini API documentation
        headers=headers,
        json=data
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        # Handle errors or unexpected response status codes
        print(f"Error: Received response code {response.status_code}")
        return None