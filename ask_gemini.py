"""Utility module for calling Google Gemini API.

This module provides a function to call Google's Gemini API using the
google.genai library (the new recommended SDK). This replaces the deprecated
google.generativeai library.

Author: Jaime López, 2025
"""

import os
from dotenv import load_dotenv


try:
    from google import genai
except ImportError:
    genai = None  # type: ignore[assignment]


load_dotenv()


# Initialize Gemini API client
api_key = os.getenv("GEMINI_API_KEY")
if api_key and genai is not None:
    client = genai.Client(api_key=api_key)
    # Default model name for content generation
    default_model = 'gemini-1.5-flash'
else:
    client = None
    default_model = None


def ask_gemini(prompt: str, model: str | None = None) -> str | None:
    """Call Google Gemini API with a prompt using the official SDK.

    Uses the google.genai library (new recommended SDK) to interact with Gemini models.
    This replaces the deprecated google.generativeai library.

    Args:
        prompt: The prompt/message to send to the API.
        model: Optional model name (defaults to 'gemini-1.5-flash' if not provided).

    Returns:
        The API response text as a string, or None if an error occurs.

    Note:
        This function uses the GEMINI_API_KEY environment variable for authentication.
        The client must be initialized before calling this function.
    """
    if client is None:
        print("Error: GEMINI_API_KEY environment variable not found or client not configured.")
        return None

    model_name = model or default_model
    if model_name is None:
        print("Error: No model specified and default model not configured.")
        return None

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text
    except (ValueError, AttributeError, RuntimeError, KeyError) as e:
        # Handle API errors, missing attributes, or runtime issues
        print(f"Error while calling Gemini API: {e}")
        return None