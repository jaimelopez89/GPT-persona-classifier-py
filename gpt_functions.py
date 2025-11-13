"""Module with utilities to call LLMs, used by the main module jobtitle_persona_enrichment.py.

This module provides functions for creating chat sessions and making API calls to OpenAI's
chat completion API. It handles session management and error handling for API requests.

Author: Jaime López, 2025
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()


def create_chat_session(system_message: str, model: str = "gpt-4o-mini") -> dict:
    """Create a chat session with a single system message.

    Args:
        system_message: The system message/instructions to set the context for the chat.
        model: The OpenAI model to use (default: "gpt-4o-mini").

    Returns:
        A dictionary containing 'model' and 'messages' keys. The 'messages' list
        contains a single system message entry.
    """
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message}
        ]
    }


def ask_chat_session(session: dict, user_message: str, timeout: int = 120) -> str:
    """Send a user message to the chat session and get the assistant's response.

    Appends the user message to the session, calls the OpenAI API, and returns
    the assistant's text response. The assistant's response is also appended to
    the session to maintain conversation history.

    This function intentionally raises exceptions on errors (429 rate limits,
    timeouts, non-200 status codes) so the caller can implement retries and backoff.

    Args:
        session: A dictionary containing 'model' and 'messages' keys (as returned
            by create_chat_session).
        user_message: The user's message/prompt to send to the API.
        timeout: Request timeout in seconds (default: 120).

    Returns:
        The assistant's response text as a string.

    Raises:
        RuntimeError: If OPENAI_API_KEY environment variable is not found.
        requests.HTTPError: For non-200 HTTP status codes (includes 429 rate limits).
        TimeoutError: If the request times out.
        requests.exceptions.RequestException: For other transport-level errors.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not found.")

    # Add user's message to the session
    session["messages"].append({"role": "user", "content": user_message})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": session["model"],
        "messages": session["messages"]
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=timeout,
        )

        if response.status_code == 200:
            response_text = response.json()['choices'][0]['message']['content'].strip()
            # Append assistant's response so that the session 'remembers' it if needed
            session["messages"].append({"role": "assistant", "content": response_text})
            return response_text

        # Non-200: build a helpful error message and raise
        retry_after_hdr = response.headers.get("Retry-After")
        retry_after_sec = None
        if retry_after_hdr is not None:
            try:
                retry_after_sec = float(retry_after_hdr)
            except (ValueError, TypeError):
                retry_after_sec = None

        try:
            err_json = response.json()
            err_msg = err_json.get("error", {}).get("message")
        except (ValueError, requests.exceptions.JSONDecodeError):
            err_msg = response.text

        # If the server provided a hint, include it in the exception message
        hint = ""
        if retry_after_sec is not None:
            hint = f" try again in {retry_after_sec:.0f}s"

        # Raise specific for 429 so callers can detect it from message
        if response.status_code == 429:
            raise requests.HTTPError(f"429 rate_limit_exceeded:{hint} | {err_msg}")
        else:
            raise requests.HTTPError(f"HTTP {response.status_code}: {err_msg}")

    except requests.exceptions.Timeout as te:
        # Surface timeouts to caller for retry with proper exception chaining
        raise TimeoutError(f"request timeout: {te}") from te

    except requests.exceptions.RequestException as re:
        # Other transport-level issues
        raise re
