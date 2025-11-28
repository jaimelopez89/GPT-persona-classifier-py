"""Lightweight client for OpenAI Chat API.

This module provides a minimal interface for creating chat sessions and
sending messages to the OpenAI Chat Completions API. It handles session
management, error handling, and retry logic.

Author: Jaime López, 2025
"""

import os
import re
import requests

def create_chat_session(system_message: str, model: str) -> dict:
    """Create a chat session with a single system message.

    Args:
        system_message: The system message/instructions to set the context.
        model: The OpenAI model to use.

    Returns:
        A dictionary containing 'model' and 'messages' keys. The 'messages' list
        contains a single system message entry.
    """
    return {"model": model, "messages": [{"role": "system", "content": system_message}]}


def ask_chat_session(session: dict, user_message: str, timeout: int = 120) -> str:
    """Send a user message to the chat session and get the assistant's response.

    Appends the user message to the session, calls the OpenAI API, and returns
    the assistant's text response. The assistant's response is also appended to
    the session to maintain conversation history.

    Args:
        session: A dictionary containing 'model' and 'messages' keys.
        user_message: The user's message/prompt to send to the API.
        timeout: Request timeout in seconds (default: 120).

    Returns:
        The assistant's response text as a string.

    Raises:
        RuntimeError: If OPENAI_API_KEY environment variable is not found.
        requests.HTTPError: For non-200 HTTP status codes (includes 429 rate limits).
        TimeoutError: If the request times out.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found")
    session["messages"].append({"role": "user", "content": user_message})
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {"model": session["model"], "messages": session["messages"]}
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers, json=data, timeout=timeout
        )
        if r.status_code == 200:
            txt = r.json()["choices"][0]["message"]["content"].strip()
            session["messages"].append({"role": "assistant", "content": txt})
            return txt
        # Extract retry-after hint if available
        hint = ""
        ra = r.headers.get("Retry-After")
        if ra:
            try:
                hint = f" try again in {float(ra):.0f}s"
            except (ValueError, TypeError):
                pass
        try:
            err = r.json().get("error", {}).get("message")
        except (ValueError, requests.exceptions.JSONDecodeError):
            err = r.text
        if r.status_code == 429:
            raise requests.HTTPError(f"429 rate_limit_exceeded:{hint} | {err}")
        raise requests.HTTPError(f"HTTP {r.status_code}: {err}")
    except requests.exceptions.Timeout as te:
        raise TimeoutError(f"request timeout: {te}") from te

def extract_retry_after_seconds(msg: str) -> float:
    """Extract retry-after time from an error message.

    Parses messages like "try again in 12.5s" to extract the number of seconds.

    Args:
        msg: Error message that may contain a retry-after hint.

    Returns:
        Number of seconds to wait, or 0.0 if no hint found.
    """
    m = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)s", msg)
    return float(m.group(1)) if m else 0.0