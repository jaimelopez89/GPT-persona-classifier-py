"""Module with utilities to call LLMs, used by the main module jobtitle_persona_enrichment.py
   Jaime López, 2025
"""

import os
# import json
import requests
from dotenv import load_dotenv

load_dotenv()

# 1. Creates a session object with system instructions
def create_chat_session(system_message, model="gpt-4o-mini"):
    """
    Initialize a chat session with a single system message.
    Returns a session dict with 'model' and 'messages'.
    """
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message}
        ]
    }

# 2. Sends new user messages within that session
def ask_chat_session(session, user_message, timeout=120):
    """
    Appends the user message to the session, calls the API once, and returns
    the assistant text. This function intentionally raises on errors (429,
    timeouts, non-200s) so the caller can implement retries and backoff.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not found.")

    # Add user's message
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
            except Exception:
                retry_after_sec = None

        try:
            err_json = response.json()
            err_msg = err_json.get("error", {}).get("message")
        except Exception:
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
        # Surface timeouts to caller for retry
        raise TimeoutError(f"request timeout: {te}")

    except requests.exceptions.RequestException as re:
        # Other transport-level issues
        raise re
