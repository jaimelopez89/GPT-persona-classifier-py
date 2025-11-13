"""Lightweight client for OpenAI Chat API.

This module provides a minimal interface for creating chat sessions and
sending messages to the OpenAI Chat Completions API. It handles session
management, error handling, and retry logic.

Author: Jaime López, 2025
"""

import os
import re
import requests

def create_chat_session(system_message: str, model: str):
    return {"model": model, "messages": [{"role": "system", "content": system_message}]}

def ask_chat_session(session: dict, user_message: str, timeout: int = 120) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found")
    session["messages"].append({"role": "user", "content": user_message})
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": session["model"], "messages": session["messages"]}
    try:
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=timeout)
        if r.status_code == 200:
            txt = r.json()["choices"][0]["message"]["content"].strip()
            session["messages"].append({"role": "assistant", "content": txt})
            return txt
        hint = ""
        ra = r.headers.get("Retry-After")
        if ra:
            try:
                hint = f" try again in {float(ra):.0f}s"
            except (ValueError, TypeError):
                pass
        try:
            err = r.json().get("error", {}).get("message")
        except Exception:
            err = r.text
        if r.status_code == 429:
            raise requests.HTTPError(f"429 rate_limit_exceeded:{hint} | {err}")
        raise requests.HTTPError(f"HTTP {r.status_code}: {err}")
    except requests.exceptions.Timeout as te:
        raise TimeoutError(f"request timeout: {te}") from te

def extract_retry_after_seconds(msg: str) -> float:
    m = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)s", msg)
    return float(m.group(1)) if m else 0.0