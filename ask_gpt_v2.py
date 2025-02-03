import requests
import json
import io
from dotenv import load_dotenv
import os

load_dotenv()

def ask_gpt_v2(system_message=None, user_message=None, model="gpt-4o-mini"):
    """
    Calls the OpenAI Chat Completion API, allowing separate system and user messages
    as well as a customizable model.

    :param system_message: (str) Content for the system role, which sets high-level instructions or context.
    :param user_message:   (str) The actual user prompt or data payload you want GPT to act upon.
    :param model:          (str) The name of the OpenAI model (e.g., "gpt-3.5-turbo-16k", "gpt-4").
    :return:               (str) The response from GPT, or None if there's an error.
    """
    api_key = os.getenv("OPENAI_API_KEY")  # Make sure this is set in your environment
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
            print(response.text)  # helpful to see any error details from OpenAI
            return None
    except Exception as e:
        print(f"Error while calling OpenAI API: {e}")
        return None