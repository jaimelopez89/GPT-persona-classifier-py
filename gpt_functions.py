import os
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
def ask_chat_session(session, user_message):
    """
    Appends the user message to the session, calls the API,
    appends the assistant response to the session, and returns it.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not found.")
        return None

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
            json=data
        )
        if response.status_code == 200:
            response_text = response.json()['choices'][0]['message']['content'].strip()
            # Append assistant's response so that the session 'remembers' it if needed
            session["messages"].append({"role": "assistant", "content": response_text})
            return response_text
        else:
            print(f"Error: Received response code {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error while calling OpenAI API: {e}")
        return None