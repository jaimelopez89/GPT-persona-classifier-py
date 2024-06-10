import requests
import json
from dotenv import load_dotenv
import os
import google.generativeai as genai


load_dotenv()


genai.configure(api_key=os.getenv["GEMINI_API_KEY"])
# The Gemini 1.5 models are versatile and work with both text-only and multimodal prompts
model = genai.GenerativeModel('gemini-1.5-flash')

# Function to call the Gemini API from Python
def ask_gemini(prompt):
    api_key = os.getenv("GEMINI_API_KEY")  # Make sure to set this in your .env file
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