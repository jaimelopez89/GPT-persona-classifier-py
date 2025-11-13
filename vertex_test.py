"""Test script for Google Vertex AI Gemini integration.

This is a simple test script to verify Vertex AI configuration and
Gemini model access. It demonstrates basic usage of the Vertex AI
GenerativeModel API.

Author: Jaime López, 2025
"""

import os
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel


load_dotenv()

# Initialize Vertex AI with project ID
project_id = os.getenv("VERTEX_PROJECT_ID")
if not project_id:
    raise RuntimeError("VERTEX_PROJECT_ID environment variable not set")

vertexai.init(project=project_id, location="us-central1")

# Create a GenerativeModel instance
model = GenerativeModel(model_name="gemini-1.5-flash-001")

# Test the model with a simple prompt
response = model.generate_content(
    "What's a good name for a flower shop that specializes in selling bouquets of dried flowers?"
)

print(response.text)