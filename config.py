"""Configuration constants for the persona classification system.

This module defines:
- Model names for streaming and batch processing
- Rate limiting and retry parameters
- Output directory paths
- Instruction file names
- Valid persona values

Author: Jaime López, 2025
"""

from pathlib import Path

# ===== Models =====
# OpenAI model to use for streaming/real-time API calls
STREAM_MODEL = "gpt-4.1-nano"
# STREAM_MODEL = "gpt-5-nano"  # Alternative model option
# OpenAI model to use for batch API calls
BATCH_MODEL = "gpt-4.1-nano"

# ===== Streaming rate-limit strategy =====
# Target tokens per minute budget to avoid rate limits
TARGET_TPM_BUDGET = 360000
# Base sleep time in seconds between API calls
BASE_SLEEP_SEC = 1.5
# Maximum number of retry attempts for failed API calls
MAX_RETRIES = 5
# Initial backoff time in seconds (exponentially increases with retries)
INITIAL_BACKOFF = 2.0
# Maximum backoff time in seconds (caps exponential backoff)
MAX_BACKOFF = 30.0
# Minimum chunk size (rows per API call) - won't go below this
MIN_CHUNK = 10
# Maximum chunk size (rows per API call) - won't go above this
MAX_CHUNK = 250
# Safety multiplier for token estimation per row
SAFETY_TOKEN_PER_ROW = 120
# Maximum number of passes to retry failed prospects
MAX_PASSES = 3

# ===== Outputs =====
# Base directory for all output files
OUTPUT_DIR = Path("/Users/Jaime/Documents/Classified Persona Output")
# Directory for skipped/invalid prospects
SKIPPED_DIR = OUTPUT_DIR / "Skipped prospects"
# Directory for checkpoint files (intermediate saves)
CHECKPOINTS_DIR = OUTPUT_DIR / "_checkpoints"

# ===== Instructions files =====
# File containing frame/context instructions for the LLM
FRAME_FILE = "frame_instructions.txt"
# File containing persona definitions for classification
PERSONAS_FILE = "persona_definitions.txt"

# ===== Personas =====
# Set of valid persona classifications that can be assigned to prospects
VALID_PERSONAS = {
    "Executive Sponsor", "Economic Buyer", "Data Product Manager/Owner",
    "Data User", "Application Developer", "Real-time Specialist",
    "Operator/Systems Administrator", "Technical Decision Maker", "Not a target",
}

# ===== Hubspot Integration =====
# Optional Hubspot report ID to pull contacts from
# Set this to automatically pull data from a Hubspot report instead of using a file
# Leave as None to use file-based input
HUBSPOT_REPORT_ID = None  # Example: "12345678"