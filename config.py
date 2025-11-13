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
STREAM_MODEL = "gpt-4.1-nano"
# STREAM_MODEL = "gpt-5-nano"
BATCH_MODEL = "gpt-4.1-nano"

# ===== Streaming rate-limit strategy =====
TARGET_TPM_BUDGET = 360_000
BASE_SLEEP_SEC = 1.5
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0
MAX_BACKOFF = 30.0
MIN_CHUNK = 10
MAX_CHUNK = 250
SAFETY_TOKEN_PER_ROW = 120
MAX_PASSES = 3

# ===== Outputs =====
OUTPUT_DIR = Path("/Users/Jaime/Documents/Classified Persona Output")
SKIPPED_DIR = OUTPUT_DIR / "Skipped prospects"
CHECKPOINTS_DIR = OUTPUT_DIR / "_checkpoints"

# ===== Instructions files =====
FRAME_FILE = "frame_instructions.txt"
PERSONAS_FILE = "persona_definitions.txt"

# ===== Personas =====
VALID_PERSONAS = {
    "Executive Sponsor", "Economic Buyer", "Data Product Manager/Owner",
    "Data User", "Application Developer", "Real-time Specialist",
    "Operator/Systems Administrator", "Technical Decision Maker", "Not a target",
}