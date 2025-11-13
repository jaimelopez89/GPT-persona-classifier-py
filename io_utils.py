"""Utility functions for I/O operations, file handling, and data processing.

This module provides functions for:
- Loading environment variables and configuration
- Reading and writing CSV files
- Filtering and processing prospect data
- Managing output directories and checkpoint files

Author: Jaime López, 2025
"""

import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from config import OUTPUT_DIR, SKIPPED_DIR, CHECKPOINTS_DIR


def load_env_or_fail() -> str:
    """Load environment variables and validate OPENAI_API_KEY is set.

    Loads environment variables from .env file and checks that OPENAI_API_KEY
    is present. Raises an error if the key is missing.

    Returns:
        The OPENAI_API_KEY value as a string.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not set in environment or .env file.
    """
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set (in env or .env).")
    return key


def ensure_dirs() -> None:
    """Create output directories if they don't exist.

    Creates OUTPUT_DIR, SKIPPED_DIR, and CHECKPOINTS_DIR with parent directories
    as needed. Does nothing if directories already exist.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SKIPPED_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def now_stamp(fmt: str = "%Y-%m-%d %H %M %S") -> str:
    """Generate a timestamp string using the current date/time.

    Args:
        fmt: Format string for datetime.strftime (default: "%Y-%m-%d %H %M %S").

    Returns:
        Formatted timestamp string.
    """
    return datetime.now().strftime(fmt)


def read_text(path: str) -> str:
    """Read a text file and return its contents.

    Args:
        path: Path to the text file to read.

    Returns:
        File contents as a string (UTF-8 encoded).
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def filter_emails(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Filter out rows containing test emails or Ververica emails.

    Args:
        df: DataFrame to filter.
        col: Name of the column containing email addresses.

    Returns:
        Filtered DataFrame with test/Ververica emails removed.
    """
    s = df[col].fillna("").astype(str)
    # Removed filter that excluded test emails because it caught too many legitimate emails e.g. statestreet, testa, smartest energy, etc.
    return df[~s.str.contains(r"@ververica", regex=True, na=False)]

def load_input_csv(path: str) -> pd.DataFrame:
    """Load a CSV file and normalize column names and types.

    Handles column name normalization (Record ID -> Prospect Id) and ensures
    common columns are treated as strings. Prospect IDs are read as strings
    to prevent Excel's scientific notation issues.

    Args:
        path: Path to the CSV file to load.

    Returns:
        DataFrame with normalized columns and string types for key fields.
        Prospect IDs are preserved as strings to prevent scientific notation conversion.
        Empty values are converted from "nan" strings to empty strings.
    """
    # Read CSV with Prospect Id as string to prevent scientific notation
    # Use dtype=str for all columns to preserve exact values
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    
    # Normalize column name if present
    if "Record ID" in df.columns and "Prospect Id" not in df.columns:
        df = df.rename(columns={"Record ID": "Prospect Id"})
    
    # Ensure key columns are strings and normalize empty values
    for c in ["Prospect Id", "Job Title", "First Name", "Last Name", "Email", "Company"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
            # Replace 'nan' strings (from dtype=str conversion) with empty strings
            df[c] = df[c].replace("nan", "").replace("NaN", "").replace("None", "")
    
    return df


def save_outputs(final_df: pd.DataFrame, skipped_df: pd.DataFrame) -> tuple[str, str]:
    """Save accepted and skipped prospect DataFrames to CSV files.

    Creates timestamped filenames and saves to OUTPUT_DIR and SKIPPED_DIR.
    Ensures output directories exist before saving. Prospect IDs are saved
    with quoting to prevent Excel from converting them to scientific notation.

    Args:
        final_df: DataFrame containing accepted prospects with valid personas.
        skipped_df: DataFrame containing skipped prospects with skip reasons.

    Returns:
        Tuple of (accepted_file_path, skipped_file_path) as strings.

    Note:
        Prospect IDs are saved with QUOTE_ALL to ensure they're treated as text,
        preventing Excel from auto-converting large numbers to scientific notation.
    """
    ensure_dirs()
    stamp = now_stamp()
    accepted_path = OUTPUT_DIR / f"Personas {stamp}.csv"
    skipped_path = SKIPPED_DIR / f"Skipped prospects {stamp}.csv"
    
    # Ensure Prospect Id is string before saving
    if "Prospect Id" in final_df.columns:
        final_df = final_df.copy()
        final_df["Prospect Id"] = final_df["Prospect Id"].astype(str)
    if "Prospect Id" in skipped_df.columns:
        skipped_df = skipped_df.copy()
        skipped_df["Prospect Id"] = skipped_df["Prospect Id"].astype(str)
    
    # Save with quoting to ensure Prospect IDs are treated as text
    # This prevents Excel from auto-converting to scientific notation
    final_df.to_csv(accepted_path, index=False, quoting=csv.QUOTE_ALL)
    skipped_df.to_csv(skipped_path, index=False, quoting=csv.QUOTE_ALL)
    
    return str(accepted_path), str(skipped_path)


def save_checkpoint_raw(name: str, content: str | dict) -> str:
    """Save checkpoint data (raw JSON or JSONL) to checkpoint directory.

    Saves intermediate results for recovery/resumption. Automatically determines
    file format based on content type (dict -> JSON, str -> JSONL).

    Args:
        name: Base name for the checkpoint file (timestamp will be appended).
        content: Either a dictionary (saved as JSON) or string (saved as JSONL).

    Returns:
        Path to the saved checkpoint file as a string.
    """
    ensure_dirs()
    if isinstance(content, dict):
        path = CHECKPOINTS_DIR / f"{name}_{now_stamp('%Y-%m-%d_%H-%M-%S')}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)
    else:
        path = CHECKPOINTS_DIR / f"{name}_{now_stamp('%Y-%m-%d_%H-%M-%S')}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    return str(path)