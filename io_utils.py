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

def load_env_or_fail():
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set (in env or .env).")
    return key

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SKIPPED_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

def now_stamp(fmt="%Y-%m-%d %H %M %S"):
    return datetime.now().strftime(fmt)

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def filter_emails(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = df[col].fillna("").astype(str)
    # Removed filter that excluded test emails because it caught too many legitimate emails e.g. statestreet, testa, smartest energy, etc.
    return df[~s.str.contains(r"@ververica", regex=True, na=False)]

def load_input_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    if "Record ID" in df.columns and "Prospect Id" not in df.columns:
        df = df.rename(columns={"Record ID": "Prospect Id"})
    for c in ["Prospect Id", "Job Title", "First Name", "Last Name", "Email", "Company"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

def save_outputs(final_df: pd.DataFrame, skipped_df: pd.DataFrame):
    ensure_dirs()
    stamp = now_stamp()
    accepted_path = OUTPUT_DIR / f"Personas {stamp}.csv"
    skipped_path = SKIPPED_DIR / f"Skipped prospects {stamp}.csv"
    final_df.to_csv(accepted_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)
    return str(accepted_path), str(skipped_path)

def save_checkpoint_raw(name: str, content: str | dict):
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