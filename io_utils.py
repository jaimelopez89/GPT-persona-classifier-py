"""Utility functions for I/O operations, file handling, and data processing.

This module provides functions for:
- Loading environment variables and configuration
- Reading and writing CSV files
- Filtering and processing prospect data
- Managing output directories and checkpoint files

Author: Jaime López, 2025
"""

import os
import csv
import json
import zipfile
import tempfile
import shutil
from pathlib import Path
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from config import OUTPUT_DIR, SKIPPED_DIR, CHECKPOINTS_DIR, HUBSPOT_REPORT_ID


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

def extract_hubspot_zip(zip_path: str) -> str:
    """Extract Hubspot zip file and return path to the contacts CSV.

    Hubspot exports come as zip files containing:
    - hubspot-export-summary (can be ignored)
    - contacts-with-job-title-but-no.csv (the file we need)

    Args:
        zip_path: Path to the Hubspot zip file.

    Returns:
        Path to the extracted contacts CSV file.

    Raises:
        ValueError: If zip file doesn't contain the expected CSV file.
        zipfile.BadZipFile: If the file is not a valid zip file.
    """
    # Create a temporary directory for extraction
    temp_dir = tempfile.mkdtemp(prefix="hubspot_extract_")

    try:
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Look for the contacts CSV file
        # Hubspot typically names it "contacts-with-job-title-but-no.csv"
        # but we'll search for any CSV that matches the pattern
        csv_patterns = [
            "contacts-with-job-title-but-no.csv",
            "contacts-with-job-title-but-no",
            "*contacts*.csv"
        ]

        found_csv = None
        for pattern in csv_patterns:
            # Try exact match first
            exact_path = os.path.join(temp_dir, pattern)
            if os.path.exists(exact_path) and exact_path.endswith('.csv'):
                found_csv = exact_path
                break

            # Try pattern matching
            import glob
            matches = glob.glob(os.path.join(temp_dir, pattern))
            csv_matches = [m for m in matches if m.endswith('.csv') and 'summary' not in m.lower()]
            if csv_matches:
                found_csv = csv_matches[0]
                break

        # If not found with patterns, search all CSV files and exclude summary
        if not found_csv:
            all_csvs = list(Path(temp_dir).glob("*.csv"))
            csvs = [str(csv) for csv in all_csvs if 'summary' not in csv.name.lower()]
            if len(csvs) == 1:
                found_csv = csvs[0]
            elif len(csvs) > 1:
                # Multiple CSVs found, prefer the one with "contacts" in the name
                contacts_csvs = [c for c in csvs if 'contact' in c.lower()]
                if contacts_csvs:
                    found_csv = contacts_csvs[0]
                else:
                    raise ValueError(
                        f"Multiple CSV files found in zip, cannot determine which to use: {csvs}"
                    )

        if not found_csv:
            raise ValueError(
                "Could not find contacts CSV file in Hubspot zip. "
                "Expected file matching 'contacts-with-job-title-but-no.csv' pattern."
            )

        # Move the CSV to a more permanent location (same directory as zip)
        zip_dir = os.path.dirname(os.path.abspath(zip_path))
        zip_basename = os.path.splitext(os.path.basename(zip_path))[0]
        extracted_csv_path = os.path.join(zip_dir, f"{zip_basename}_extracted.csv")

        # Copy the CSV to the final location
        shutil.copy2(found_csv, extracted_csv_path)

        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        return extracted_csv_path

    except Exception:
        # Clean up on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def pull_hubspot_data(report_id: str | None = None) -> pd.DataFrame:
    """Pull contact data from Hubspot report.

    Args:
        report_id: Optional Hubspot report ID. If None, uses HUBSPOT_REPORT_ID from config.

    Returns:
        DataFrame with contact data ready for processing.
    """
    try:
        from hubspot_client import pull_report_contacts
        report_id = report_id or HUBSPOT_REPORT_ID
        return pull_report_contacts(report_id=report_id)
    except ImportError as exc:
        raise RuntimeError(
            "Hubspot integration not available. Install hubspot-api-client package."
        ) from exc


def resolve_input_file(input_path: str | None = None) -> str:
    """Resolve input file path, handling CSV files, Hubspot zip files, or Hubspot API.

    If input_path is None and HUBSPOT_REPORT_ID is set, pulls data from Hubspot.
    If the input is a zip file, it will be extracted and the contacts CSV
    will be located and returned. If it's already a CSV, it's returned as-is.

    Args:
        input_path: Optional path to input file (CSV or zip). If None and
            HUBSPOT_REPORT_ID is set, pulls from Hubspot API.

    Returns:
        Path to the CSV file to process, or raises if pulling from Hubspot.

    Raises:
        ValueError: If file format is not supported or expected files are missing.
        FileNotFoundError: If the input file doesn't exist.
        RuntimeError: If Hubspot pull is requested but not available.
    """
    # Check if we should pull from Hubspot
    if input_path is None and HUBSPOT_REPORT_ID:
        print(f"Pulling data from Hubspot report {HUBSPOT_REPORT_ID}...")
        df = pull_hubspot_data()
        # Save to temporary CSV for processing
        temp_csv = os.path.join(str(OUTPUT_DIR), f"hubspot_pull_{now_stamp('%Y%m%d_%H%M%S')}.csv")
        ensure_dirs()
        df.to_csv(temp_csv, index=False)
        print(f"Saved Hubspot data to: {temp_csv}")
        return temp_csv

    if input_path is None:
        raise ValueError("No input path provided and HUBSPOT_REPORT_ID not set")

    input_path = os.path.abspath(os.path.expanduser(input_path))

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Check if it's a zip file
    if zipfile.is_zipfile(input_path):
        print(f"Detected Hubspot zip file: {input_path}")
        print("Extracting and locating contacts CSV...")
        csv_path = extract_hubspot_zip(input_path)
        print(f"Using extracted CSV: {csv_path}")
        return csv_path

    # Check if it's a CSV file
    if input_path.lower().endswith('.csv'):
        return input_path

    # Unknown file type
    raise ValueError(
        f"Unsupported file type. Expected CSV file or Hubspot zip file, "
        f"got: {os.path.basename(input_path)}"
    )


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


def save_outputs(final_df: pd.DataFrame, skipped_df: pd.DataFrame, import_to_hubspot: bool = False) -> tuple[str, str]:
    """Save accepted and skipped prospect DataFrames to CSV files.

    Creates timestamped filenames and saves to OUTPUT_DIR and SKIPPED_DIR.
    Optionally imports accepted prospects to Hubspot. Ensures output directories
    exist before saving. Prospect IDs are saved with quoting to prevent Excel
    from converting them to scientific notation.

    Args:
        final_df: DataFrame containing accepted prospects with valid personas.
        skipped_df: DataFrame containing skipped prospects with skip reasons.
        import_to_hubspot: If True, import accepted prospects to Hubspot after saving.

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

    # Import to Hubspot if requested
    if import_to_hubspot and not final_df.empty:
        try:
            from hubspot_client import import_classified_contacts
            print("\n========= Importing to Hubspot =========")
            stats = import_classified_contacts(final_df, update_existing=True)
            print("Hubspot import complete:")
            print(f"  Updated: {stats['updated']}")
            print(f"  Created: {stats['created']}")
            print(f"  Failed: {stats['failed']}")
            if stats['errors']:
                print("  Errors (first 10):")
                for err in stats['errors'][:10]:
                    print(f"    - {err}")
        except ImportError:
            print("Warning: Hubspot integration not available. Skipping import.")
        except (ValueError, RuntimeError, KeyError) as e:
            print(f"Warning: Failed to import to Hubspot: {e}")

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