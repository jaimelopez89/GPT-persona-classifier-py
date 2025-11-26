"""Parsing utilities for LLM responses and batch outputs.

This module provides functions for:
- Parsing CSV-like output from streaming LLM responses
- Parsing JSONL output from batch API responses
- Sanitizing job titles
- Determining skip reasons for invalid responses
- Fuzzy matching invalid personas to valid ones

Author: Jaime López, 2025
"""

import io
import re
import json
import pandas as pd
from difflib import get_close_matches
from config import VALID_PERSONAS


def sanitize_job_title(title: str) -> str:
    """Remove commas from job titles to preserve CSV structure.

    Commas in job titles can break CSV parsing, so they are replaced with spaces.

    Args:
        title: Job title string (may be None).

    Returns:
        Job title with commas replaced by spaces, or empty string if None.
    """
    return re.sub(",", " ", str(title or ""))


def parse_llm_csv(enriched_result: str) -> pd.DataFrame:
    """Parse CSV-like output from LLM into a DataFrame.

    Parses streaming LLM responses that are formatted as CSV with columns:
    Prospect Id, Job Title, Persona, Persona Certainty.

    Args:
        enriched_result: CSV-formatted string from LLM (may contain multiple rows).

    Returns:
        DataFrame with columns: Prospect Id, Job Title, Persona, Persona Certainty.
        Returns empty DataFrame with these columns if input is empty.
    """
    if not enriched_result.strip():
        return pd.DataFrame(columns=["Prospect Id", "Job Title", "Persona", "Persona Certainty"])
    # Probe to check for extra columns
    _probe = pd.read_csv(io.StringIO(enriched_result), header=None, on_bad_lines="warn")
    if _probe.shape[1] > 4:
        print("Warning: extra columns present; using first four.")
    # Parse with explicit column names and types
    df = pd.read_csv(
        io.StringIO(enriched_result), header=None,
        names=["Prospect Id", "Job Title", "Persona", "Persona Certainty"],
        usecols=[0, 1, 2, 3], dtype=str, on_bad_lines="warn"
    )
    return df


def determine_skip_reason(row: pd.Series) -> str | None:
    """Determine why a prospect should be skipped (if any).

    Checks if the persona is missing or invalid. Returns None if the prospect
    should be accepted (has a valid persona).

    Args:
        row: DataFrame row containing at least a "Persona" column.

    Returns:
        Skip reason string if prospect should be skipped, None otherwise.
    """
    persona = str(row.get("Persona", "") or "").strip()
    if not persona:
        return "No LLM response"
    if persona not in VALID_PERSONAS:
        return f"Invalid persona: {persona}"
    return None


def parse_batch_output_jsonl(jsonl_str: str) -> tuple[dict[str, str], dict[str, str]]:
    """Parse JSONL output from OpenAI Batch API.

    Parses the JSONL response file from a batch job, extracting successful
    responses and error information.

    Args:
        jsonl_str: JSONL string from batch API output file.

    Returns:
        Tuple of (results_dict, errors_dict):
        - results_dict: Maps custom_id to response content for successful requests.
        - errors_dict: Maps custom_id to error message for failed requests.
    """
    out, errors = {}, {}
    for line in jsonl_str.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cid = str(obj.get("custom_id", ""))
        resp = obj.get("response", {})
        status = int(resp.get("status_code", 0) or 0)
        if status == 200:
            try:
                # Extract content from successful response
                content = resp["body"]["choices"][0]["message"]["content"]
                out[cid] = content
            except (KeyError, TypeError, IndexError) as e:
                errors[cid] = f"Malformed success body: {e}"
        else:
            # Extract error message from failed response
            body = resp.get("body") or {}
            msg = None
            try:
                msg = body.get("error", {}).get("message")
            except (AttributeError, TypeError):
                msg = None
            errors[cid] = f"HTTP {status}: {msg or body}"
    return out, errors


def fuzzy_match_invalid_personas(
    skipped_df: pd.DataFrame, threshold: int = 60
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attempt to fuzzy-match invalid personas to valid ones.

    Takes skipped prospects with "Invalid persona" reason and tries to match
    the invalid persona string to one of the allowed personas using fuzzy
    string matching. This fixes cases where the LLM outputs a persona that
    is similar to a valid one but with minor typos or variations.

    Args:
        skipped_df: DataFrame of skipped prospects, including "Persona" and
            "Skip Reason" columns.
        threshold: Minimum similarity score (0-100) for a match to be
            considered valid. Default is 60.

    Returns:
        A tuple of (corrected_df, still_skipped_df):
        - corrected_df: DataFrame with personas corrected and "Skip Reason"
            cleared. These prospects can now be accepted.
        - still_skipped_df: DataFrame of prospects that could not be corrected.
    """
    if skipped_df.empty:
        return pd.DataFrame(), skipped_df.copy()

    # Filter for prospects with "Invalid persona" skip reason
    invalid_mask = skipped_df["Skip Reason"].astype(str).str.startswith(
        "Invalid persona:"
    )
    invalid_df = skipped_df[invalid_mask].copy()
    other_skipped = skipped_df[~invalid_mask].copy()

    if invalid_df.empty:
        return pd.DataFrame(), skipped_df.copy()

    corrected_rows = []
    still_invalid_rows = []

    for _, row in invalid_df.iterrows():
        invalid_persona = str(row.get("Persona", "")).strip()
        if not invalid_persona:
            still_invalid_rows.append(row)
            continue

        # Try to find a close match
        matches = get_close_matches(
            invalid_persona,
            list(VALID_PERSONAS),
            n=1,
            cutoff=threshold / 100.0
        )

        if matches:
            # Found a match - correct the persona and clear skip reason
            corrected_row = row.copy()
            corrected_row["Persona"] = matches[0]
            corrected_row["Skip Reason"] = None
            corrected_rows.append(corrected_row)
        else:
            # No match found - keep as skipped
            still_invalid_rows.append(row)

    corrected_df = pd.DataFrame(corrected_rows) if corrected_rows else pd.DataFrame()
    still_invalid_df = pd.DataFrame(still_invalid_rows) if still_invalid_rows else pd.DataFrame()

    # Combine still_invalid with other skipped prospects
    if not still_invalid_df.empty:
        still_skipped_df = pd.concat(
            [other_skipped, still_invalid_df], ignore_index=True
        )
    else:
        still_skipped_df = other_skipped.copy()

    return corrected_df, still_skipped_df