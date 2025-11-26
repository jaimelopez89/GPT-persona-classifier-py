#!/usr/bin/env python3
"""Batch enrichment script using OpenAI Batch API.

This script processes prospect data in batches using the OpenAI Batch API,
which is more cost-effective for large datasets. It:
- Loads prospect data from CSV
- Filters invalid emails and empty job titles
- Creates batch requests for persona classification
- Polls for completion and processes results
- Saves accepted and skipped prospects to separate files

Can be run from command line with optional --input and --resume-batch-id flags.

Author: Jaime López, 2025
"""

import os
import sys
import json
import pandas as pd

from config import BATCH_MODEL, FRAME_FILE, PERSONAS_FILE, VALID_PERSONAS
from io_utils import (
    load_env_or_fail, load_input_csv, filter_emails, read_text, save_outputs,
    save_checkpoint_raw, resolve_input_file
)
from parsing import (
    sanitize_job_title, parse_batch_output_jsonl, fuzzy_match_invalid_personas
)
from batch_core import upload_file_for_batch, create_batch, poll_batch_until_done, download_file_content


def build_requests_jsonl(df: pd.DataFrame, system_instructions: str, model: str, temperature: float = 0.0) -> bytes:
    """Build a JSONL file for OpenAI Batch API from prospect data.

    Creates a JSONL file where each line is a batch API request for one prospect.
    Each request asks the LLM to classify a job title into a persona.

    Args:
        df: DataFrame containing prospects with "Prospect Id" and "Job Title" columns.
        system_instructions: System message with persona definitions and instructions.
        model: OpenAI model name to use.
        temperature: Sampling temperature (default: 0.0 for deterministic output).

    Returns:
        JSONL file content as bytes (UTF-8 encoded).
    """
    # Add output format instructions to system message
    sys_msg = (
        system_instructions.strip()
        + "\n\nCRITICAL OUTPUT FORMAT: Respond with a SINGLE JSON object only.\n"
        + 'Required keys: {"persona": <one of the defined personas>, "certainty": <0-100 integer or %>}.\n'
        + "Do not include extra keys, code fences, or commentary."
    )
    items = []
    for _, r in df.iterrows():
        pid = str(r["Prospect Id"])
        # Format user message with prospect ID and sanitized job title
        user = f"Prospect Id: {pid}\nJob Title: {sanitize_job_title(r['Job Title'])}\n\nReturn ONLY the JSON."
        items.append({
            "custom_id": pid,  # Use prospect ID as custom_id for tracking
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model, "messages": [{"role":"system","content":sys_msg},{"role":"user","content":user}], "temperature": temperature}
        })
    # Convert to JSONL format (one JSON object per line)
    return ("\n".join(json.dumps(x, ensure_ascii=False) for x in items)).encode("utf-8")

def main(input_file_path: str, resume_batch_id: str | None = None, print_status: bool = True):
    api_key = load_env_or_fail()
    df = load_input_csv(input_file_path)
    df = filter_emails(df, "Email")
    print(f"After email filter: {len(df)} prospects.")

    # Filter out rows with empty job titles (check for both pandas NaN and empty strings)
    df = df[df["Job Title"].notna() & (df["Job Title"].astype(str).str.strip() != "")]
    print(f"After non-empty Job Title filter: {len(df)} prospects.")

    df = df[["Prospect Id", "Email", "Job Title"]]
    if df.empty:
        print("No valid rows to process.")
        return

    # Load system instructions
    frame = read_text(FRAME_FILE)
    personas = read_text(PERSONAS_FILE)
    system = frame + personas

    # Create new batch or resume existing one
    if resume_batch_id:
        batch_id = resume_batch_id
        print(f"Resuming batch {batch_id}")
        meta = poll_batch_until_done(api_key, batch_id, echo=print_status)
    else:
        # Build and upload requests file
        req = build_requests_jsonl(df, system_instructions=system, model=BATCH_MODEL)
        input_file_id = upload_file_for_batch(api_key, req)
        print(f"Uploaded requests file id: {input_file_id}")
        batch_id = create_batch(api_key, input_file_id)
        print(f"Created batch id: {batch_id}")
        meta = poll_batch_until_done(api_key, batch_id, echo=print_status)

    # Validate batch completed successfully
    if meta.get("status") != "completed":
        save_checkpoint_raw(f"batch_{batch_id}_meta", meta)
        raise RuntimeError(f"Batch not completed: {meta.get('status')}")

    # Download and save batch results
    out_file = meta.get("output_file_id") or (meta.get("output_file_ids") or [None])[0]
    if not out_file:
        raise RuntimeError("No output file id in batch response.")

    jsonl_output = download_file_content(api_key, out_file)
    save_checkpoint_raw(f"batch_{batch_id}_output", jsonl_output)

    # Parse batch output
    result_map, errors_map = parse_batch_output_jsonl(jsonl_output)

    # Extract persona classifications from JSON responses
    rows = []
    for pid, content in result_map.items():
        try:
            obj = json.loads(content)
            rows.append({"Prospect Id": pid, "Persona": str(obj.get("persona", "")).strip(),
                         "Persona Certainty": str(obj.get("certainty","")).strip()})
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            errors_map[pid] = f"Invalid JSON: {e}: {content[:160]}..."

    # Merge results with original data
    formatted = pd.DataFrame(rows, columns=["Prospect Id", "Persona", "Persona Certainty"])
    merged = df.merge(formatted, on="Prospect Id", how="left")

    def skip_reason(row: pd.Series) -> str | None:
        """Determine skip reason for a prospect row."""
        pid = str(row.get("Prospect Id", ""))
        persona = str(row.get("Persona", "") or "").strip()
        if not persona:
            return f"Batch error: {errors_map.get(pid, 'No LLM response')}"
        if persona not in VALID_PERSONAS:
            return f"Invalid persona: {persona}"
        return None

    # Separate accepted and skipped prospects
    merged["Skip Reason"] = merged.apply(skip_reason, axis=1)
    final_df = merged[merged["Skip Reason"].isna()].copy()
    skipped_df = merged[merged["Skip Reason"].notna()].copy()
    # Clean up duplicates and validate personas
    final_df = final_df.drop_duplicates(subset=["Prospect Id"], keep="first")
    final_df = final_df[final_df["Persona"].isin(VALID_PERSONAS)]

    # Try to fuzzy-match invalid personas to valid ones
    if not skipped_df.empty:
        corrected_df, still_skipped_df = fuzzy_match_invalid_personas(skipped_df)
        if not corrected_df.empty:
            print(
                f"\nFuzzy matching corrected {len(corrected_df)} invalid personas"
            )
            # Merge corrected prospects back into final_df
            # Need to merge with original data to get all columns
            corrected_merged = df.merge(
                corrected_df[["Prospect Id", "Persona", "Persona Certainty"]],
                on="Prospect Id", how="inner"
            )
            final_df = pd.concat([final_df, corrected_merged], ignore_index=True)
            final_df = final_df.drop_duplicates(
                subset=["Prospect Id"], keep="first"
            )
            skipped_df = still_skipped_df

    # Save results
    accepted_path, skipped_path = save_outputs(final_df, skipped_df)
    print("\n========= Processing Results =========")
    print(f"{len(final_df)} prospects updated")
    print(f"{len(skipped_df)} prospects skipped")
    print(f"\nAccepted: {accepted_path}\nSkipped:  {skipped_path}\nBatch id: {batch_id}")


def _resolve_input_path(arg_path: str | None) -> str:
    """Resolve input file path from argument or user prompt, handling Hubspot zips.

    If arg_path is provided, use it. Otherwise, prompt the user in the terminal.
    Automatically handles Hubspot zip files by extracting and locating the CSV.
    Cleans quotes, expands ~, and validates existence.

    Args:
        arg_path: Optional file path from command line argument.

    Returns:
        Path to the CSV file to process (extracted from zip if needed).

    Raises:
        SystemExit: If no input provided and EOFError occurs, or file doesn't exist.
    """
    if not arg_path:
        try:
            arg_path = input("Input the absolute path of the input file with prospects and no persona: ").strip()
        except EOFError:
            print("No input received and --input not provided. Exiting.")
            sys.exit(1)

    arg_path = arg_path.strip().strip('"').strip("'")
    arg_path = os.path.expanduser(arg_path)

    if not os.path.isabs(arg_path):
        arg_path = os.path.abspath(arg_path)

    # Use resolve_input_file to handle both CSV and Hubspot zip files
    try:
        return resolve_input_file(arg_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Batch enrichment")
    # --input optional; prompt if not supplied
    ap.add_argument("--input", required=False, help="Path to prospects CSV (if omitted, you will be prompted)")
    ap.add_argument("--resume-batch-id", default=None)
    ap.add_argument("--print-status", action="store_true")
    args = ap.parse_args()

    input_path = _resolve_input_path(args.input)
    main(input_path, args.resume_batch_id, args.print_status)