#!/usr/bin/env python3
"""Streaming enrichment script with adaptive chunking and retry logic.

This script processes prospect data using the OpenAI Chat API in streaming mode
with adaptive chunk sizing. It:
- Dynamically adjusts chunk size based on rate limits
- Implements exponential backoff with jitter for retries
- Processes data in multiple passes for failed prospects
- Provides progress tracking and error reporting
- Saves accepted and skipped prospects to separate files

The adaptive chunking helps maximize throughput while respecting rate limits.

Author: Jaime López, 2025
"""

import time
import math
import random
import os
import sys
import traceback
import requests
import pandas as pd
from tqdm import tqdm

from config import (
    STREAM_MODEL, TARGET_TPM_BUDGET, BASE_SLEEP_SEC, MAX_RETRIES, INITIAL_BACKOFF,
    MAX_BACKOFF, MIN_CHUNK, MAX_CHUNK, SAFETY_TOKEN_PER_ROW, MAX_PASSES, VALID_PERSONAS,
    FRAME_FILE, PERSONAS_FILE
)
from io_utils import (
    load_env_or_fail, load_input_csv, filter_emails, read_text, save_outputs,
    resolve_input_file
)
from parsing import (
    sanitize_job_title, parse_llm_csv, determine_skip_reason,
    fuzzy_match_invalid_personas
)
from llm_client import create_chat_session, ask_chat_session, extract_retry_after_seconds


def estimate_tokens(n: int) -> int:
    """Estimate the number of tokens for n rows.

    Uses a safety multiplier per row to ensure we stay within token budgets.

    Args:
        n: Number of rows to estimate tokens for.

    Returns:
        Estimated token count (n * SAFETY_TOKEN_PER_ROW).
    """
    return n * SAFETY_TOKEN_PER_ROW


def call_with_retries(session: dict, payload_text: str, chunk_size: int) -> tuple[str, int]:
    """Call the LLM API with retry logic and adaptive chunk sizing.

    Implements exponential backoff with jitter and reduces chunk size on rate limits.
    Retries up to MAX_RETRIES times before giving up.

    Args:
        session: The chat session dictionary (from create_chat_session).
        payload_text: The text payload to send to the API.
        chunk_size: Current chunk size (may be reduced on rate limits).

    Returns:
        A tuple of (response_text, updated_chunk_size). The chunk_size may be
        reduced if rate limits are encountered.

    Raises:
        The last exception encountered if all retries are exhausted.
    """
    local_chunk = chunk_size
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = ask_chat_session(session=session, user_message=payload_text)
            return resp, local_chunk
        except (TimeoutError, requests.HTTPError, requests.exceptions.RequestException) as e:
            msg = str(e)
            last_err = e
            server_wait = extract_retry_after_seconds(msg)
            # Exponential backoff with jitter
            backoff = min(MAX_BACKOFF, (INITIAL_BACKOFF * (2 ** attempt)) + random.uniform(0, 1.0))
            sleep_for = max(server_wait, backoff)
            if "rate limit" in msg.lower() or "429" in msg:
                # Reduce chunk size to lower token pressure
                new_size = max(MIN_CHUNK, math.floor(local_chunk / 2))
                if new_size < local_chunk:
                    print(f"Rate limit: chunk {local_chunk} → {new_size}, retrying in {sleep_for:.1f}s")
                    local_chunk = new_size
                else:
                    print(f"Rate limit: retrying in {sleep_for:.1f}s with chunk {local_chunk}")
            else:
                print(f"Error: {msg}. Retrying in {sleep_for:.1f}s")
            time.sleep(sleep_for)
    raise last_err

def main(input_file_path: str):
    load_env_or_fail()
    df = load_input_csv(input_file_path)
    df = filter_emails(df, "Email")
    print(f"After email filter: {len(df)} prospects.")

    # Filter out rows with empty job titles (check for both pandas NaN and empty strings)
    df = df[df["Job Title"].notna() & (df["Job Title"].astype(str).str.strip() != "")]
    print(f"After non-empty Job Title filter: {len(df)} prospects.")

    df = df[["Prospect Id", "Email", "Job Title"]]
    print(f"Final: {len(df)} prospects to process.")

    # Load system instructions and persona definitions
    frame = read_text(FRAME_FILE)
    personas = read_text(PERSONAS_FILE)
    system = frame + personas
    session = create_chat_session(system_message=system, model=STREAM_MODEL)

    # Initialize chunk size and tracking variables
    # Start at MAX_CHUNK for faster initial processing (will reduce if rate limited)
    current_chunk = MAX_CHUNK
    remaining_ids = set(df["Prospect Id"].tolist())
    all_results = []

    # Multi-pass processing: retry failed prospects with smaller chunks
    for p in range(1, MAX_PASSES + 1):
        if not remaining_ids:
            break
        print(f"\n===== PASS {p}/{MAX_PASSES} | remaining={len(remaining_ids)} | chunk={current_chunk} =====")
        df_pass = df[df["Prospect Id"].isin(remaining_ids)].copy()
        failed_ids = set()

        i = 0
        with tqdm(total=len(df_pass)) as bar:
            while i < len(df_pass):
                end_i = min(i + current_chunk, len(df_pass))
                chunk = df_pass.iloc[i:end_i].copy()
                # Sanitize job titles (remove commas to preserve CSV structure)
                chunk.loc[:, "Job Title"] = chunk["Job Title"].apply(sanitize_job_title)

                # Calculate pacing to stay within token budget
                est = estimate_tokens(len(chunk))
                pace = max(BASE_SLEEP_SEC, est / max(1, TARGET_TPM_BUDGET) * 60.0)

                # Format chunk as CSV-like table for LLM
                job_titles_table = "\n".join([f"{r['Prospect Id']},{r['Job Title']}" for _, r in chunk.iterrows()])
                try:
                    resp, current_chunk = call_with_retries(session, job_titles_table, current_chunk)
                    if resp:
                        all_results.append(resp)
                except (TimeoutError, requests.HTTPError, requests.exceptions.RequestException, RuntimeError) as e:
                    failed_ids.update(chunk["Prospect Id"].tolist())
                    # Print a clear message plus full traceback so the root cause is visible
                    print("\n===== ERROR DURING CHUNK PROCESSING =====")
                    print(f"Chunk index range: {i}:{end_i} (pass {p})")
                    print(f"Prospect Ids in failing chunk: {chunk['Prospect Id'].tolist()}")
                    print(f"Exception: {repr(e)}")
                    traceback.print_exc()
                    print("===== END ERROR =====\n")

                # Pacing with jitter to avoid synchronized requests
                time.sleep(pace + random.uniform(0, 0.75))
                bar.update(end_i - i)
                i = end_i

        # Parse results and identify successfully processed prospects
        enriched = "\n".join([r for r in all_results if r])
        formatted = parse_llm_csv(enriched)
        ok_ids = set()
        if not formatted.empty:
            formatted = formatted.drop_duplicates(subset=["Prospect Id"], keep="first")
            ok_ids = set(formatted[formatted["Persona"].isin(VALID_PERSONAS)]["Prospect Id"].astype(str))

        # Update remaining IDs: remove successful ones, add failed ones
        before = len(remaining_ids)
        remaining_ids = (remaining_ids - ok_ids) | failed_ids
        after = len(remaining_ids)
        print(f"PASS {p} summary: processed {before - after}, remaining {after}")
        # Reduce chunk size for next pass if there are still remaining IDs
        if remaining_ids:
            current_chunk = max(MIN_CHUNK, current_chunk // 2)

    # Final parse + merge + save
    enriched = "\n".join([r for r in all_results if r])
    formatted = parse_llm_csv(enriched)

    # Merge with original data and determine skip reasons
    merged = df.merge(formatted, on="Prospect Id", how="left")
    merged["Skip Reason"] = merged.apply(determine_skip_reason, axis=1)

    # Separate accepted and skipped prospects
    final_df = merged[merged["Skip Reason"].isna()].copy()
    skipped_df = merged[merged["Skip Reason"].notna()].copy()

    # Clean up duplicate Job Title columns from merge
    if "Job Title_y" in final_df.columns:
        final_df = final_df.drop(columns="Job Title_y").rename(
            columns={"Job Title_x": "Job Title"}
        )
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
            # Clean up duplicate Job Title columns if present
            if "Job Title_y" in corrected_merged.columns:
                corrected_merged = corrected_merged.drop(
                    columns="Job Title_y"
                ).rename(columns={"Job Title_x": "Job Title"})
            final_df = pd.concat([final_df, corrected_merged], ignore_index=True)
            final_df = final_df.drop_duplicates(
                subset=["Prospect Id"], keep="first"
            )
            skipped_df = still_skipped_df

    # Save outputs
    accepted_path, skipped_path = save_outputs(final_df, skipped_df)
    print("\n========= Processing Results =========")
    print(f"{len(final_df)} prospects updated")
    print(f"{len(skipped_df)} prospects skipped")
    print(f"\nAccepted: {accepted_path}\nSkipped:  {skipped_path}")


def _resolve_input_path(arg_path: str | None) -> str:
    """Resolve input file path from argument or user prompt, handling Hubspot zips.

    If arg_path is provided, use it. Otherwise, prompt the user in the terminal.
    Automatically handles Hubspot zip files by extracting and locating the CSV.
    Cleans quotes, expands ~, and validates existence.

    Args:
        arg_path: Optional file path from command line argument.

    Returns:
        Path to the CSV file to process (extracted from zip if needed).
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
        # Allow relative paths by making them absolute
        arg_path = os.path.abspath(arg_path)

    # Use resolve_input_file to handle both CSV and Hubspot zip files
    try:
        return resolve_input_file(arg_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Streaming (adaptive) enrichment")
    # --input is now optional; we’ll prompt if missing
    ap.add_argument("--input", required=False, help="Path to prospects CSV (if omitted, you will be prompted)")
    args = ap.parse_args()

    input_path = _resolve_input_path(args.input)
    main(input_path)
