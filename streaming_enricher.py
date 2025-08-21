#!/usr/bin/env python3
import time, math, random, os, sys, pandas as pd
from tqdm import tqdm

from config import (
    STREAM_MODEL, TARGET_TPM_BUDGET, BASE_SLEEP_SEC, MAX_RETRIES, INITIAL_BACKOFF,
    MAX_BACKOFF, MIN_CHUNK, MAX_CHUNK, SAFETY_TOKEN_PER_ROW, MAX_PASSES, VALID_PERSONAS,
    FRAME_FILE, PERSONAS_FILE
)
from io_utils import load_env_or_fail, load_input_csv, filter_emails, read_text, save_outputs
from parsing import sanitize_job_title, parse_llm_csv, determine_skip_reason
from llm_client import create_chat_session, ask_chat_session, extract_retry_after_seconds

def estimate_tokens(n: int) -> int:
    return n * SAFETY_TOKEN_PER_ROW

def call_with_retries(session, payload_text: str, chunk_size: int):
    local_chunk = chunk_size
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = ask_chat_session(session=session, user_message=payload_text)
            return resp, local_chunk
        except Exception as e:
            msg = str(e); last_err = e
            server_wait = extract_retry_after_seconds(msg)
            backoff = min(MAX_BACKOFF, (INITIAL_BACKOFF * (2 ** attempt)) + random.uniform(0, 1.0))
            sleep_for = max(server_wait, backoff)
            if "rate limit" in msg.lower() or "429" in msg:
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

def main(input_path: str):
    load_env_or_fail()
    df = load_input_csv(input_path)
    df = filter_emails(df, "Email")
    df = df[df["Job Title"].notna()]
    df = df[["Prospect Id", "Email", "Job Title"]]

    frame = read_text(FRAME_FILE)
    personas = read_text(PERSONAS_FILE)
    system = frame + personas
    session = create_chat_session(system_message=system, model=STREAM_MODEL)

    current_chunk = min(MAX_CHUNK, 80)
    remaining_ids = set(df["Prospect Id"].tolist())
    all_results = []

    for p in range(1, MAX_PASSES + 1):
        if not remaining_ids: break
        print(f"\n===== PASS {p}/{MAX_PASSES} | remaining={len(remaining_ids)} | chunk={current_chunk} =====")
        df_pass = df[df["Prospect Id"].isin(remaining_ids)].copy()
        failed_ids = set()

        i = 0
        with tqdm(total=len(df_pass)) as bar:
            while i < len(df_pass):
                end_i = min(i + current_chunk, len(df_pass))
                chunk = df_pass.iloc[i:end_i].copy()
                chunk.loc[:, "Job Title"] = chunk["Job Title"].apply(sanitize_job_title)

                est = estimate_tokens(len(chunk))
                pace = max(BASE_SLEEP_SEC, est / max(1, TARGET_TPM_BUDGET) * 60.0)

                job_titles_table = "\n".join([f"{r['Prospect Id']},{r['Job Title']}" for _, r in chunk.iterrows()])
                try:
                    resp, current_chunk = call_with_retries(session, job_titles_table, current_chunk)
                    if resp: all_results.append(resp)
                except Exception as e:
                    failed_ids.update(chunk["Prospect Id"].tolist())
                    print(f"Final failure for idx {i}:{end_i} (pass {p}) -> {e}")

                time.sleep(pace + random.uniform(0, 0.75))
                bar.update(end_i - i)
                i = end_i

        enriched = "\n".join([r for r in all_results if r])
        formatted = parse_llm_csv(enriched)
        ok_ids = set()
        if not formatted.empty:
            formatted = formatted.drop_duplicates(subset=["Prospect Id"], keep="first")
            ok_ids = set(formatted[formatted["Persona"].isin(VALID_PERSONAS)]["Prospect Id"].astype(str))

        before = len(remaining_ids)
        remaining_ids = (remaining_ids - ok_ids) | failed_ids
        after = len(remaining_ids)
        print(f"PASS {p} summary: processed {before - after}, remaining {after}")
        if remaining_ids:
            current_chunk = max(MIN_CHUNK, current_chunk // 2)

    # Final parse + merge + save
    enriched = "\n".join([r for r in all_results if r])
    formatted = parse_llm_csv(enriched)

    merged = df.merge(formatted, on="Prospect Id", how="left")
    merged["Skip Reason"] = merged.apply(determine_skip_reason, axis=1)

    final_df = merged[merged["Skip Reason"].isna()].copy()
    skipped_df = merged[merged["Skip Reason"].notna()].copy()

    if "Job Title_y" in final_df.columns:
        final_df = final_df.drop(columns="Job Title_y").rename(columns={"Job Title_x": "Job Title"})
    final_df = final_df.drop_duplicates(subset=["Prospect Id"], keep="first")
    final_df = final_df[final_df["Persona"].isin(VALID_PERSONAS)]

    accepted_path, skipped_path = save_outputs(final_df, skipped_df)
    print("\n========= Processing Results =========")
    print(f"{len(final_df)} prospects updated")
    print(f"{len(skipped_df)} prospects skipped")
    print(f"\nAccepted: {accepted_path}\nSkipped:  {skipped_path}")
    
def _resolve_input_path(arg_path: str | None) -> str:
    """
    If arg_path is provided, use it. Otherwise, prompt the user in the terminal.
    Cleans quotes, expands ~, and validates existence.
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

    if not os.path.exists(arg_path):
        print(f"Input file not found: {arg_path}")
        sys.exit(1)

    return arg_path

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Streaming (adaptive) enrichment")
    # --input is now optional; we’ll prompt if missing
    ap.add_argument("--input", required=False, help="Path to prospects CSV (if omitted, you will be prompted)")
    args = ap.parse_args()

    input_path = _resolve_input_path(args.input)
    main(input_path)
