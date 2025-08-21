#!/usr/bin/env python3
import os, sys, json, pandas as pd

from config import BATCH_MODEL, FRAME_FILE, PERSONAS_FILE, VALID_PERSONAS
from io_utils import load_env_or_fail, load_input_csv, filter_emails, read_text, save_outputs, save_checkpoint_raw
from parsing import sanitize_job_title, parse_batch_output_jsonl
from batch_core import upload_file_for_batch, create_batch, poll_batch_until_done, download_file_content

def build_requests_jsonl(df: pd.DataFrame, system_instructions: str, model: str, temperature: float = 0.0) -> bytes:
    sys_msg = (
        system_instructions.strip()
        + "\n\nCRITICAL OUTPUT FORMAT: Respond with a SINGLE JSON object only.\n"
        + 'Required keys: {"persona": <one of the defined personas>, "certainty": <0-100 integer or %>}.\n'
        + "Do not include extra keys, code fences, or commentary."
    )
    items = []
    for _, r in df.iterrows():
        pid = str(r["Prospect Id"])
        user = f"Prospect Id: {pid}\nJob Title: {sanitize_job_title(r['Job Title'])}\n\nReturn ONLY the JSON."
        items.append({
            "custom_id": pid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model, "messages": [{"role":"system","content":sys_msg},{"role":"user","content":user}], "temperature": temperature}
        })
    return ("\n".join(json.dumps(x, ensure_ascii=False) for x in items)).encode("utf-8")

def main(input_path: str, resume_batch_id: str | None = None, print_status: bool = True):
    api_key = load_env_or_fail()
    df = load_input_csv(input_path)
    df = filter_emails(df, "Email")
    df = df[df["Job Title"].notna()]
    df = df[["Prospect Id", "Email", "Job Title"]]
    if df.empty:
        print("No valid rows to process."); return

    frame = read_text(FRAME_FILE); personas = read_text(PERSONAS_FILE)
    system = frame + personas

    if resume_batch_id:
        batch_id = resume_batch_id
        print(f"Resuming batch {batch_id}")
        meta = poll_batch_until_done(api_key, batch_id, echo=print_status)
    else:
        req = build_requests_jsonl(df, system_instructions=system, model=BATCH_MODEL)
        input_file_id = upload_file_for_batch(api_key, req)
        print(f"Uploaded requests file id: {input_file_id}")
        batch_id = create_batch(api_key, input_file_id)
        print(f"Created batch id: {batch_id}")
        meta = poll_batch_until_done(api_key, batch_id, echo=print_status)

    if meta.get("status") != "completed":
        save_checkpoint_raw(f"batch_{batch_id}_meta", meta)
        raise RuntimeError(f"Batch not completed: {meta.get('status')}")

    out_file = meta.get("output_file_id") or (meta.get("output_file_ids") or [None])[0]
    if not out_file:
        raise RuntimeError("No output file id in batch response.")

    jsonl_output = download_file_content(api_key, out_file)
    save_checkpoint_raw(f"batch_{batch_id}_output", jsonl_output)

    result_map, errors_map = parse_batch_output_jsonl(jsonl_output)

    rows = []
    for pid, content in result_map.items():
        try:
            obj = json.loads(content)
            rows.append({"Prospect Id": pid, "Persona": str(obj.get("persona", "")).strip(),
                         "Persona Certainty": str(obj.get("certainty","")).strip()})
        except Exception as e:
            errors_map[pid] = f"Invalid JSON: {e}: {content[:160]}..."

    formatted = pd.DataFrame(rows, columns=["Prospect Id", "Persona", "Persona Certainty"])
    merged = df.merge(formatted, on="Prospect Id", how="left")

    def skip_reason(row):
        pid = str(row.get("Prospect Id",""))
        persona = str(row.get("Persona","") or "").strip()
        if not persona:
            return f"Batch error: {errors_map.get(pid,'No LLM response')}"
        if persona not in VALID_PERSONAS:
            return f"Invalid persona: {persona}"
        return None

    merged["Skip Reason"] = merged.apply(skip_reason, axis=1)
    final_df = merged[merged["Skip Reason"].isna()].copy()
    skipped_df = merged[merged["Skip Reason"].notna()].copy()
    final_df = final_df.drop_duplicates(subset=["Prospect Id"], keep="first")
    final_df = final_df[final_df["Persona"].isin(VALID_PERSONAS)]

    accepted_path, skipped_path = save_outputs(final_df, skipped_df)
    print("\n========= Processing Results =========")
    print(f"{len(final_df)} prospects updated")
    print(f"{len(skipped_df)} prospects skipped")
    print(f"\nAccepted: {accepted_path}\nSkipped:  {skipped_path}\nBatch id: {batch_id}")
    
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
        arg_path = os.path.abspath(arg_path)

    if not os.path.exists(arg_path):
        print(f"Input file not found: {arg_path}")
        sys.exit(1)

    return arg_path

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