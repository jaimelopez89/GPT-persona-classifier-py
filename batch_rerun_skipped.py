#!/usr/bin/env python3
import json, pandas as pd
from config import BATCH_MODEL, FRAME_FILE, PERSONAS_FILE, VALID_PERSONAS, OUTPUT_DIR, SKIPPED_DIR
from io_utils import load_env_or_fail, read_text, save_checkpoint_raw, now_stamp
from parsing import sanitize_job_title, parse_batch_output_jsonl
from batch_core import upload_file_for_batch, create_batch, poll_batch_until_done, download_file_content

def build_requests_jsonl(df: pd.DataFrame, system_instructions: str, model: str) -> bytes:
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
            "body": {"model": model, "messages":[{"role":"system","content":sys_msg},{"role":"user","content":user}], "temperature":0.0}
        })
    return ("\n".join(json.dumps(x, ensure_ascii=False) for x in items)).encode("utf-8")

def main(skipped_path: str, print_status: bool = True):
    api_key = load_env_or_fail()
    df_skip = pd.read_csv(skipped_path, dtype=str)

    # rows still lacking Persona
    mask = ~df_skip.get("Persona","").astype(str).str.strip().astype(bool)
    df_todo = df_skip[mask].copy()
    if df_todo.empty:
        print("No rows without Persona in skipped CSV — nothing to re-run.")
        return

    frame = read_text(FRAME_FILE); personas = read_text(PERSONAS_FILE)
    system = frame + personas

    req = build_requests_jsonl(df_todo, system, model=BATCH_MODEL)
    input_file_id = upload_file_for_batch(api_key, req)
    print(f"Uploaded requests file id: {input_file_id}")
    batch_id = create_batch(api_key, input_file_id)
    print(f"Created batch id: {batch_id}")

    meta = poll_batch_until_done(api_key, batch_id, echo=print_status)
    if meta.get("status") != "completed":
        save_checkpoint_raw(f"rerun_batch_{batch_id}_meta", meta)
        raise RuntimeError(f"Batch rerun not completed: {meta.get('status')}")

    out_file = meta.get("output_file_id") or (meta.get("output_file_ids") or [None])[0]
    if not out_file:
        raise RuntimeError("No output file id in batch response.")
    jsonl_output = download_file_content(api_key, out_file)
    save_checkpoint_raw(f"rerun_batch_{batch_id}_output", jsonl_output)

    result_map, errors_map = parse_batch_output_jsonl(jsonl_output)

    rows = []
    for pid, content in result_map.items():
        try:
            obj = json.loads(content)
            rows.append({"Prospect Id": pid, "Persona": str(obj.get("persona","")).strip(),
                         "Persona Certainty": str(obj.get("certainty","")).strip()})
        except Exception as e:
            errors_map[pid] = f"Invalid JSON: {e}: {content[:160]}..."

    formatted = pd.DataFrame(rows, columns=["Prospect Id", "Persona", "Persona Certainty"])
    merged = df_skip.merge(formatted, on="Prospect Id", how="left", suffixes=("_old",""))

    def choose(new, old): return new if (isinstance(new, str) and new.strip()) else old
    if "Persona_old" in merged.columns:
        merged["Persona"] = merged.apply(lambda r: choose(r.get("Persona"), r.get("Persona_old")), axis=1)
        merged.drop(columns=["Persona_old"], inplace=True, errors="ignore")
    if "Persona Certainty_old" in merged.columns:
        merged["Persona Certainty"] = merged.apply(lambda r: choose(r.get("Persona Certainty"), r.get("Persona Certainty_old")), axis=1)
        merged.drop(columns=["Persona Certainty_old"], inplace=True, errors="ignore")

    def skip_reason(row):
        pid = str(row.get("Prospect Id", ""))
        persona = str(row.get("Persona","") or "").strip()
        if not persona:
            return f"Batch error: {errors_map.get(pid,'No LLM response')}"
        if persona not in VALID_PERSONAS:
            return f"Invalid persona: {persona}"
        return None

    merged["Skip Reason"] = merged.apply(skip_reason, axis=1)
    final_df = merged[merged["Skip Reason"].isna()].copy()
    still_skipped = merged[merged["Skip Reason"].notna()].copy()
    final_df = final_df.drop_duplicates(subset=["Prospect Id"], keep="first")
    final_df = final_df[final_df["Persona"].isin(VALID_PERSONAS)]

    stamp = now_stamp()
    accepted_path = OUTPUT_DIR / f"Personas Rerun {stamp}.csv"
    skipped_out = SKIPPED_DIR / f"Skipped prospects Rerun {stamp}.csv"
    final_df.to_csv(accepted_path, index=False)
    still_skipped.to_csv(skipped_out, index=False)

    print("\n========= Rerun Results =========")
    print(f"{len(final_df)} prospects updated on rerun")
    print(f"{len(still_skipped)} prospects still skipped")
    print(f"\nAccepted (rerun): {accepted_path}\nSkipped (rerun):  {skipped_out}\nBatch id: {batch_id}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Rerun only skipped via Batch API")
    ap.add_argument("--skipped", required=True)
    ap.add_argument("--print-status", action="store_true")
    args = ap.parse_args()
    main(args.skipped, args.print_status)