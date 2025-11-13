"""Parsing utilities for LLM responses and batch outputs.

This module provides functions for:
- Parsing CSV-like output from streaming LLM responses
- Parsing JSONL output from batch API responses
- Sanitizing job titles
- Determining skip reasons for invalid responses

Author: Jaime López, 2025
"""

import io
import re
import json
import pandas as pd
from config import VALID_PERSONAS

def sanitize_job_title(title: str) -> str:
    return re.sub(",", " ", str(title or ""))

# Streaming: parse CSV-like assistant output
def parse_llm_csv(enriched_result: str) -> pd.DataFrame:
    if not enriched_result.strip():
        return pd.DataFrame(columns=["Prospect Id", "Job Title", "Persona", "Persona Certainty"])
    _probe = pd.read_csv(io.StringIO(enriched_result), header=None, on_bad_lines="warn")
    if _probe.shape[1] > 4:
        print("Warning: extra columns present; using first four.")
    df = pd.read_csv(
        io.StringIO(enriched_result), header=None,
        names=["Prospect Id", "Job Title", "Persona", "Persona Certainty"],
        usecols=[0,1,2,3], dtype=str, on_bad_lines="warn"
    )
    return df

def determine_skip_reason(row) -> str | None:
    persona = str(row.get("Persona", "") or "").strip()
    if not persona:
        return "No LLM response"
    if persona not in VALID_PERSONAS:
        return f"Invalid persona: {persona}"
    return None

# Batch outputs (JSONL of responses)
def parse_batch_output_jsonl(jsonl_str: str):
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
                content = resp["body"]["choices"][0]["message"]["content"]
                out[cid] = content
            except (KeyError, TypeError, IndexError) as e:
                errors[cid] = f"Malformed success body: {e}"
        else:
            body = resp.get("body") or {}
            msg = None
            try:
                msg = body.get("error", {}).get("message")
            except (AttributeError, TypeError):
                msg = None
            errors[cid] = f"HTTP {status}: {msg or body}"
    return out, errors