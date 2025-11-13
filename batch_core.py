"""Core functions for OpenAI Batch API operations.

This module provides functions for:
- Uploading files for batch processing
- Creating and managing batch jobs
- Polling batch status with ETA estimation
- Downloading batch results

The batch API allows processing large numbers of requests asynchronously,
which is more cost-effective and rate-limit friendly than streaming.

Author: Jaime López, 2025
"""

import time
import json
import math
import requests

# ---------- File & Batch creation helpers (restored) ----------

def upload_file_for_batch(api_key: str, jsonl_bytes: bytes, filename: str = "requests.jsonl") -> str:
    """
    Uploads a JSONL file (one request per line) for use with the Batch API.
    Returns the uploaded file's ID.
    """
    files = {"file": (filename, jsonl_bytes, "application/jsonl")}
    data = {"purpose": "batch"}
    r = requests.post(
        "https://api.openai.com/v1/files",
        headers={"Authorization": f"Bearer {api_key}"},
        files=files,
        data=data,
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["id"]

def create_batch(api_key: str, input_file_id: str, completion_window: str = "24h") -> str:
    """
    Creates a batch job that will process the uploaded JSONL file.
    Returns the batch ID.
    """
    payload = {
        "input_file_id": input_file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": completion_window,
    }
    r = requests.post(
        "https://api.openai.com/v1/batches",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["id"]

def retrieve_batch(api_key: str, batch_id: str) -> dict:
    r = requests.get(
        f"https://api.openai.com/v1/batches/{batch_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,  # generous per-call timeout
    )
    r.raise_for_status()
    return r.json()

def download_file_content(api_key: str, file_id: str, *, stream: bool = False, timeout: int = 300) -> str:
    """
    Downloads the content of a file. For very large outputs, you may want to set stream=True
    and write to disk chunk-by-chunk. By default, returns the entire text content.
    """
    url = f"https://api.openai.com/v1/files/{file_id}/content"
    headers = {"Authorization": f"Bearer {api_key}"}
    if stream:
        with requests.get(url, headers=headers, timeout=timeout, stream=True) as r:
            r.raise_for_status()
            chunks = []
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    chunks.append(chunk)
            return b"".join(chunks).decode("utf-8", errors="replace")
    else:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.text

# ---------- ETA helpers & resilient poller (new) ----------

def _fmt_eta(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "—"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"

def _estimate_eta(meta: dict) -> tuple[int, int, float | None]:
    """
    Returns (completed, total, eta_seconds or None) using:
      - request_counts.completed / total
      - in_progress_at (or started_at) to compute throughput
    """
    rc = (meta or {}).get("request_counts") or {}
    completed = int(rc.get("completed") or 0)
    failed = int(rc.get("failed") or 0)
    total = int(rc.get("total") or (completed + failed))
    if not total:
        return completed, total, None

    # Estimate throughput since work started
    in_prog = meta.get("in_progress_at") or meta.get("started_at")
    if not in_prog:
        return completed, total, None

    now = int(time.time())
    elapsed = max(1, now - int(in_prog))  # seconds
    rate = completed / elapsed  # requests per second
    remaining = max(0, total - completed)
    eta = None if rate <= 0 else remaining / rate
    return completed, total, eta

def poll_batch_until_done(
    api_key: str,
    batch_id: str,
    poll_interval: float = 20.0,
    max_backoff: float = 120.0,
    echo: bool = False,
    hard_timeout_seconds: float | None = None,
) -> dict:
    """
    Robust poller:
      - Keeps polling until the batch reaches a terminal state (completed/failed/canceled)
      - Exponential backoff (with cap) on transient HTTP errors
      - Prints progress + ETA using request_counts and timestamps
      - Optional hard_timeout_seconds to abort after a fixed wall-clock time (e.g., in CI)
    """
    start = time.time()
    backoff = poll_interval
    while True:
        if hard_timeout_seconds is not None and (time.time() - start) > hard_timeout_seconds:
            raise TimeoutError(f"Batch {batch_id} did not finish within {hard_timeout_seconds}s")

        try:
            meta = retrieve_batch(api_key, batch_id)
            status = meta.get("status")
            completed, total, eta = _estimate_eta(meta)
            eta_txt = _fmt_eta(eta)

            if echo:
                print(json.dumps(meta, indent=2))
            else:
                print(f"Batch {batch_id} status={status} | {completed}/{total} done | ETA {eta_txt}")

            if status in ("completed", "failed", "cancelled", "canceled"):
                return meta

            time.sleep(poll_interval)
            backoff = poll_interval  # reset backoff on success
        except requests.RequestException as e:
            # Transient network issue: back off and keep trying
            print(f"[poll] network error: {e}; backing off {int(backoff)}s…")
            time.sleep(backoff)
            backoff = min(max_backoff, backoff * 2)

# Optional: tidy exports for linters/IDEs
__all__ = [
    "upload_file_for_batch",
    "create_batch",
    "retrieve_batch",
    "download_file_content",
    "poll_batch_until_done",
]