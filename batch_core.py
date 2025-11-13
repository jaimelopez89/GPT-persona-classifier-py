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


# ---------- File & Batch creation helpers ----------

def upload_file_for_batch(api_key: str, jsonl_bytes: bytes, filename: str = "requests.jsonl") -> str:
    """Upload a JSONL file to OpenAI for use with the Batch API.

    Uploads a file containing batch requests (one JSON object per line) to
    OpenAI's file storage. The file must be formatted according to Batch API
    requirements.

    Args:
        api_key: OpenAI API key for authentication.
        jsonl_bytes: The JSONL file content as bytes.
        filename: Name for the uploaded file (default: "requests.jsonl").

    Returns:
        The file ID of the uploaded file (used to create batch jobs).

    Raises:
        requests.HTTPError: If the upload fails (non-2xx status code).
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
    """Create a batch job to process an uploaded JSONL file.

    Creates a batch job that will asynchronously process all requests in the
    uploaded file. The job runs in the background and can be polled for status.

    Args:
        api_key: OpenAI API key for authentication.
        input_file_id: File ID from upload_file_for_batch.
        completion_window: Maximum time for the batch to complete (default: "24h").

    Returns:
        The batch ID (used to poll for status and retrieve results).

    Raises:
        requests.HTTPError: If batch creation fails (non-2xx status code).
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
    """Retrieve the current status and metadata of a batch job.

    Args:
        api_key: OpenAI API key for authentication.
        batch_id: The batch ID returned from create_batch.

    Returns:
        Dictionary containing batch metadata including status, request counts,
        timestamps, and output file ID (when completed).

    Raises:
        requests.HTTPError: If the request fails (non-2xx status code).
    """
    r = requests.get(
        f"https://api.openai.com/v1/batches/{batch_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,  # Generous per-call timeout
    )
    r.raise_for_status()
    return r.json()


def download_file_content(api_key: str, file_id: str, *, stream: bool = False, timeout: int = 300) -> str:
    """Download the content of a file from OpenAI.

    Downloads file content, with optional streaming for very large files.
    By default, loads the entire file into memory.

    Args:
        api_key: OpenAI API key for authentication.
        file_id: The file ID to download.
        stream: If True, download in chunks (useful for large files).
        timeout: Request timeout in seconds (default: 300).

    Returns:
        File contents as a UTF-8 decoded string.

    Raises:
        requests.HTTPError: If the download fails (non-2xx status code).
    """
    url = f"https://api.openai.com/v1/files/{file_id}/content"
    headers = {"Authorization": f"Bearer {api_key}"}
    if stream:
        # Stream download for large files
        with requests.get(url, headers=headers, timeout=timeout, stream=True) as r:
            r.raise_for_status()
            chunks = []
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                if chunk:
                    chunks.append(chunk)
            return b"".join(chunks).decode("utf-8", errors="replace")
    else:
        # Simple download for smaller files
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.text


# ---------- ETA helpers & resilient poller ----------

def _fmt_eta(seconds: float | None) -> str:
    """Format estimated time remaining as a human-readable string.

    Args:
        seconds: Number of seconds remaining, or None if unknown.

    Returns:
        Formatted string like "2h 30m 15s", "45m 30s", "30s", or "—" if unknown.
    """
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
    """Estimate time remaining for a batch job.

    Calculates ETA based on completed requests and elapsed time since the
    batch started processing.

    Args:
        meta: Batch metadata dictionary from retrieve_batch.

    Returns:
        Tuple of (completed_count, total_count, eta_seconds):
        - completed_count: Number of requests completed.
        - total_count: Total number of requests.
        - eta_seconds: Estimated seconds remaining, or None if cannot estimate.
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
    """Poll a batch job until it reaches a terminal state.

    Continuously polls the batch status until it completes, fails, or is canceled.
    Implements exponential backoff on network errors and provides progress updates
    with ETA estimates.

    Args:
        api_key: OpenAI API key for authentication.
        batch_id: The batch ID to poll.
        poll_interval: Seconds between polls when successful (default: 20.0).
        max_backoff: Maximum backoff time in seconds for retries (default: 120.0).
        echo: If True, print full JSON metadata; if False, print summary (default: False).
        hard_timeout_seconds: Abort polling after this many seconds (default: None).

    Returns:
        Final batch metadata dictionary when batch reaches terminal state.

    Raises:
        TimeoutError: If hard_timeout_seconds is exceeded.
        requests.RequestException: For persistent network errors (after backoff).
    """
    start = time.time()
    backoff = poll_interval
    while True:
        # Check hard timeout if set
        if hard_timeout_seconds is not None and (time.time() - start) > hard_timeout_seconds:
            raise TimeoutError(f"Batch {batch_id} did not finish within {hard_timeout_seconds}s")

        try:
            meta = retrieve_batch(api_key, batch_id)
            status = meta.get("status")
            completed, total, eta = _estimate_eta(meta)
            eta_txt = _fmt_eta(eta)

            if echo:
                # Print full JSON for debugging
                print(json.dumps(meta, indent=2))
            else:
                # Print concise progress summary
                print(f"Batch {batch_id} status={status} | {completed}/{total} done | ETA {eta_txt}")

            # Check if batch is in terminal state
            if status in ("completed", "failed", "cancelled", "canceled"):
                return meta

            time.sleep(poll_interval)
            backoff = poll_interval  # Reset backoff on success
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