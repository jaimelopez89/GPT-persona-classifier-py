# batch_core.py
import time, json, math, requests

def retrieve_batch(api_key: str, batch_id: str):
    r = requests.get(
        f"https://api.openai.com/v1/batches/{batch_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,  # generous per-HTTP call timeout
    )
    r.raise_for_status()
    return r.json()

def _fmt_eta(seconds: float) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "—"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h {m}m {s}s"
    if m: return f"{m}m {s}s"
    return f"{s}s"

def _estimate_eta(meta: dict) -> tuple[int, int, int | None]:
    """
    Returns (completed, total, eta_seconds or None) using:
      - request_counts.completed / total
      - in_progress_at timestamp to compute throughput
    """
    rc = (meta or {}).get("request_counts") or {}
    completed = int(rc.get("completed") or 0)
    failed = int(rc.get("failed") or 0)
    total = int(rc.get("total") or (completed + failed))
    # If total is 0 or missing, no ETA
    if not total:
        return completed, total, None

    # Throughput since in_progress_at
    in_prog = meta.get("in_progress_at") or meta.get("started_at")
    now = int(time.time())
    if not in_prog:
        return completed, total, None

    elapsed = max(1, now - int(in_prog))  # seconds
    rate = completed / elapsed  # req/sec
    remaining = max(0, total - completed)
    eta = None if rate <= 0 else remaining / rate
    return completed, total, eta

def poll_batch_until_done(
    api_key: str,
    batch_id: str,
    poll_interval: float = 20.0,
    max_backoff: float = 120.0,
    echo: bool = False,
):
    """
    Robust poller:
      - No global timeout: will keep polling until terminal status
      - Exponential backoff (with cap) on transient HTTP errors
      - Prints progress + ETA using request_counts
    """
    backoff = poll_interval
    while True:
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
            backoff = poll_interval  # reset on success
        except requests.RequestException as e:
            # transient network issue: back off and keep trying
            print(f"[poll] network error: {e}; backing off {int(backoff)}s…")
            time.sleep(backoff)
            backoff = min(max_backoff, backoff * 2)