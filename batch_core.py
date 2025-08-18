import requests, time, json

def upload_file_for_batch(api_key: str, jsonl_bytes: bytes, filename="requests.jsonl"):
    files = {"file": (filename, jsonl_bytes, "application/jsonl")}
    data = {"purpose": "batch"}
    r = requests.post("https://api.openai.com/v1/files",
                      headers={"Authorization": f"Bearer {api_key}"},
                      files=files, data=data, timeout=120)
    r.raise_for_status()
    return r.json()["id"]

def create_batch(api_key: str, input_file_id: str, completion_window="24h"):
    payload = {"input_file_id": input_file_id, "endpoint": "/v1/chat/completions", "completion_window": completion_window}
    r = requests.post("https://api.openai.com/v1/batches",
                      headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                      json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["id"]

def retrieve_batch(api_key: str, batch_id: str):
    r = requests.get(f"https://api.openai.com/v1/batches/{batch_id}",
                     headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
    r.raise_for_status()
    return r.json()

def download_file_content(api_key: str, file_id: str) -> str:
    r = requests.get(f"https://api.openai.com/v1/files/{file_id}/content",
                     headers={"Authorization": f"Bearer {api_key}"}, timeout=300)
    r.raise_for_status()
    return r.text

def poll_batch_until_done(api_key: str, batch_id: str, poll_interval: float = 15.0, echo=False):
    while True:
        meta = retrieve_batch(api_key, batch_id)
        if echo: print(json.dumps(meta, indent=2))
        else:    print(f"Batch {batch_id} status: {meta.get('status')}")
        if meta.get("status") in ("completed", "failed", "cancelled", "canceled"):
            return meta
        time.sleep(poll_interval)