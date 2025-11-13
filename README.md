# GPT-persona-classifier-py

This repository contains a Python toolkit for automatically classifying prospects into personas based on their **job titles** (and other fields you choose). It’s designed to work both as a **local script** (for ad-hoc runs) and as part of an **automated nightly pipeline** (e.g. exporting from HubSpot, enriching with OpenAI, and importing results back).

At a high level:

- You provide a **CSV** of prospects (with columns like `Prospect Id`, `Email`, `Job Title`).
- The code calls **OpenAI Chat models** with your custom instructions and persona definitions.
- It outputs:
  - A **personas CSV** (accepted rows with valid personas)
  - A **skipped CSV** (rows that couldn’t be classified, with a “Skip Reason”)
- There are two main processing modes:
  - **Streaming mode**: direct API calls with retry/backoff, chunking, and multi-pass re-tries.
  - **Batch mode**: OpenAI Batch API for very large offline runs (plus a “rerun only skipped” tool).

---

## Repository Structure

Key modules:

- **`config.py`**  
  Central configuration:
  - Model names (`STREAM_MODEL`, `BATCH_MODEL`)
  - Rate-limit tuning (TPM budget, sleep intervals, chunk sizes, max passes)
  - Output directories (accepted, skipped, checkpoints)
  - Paths to instruction files (`frame_instructions.txt`, `persona_definitions.txt`)
  - Canonical set of `VALID_PERSONAS`

- **`io_utils.py`**  
  I/O utilities:
  - Load `.env` and `OPENAI_API_KEY`
  - Load the input CSV and normalize columns (e.g. `Record ID` → `Prospect Id`)
  - Filter internal/test emails (`filter_emails`)
  - Read instructions from disk
  - Save accepted / skipped CSVs and checkpoint files

- **`parsing.py`**  
  Parsing and data cleaning:
  - `sanitize_job_title()` – normalizes job titles (e.g. remove commas).
  - `parse_llm_csv()` – parses CSV-style responses (streaming mode).
  - `parse_batch_output_jsonl()` – parses OpenAI Batch JSONL output into a map of `ProspectId → response`.
  - `determine_skip_reason()` – explains why a row was skipped (no response, invalid persona, etc).

- **`llm_client.py`**  
  Thin OpenAI Chat client for streaming mode:
  - `create_chat_session()` – constructs a chat session with system message + model.
  - `ask_chat_session()` – calls the `/v1/chat/completions` endpoint with proper error handling and timeouts.
  - `extract_retry_after_seconds()` – parses “try again in Xs” from error messages to help with backoff.

- **`streaming_enricher.py`**  
  The streaming (online) enrichment script:
  - Accepts an input CSV path from `--input` or prompts interactively.
  - Loads and filters prospects.
  - Builds a single chat session with combined system instructions:
    - `frame_instructions.txt`
    - `persona_definitions.txt`
  - Processes prospects in **chunks** of job titles, with:
    - Token estimation (`estimate_tokens`) to pace requests.
    - Adaptive chunk sizing & retries (`call_with_retries`) to respect rate limits.
    - **Multi-pass** loop:
      - Pass 1: process everyone.
      - Pass 2..N: re-run only prospects that:
        - Were in failing chunks, or
        - Still have no valid persona.
  - Aggregates the model’s CSV-like responses, parses them, merges with the original data, and writes:
    - Final persona file (accepted).
    - Skipped file with `Skip Reason`.

- **`batch_core.py`**  
  Shared helpers for OpenAI Batch API:
  - Upload a JSONL file (`upload_file_for_batch`).
  - Create a batch job (`create_batch`).
  - Poll batch status until completion (with ETA estimation and exponential backoff).
  - Download the batch output file content.

- **`batch_enricher.py`**  
  The batch-mode enrichment script:
  - Accepts input CSV from `--input` or prompts the user.
  - Optionally resumes from an existing `--resume-batch-id`.
  - Builds a JSONL file with **one request per prospect**, each specifying:
    - The system instructions.
    - A user message with `Prospect Id` and `Job Title`.
    - A forced JSON response format for robust parsing.
  - Uses `batch_core` to:
    - Upload the JSONL.
    - Create and poll a batch.
    - Download the final JSONL results.
  - Parses each assistant response as JSON (`{ persona, certainty }`).
  - Merges results with the input dataframe and writes the same accepted/skipped CSVs as streaming mode.

- **`batch_rerun_skipped.py`**  
  A tool to **rerun only skipped prospects** (Batch mode):
  - Reads a previous “Skipped prospects …” CSV.
  - Picks only rows that still lack a persona.
  - Creates a new batch job for those rows.
  - Merges any new persona results back into the skipped set.
  - Outputs:
    - `Personas Rerun <timestamp>.csv`
    - `Skipped prospects Rerun <timestamp>.csv`

- **HubSpot integration helpers (optional, if present)**  
  There may be small helpers like:
  - `hubspot_export.py` – uses HubSpot Export API to pull a CSV (e.g. daily contacts from a list).
  - `hubspot_import.py` – uses HubSpot Imports API to push the enriched CSV back to HubSpot.
  - `scripts/nightly_driver.py` – an orchestrator that:
    - Exports from HubSpot.
    - Runs `batch_enricher`.
    - Imports results back into HubSpot.
  These tie the classifier into a fully automated, nightly enrichment pipeline.

- **Makefile / justfile / README**  
  Convenience wrappers to:
  - Create a virtualenv and install dependencies.
  - Run lint/format.
  - Run the streaming/batch/rerun workflows with single commands.

---

## How It Works (Conceptually)

1. **Input data**  
   A CSV of prospects with at minimum:
   - `Prospect Id` (or `Record ID`, which is renamed)
   - `Email`
   - `Job Title`  
   Additional fields (e.g. `First Name`, `Last Name`, `Company`) can be present but are not required for the LLM call in the base flow.

2. **Filtering**  
   The `filter_emails` function removes:
   - Internal Ververica emails (by domain).
   - Obvious test/dummy/fake addresses (configurable).
   Rows without a usable job title are also excluded from enrichment.

3. **Instructions**  
   Two text files drive the persona logic:
   - `frame_instructions.txt` – general framing (“You are a B2B persona classifier”, expected output style, etc.).
   - `persona_definitions.txt` – definitions of personas like:
     - *Executive Sponsor*
     - *Economic Buyer*
     - *Data Product Manager/Owner*
     - *Data User*
     - *Application Developer*
     - *Real-time Specialist*
     - *Operator/Systems Administrator*
     - *Technical Decision Maker*
     - *Not a target*

   These are concatenated and used as the system message for the model.

4. **LLM call pattern**

   - **Streaming mode**:
     - Groups prospects into chunks (e.g. 20–80 rows).
     - Sends them as a single user message:  
       `ProspectId,Job Title` per line.
     - Model responds with CSV-like lines (`ProspectId,Job Title,Persona,Persona Certainty`).
     - Script parses and merges results.

   - **Batch mode**:
     - One JSONL line per prospect:
       - Single job title + prospect id.
       - Hard requirement that the model returns pure JSON:
         ```json
         {"persona": "Economic Buyer", "certainty": 92}
         ```
     - Results are read from the batch output file and merged by `Prospect Id`.

5. **Multi-pass / rerun logic**
   - Streaming mode:
     - Runs multiple passes over the data, shrinking chunk size and re-attempting only rows that remain missing or invalid.
   - Batch mode:
     - Designed to handle very large datasets in one go.
     - `batch_rerun_skipped.py` re-submits only previously skipped rows.

6. **Outputs**
   - **Accepted** (persona classified) CSV:
     - Includes original columns plus `Persona` and `Persona Certainty`.
   - **Skipped** CSV:
     - Includes `Skip Reason` explaining:
       - No LLM response.
       - Invalid persona (not in `VALID_PERSONAS`).
       - Batch error details.

---

## Quickstart

### 1. Setup environment

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
