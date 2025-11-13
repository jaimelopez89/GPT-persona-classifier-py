"""This is the main module that structures the Prospect and Job Title tables
   and submits them to the LLM using gpt_functions.py
   Jaime López, 2025
"""

import re
import io
import time
import random
import math
import os.path
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
from gpt_functions import *


load_dotenv()

# Function to filter out emails from Ververica and test emails
def filter_emails(df, column_name):
    df[column_name] = df[column_name].fillna('').astype(str)
    df = df[~df[column_name].str.contains("@ververica|test", regex=True)]
    return df

# Load the CSV file with prospects
path = input("Input the absolute path of the input file with prospects and no persona: ")
path = path.replace('"', '')
df = pd.read_csv(path, dtype=str)

# Rename "Record ID" to "Prospect Id" if necessary
if "Record ID" in df.columns:
    df.rename(columns={"Record ID": "Prospect Id"}, inplace=True)

# Ensure these columns are strings
cols_to_convert = ["Prospect Id", "Job Title", "First Name", "Last Name", "Email", "Company"]
cols_in_df = [col for col in cols_to_convert if col in df.columns]
df[cols_in_df] = df[cols_in_df].astype(str)

# Filter out unwanted emails and rows without a job title
df_filtered = filter_emails(df, 'Email')
df_filtered = df_filtered[df_filtered['Job Title'].notna()]

# Keep only the useful columns
cols_to_keep = ["Prospect Id", "Email", "Job Title"]
df_filtered = df_filtered[cols_to_keep]

# Load external instructions and persona definitions
with open("frame_instructions.txt", "r", encoding="utf-8") as f:
    frame_instructions = f.read()
with open("persona_definitions.txt", "r", encoding="utf-8") as f:
    persona_definitions = f.read()
complete_system_instructions = frame_instructions + persona_definitions

# Prepare the LLM session
session = create_chat_session(
    system_message=complete_system_instructions,
    # model="gpt-4o-mini"  # or "gpt-3.5-turbo-16k", "gpt-4", etc.
    model="gpt-4.1-nano"  # or "gpt-3.5-turbo-16k", "gpt-4", etc.
)

# ===== Adaptive chunking, pacing, and robust retries =====
# Conservative budget to avoid 429s for tokens per minute (TPM)
TARGET_TPM_BUDGET = 360_000   # tokens per minute budget
BASE_SLEEP_SEC = 1.5          # base sleep used in pacing
MAX_RETRIES = 6               # total attempts per API call
INITIAL_BACKOFF = 2.0         # seconds; grows exponentially with jitter
MAX_BACKOFF = 30.0            # cap between retries
MIN_CHUNK = 10                # do not go below this many rows per chunk
MAX_CHUNK = 100               # do not go above this many rows per chunk
SAFETY_TOKEN_PER_ROW = 120    # rough token estimate per row

total_rows = len(df_filtered)
print("Total rows:", total_rows)

current_chunk_size = min(MAX_CHUNK, 80)
results = []

def estimate_tokens(num_rows: int) -> int:
    # Very rough estimate: each row contributes ~SAFETY_TOKEN_PER_ROW tokens
    return num_rows * SAFETY_TOKEN_PER_ROW

def extract_retry_after_seconds(msg: str) -> float:
    # Try to parse a suggested wait time from the API error message, e.g. "try again in 12.1s"
    m = re.search(r"try again in ([0-9]+\.[0-9]+|[0-9]+)s", msg)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return 0.0
    return 0.0

def call_with_retries(payload_text: str, chunk_size: int):
    """Call the API with retries. Returns (response_text, updated_chunk_size)."""
    local_chunk_size = chunk_size
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = ask_chat_session(
                session=session,
                user_message=payload_text
            )
            return resp, local_chunk_size
        except Exception as e:
            msg = str(e)
            last_err = e
            server_wait = extract_retry_after_seconds(msg)
            backoff = min(MAX_BACKOFF, (INITIAL_BACKOFF * (2 ** attempt)) + random.uniform(0, 1.0))
            sleep_for = max(server_wait, backoff)

            # If rate limited, reduce chunk size to lower token pressure
            if "rate limit" in msg.lower() or "429" in msg:
                new_size = max(MIN_CHUNK, math.floor(local_chunk_size / 2))
                if new_size < local_chunk_size:
                    print(f"Rate limit detected. Reducing chunk size {local_chunk_size} -> {new_size} and retrying in {sleep_for:.1f}s")
                    local_chunk_size = new_size
                else:
                    print(f"Rate limit detected. Retrying in {sleep_for:.1f}s with chunk size {local_chunk_size}")
            else:
                print(f"Error: {msg}. Retrying in {sleep_for:.1f}s")

            time.sleep(sleep_for)
    # Exhausted retries
    raise last_err

i = 0
with tqdm(total=total_rows) as pbar:
    while i < total_rows:
        end_i = min(i + current_chunk_size, total_rows)
        chunk = df_filtered.iloc[i:end_i].copy()

        # Clean Job Title values (avoid commas to keep CSV shape)
        chunk.loc[:, 'Job Title'] = chunk['Job Title'].apply(lambda x: re.sub(",", " ", x))

        est_tokens = estimate_tokens(len(chunk))
        # Convert TPM budget into a sleep between requests
        pace_seconds = max(BASE_SLEEP_SEC, est_tokens / max(1, TARGET_TPM_BUDGET) * 60.0)

        job_titles_table = "\n".join(
            [f"{row['Prospect Id']},{row['Job Title']}" for _, row in chunk.iterrows()]
        )

        try:
            response, current_chunk_size = call_with_retries(job_titles_table, current_chunk_size)
            if response:
                results.append(response)
        except Exception as e:
            print(f"Final failure for rows {i}:{end_i} -> {e}")

        # Pacing to stay under TPM with a small jitter
        sleep_time = pace_seconds + random.uniform(0, 0.75)
        time.sleep(sleep_time)

        processed = end_i - i
        i = end_i
        pbar.update(processed)

print("Adaptive processing complete. Chunks may have been resized to respect limits.")

# Combine and clean the LLM responses
filtered_results = [result for result in results if result]
enriched_result = "\n".join(filtered_results)

# Attempt to parse the enriched result into a DataFrame
try:
    # Try reading without column restrictions first
    df_test = pd.read_csv(io.StringIO(enriched_result), header=None, on_bad_lines='warn')
    if df_test.shape[1] > 4:
        print("Warning: Extra columns detected. Only the first four columns will be used.")
    formatted_results = pd.read_csv(
        io.StringIO(enriched_result),
        header=None,
        names=["Prospect Id", "Job Title", "Persona", "Persona Certainty"],
        usecols=[0, 1, 2, 3],
        dtype={'Prospect Id': str, 'Job Title': str, 'Persona': str, 'Persona Certainty': str},
        on_bad_lines='warn',
    )
except Exception as e:
    print(f"Error processing CSV data: {e}")
    formatted_results = pd.DataFrame()  # In case of error, use empty DataFrame

# Define valid personas
valid_personas = [
    'Executive Sponsor', 'Economic Buyer', 'Data Product Manager/Owner',
    'Data User', 'Application Developer', 'Real-time Specialist',
    'Operator/Systems Administrator', 'Technical Decision Maker', 'Not a target'
]

# Merge the original prospects with the enrichment results using a left merge
merged_df = pd.merge(df_filtered, formatted_results, on="Prospect Id", how="left")

# Determine why each prospect is skipped (if it is skipped)
def determine_skip_reason(row):
    if pd.isna(row['Persona']):
        return "No LLM response"
    elif row['Persona'] not in valid_personas:
        return f"Invalid persona: {row['Persona']}"
    else:
        return None

merged_df["Skip Reason"] = merged_df.apply(determine_skip_reason, axis=1)

# Separate accepted prospects (no skip reason) from skipped ones
final_result = merged_df[merged_df["Skip Reason"].isna()].copy()
skipped_df = merged_df[merged_df["Skip Reason"].notna()].copy()

# Remove duplicate columns if necessary (e.g. duplicate Job Title)
if "Job Title_y" in final_result.columns:
    final_result = final_result.drop(columns="Job Title_y")
    final_result.rename(columns={'Job Title_x': 'Job Title'}, inplace=True)

final_result.drop_duplicates(subset=["Prospect Id"], keep="first", inplace=True)

# Sanitize final results by ensuring valid personas only
final_result = final_result[final_result['Persona'].isin(valid_personas)]

# Print sample output and persona distribution
print(final_result.head())
persona_counts = final_result['Persona'].value_counts(normalize=True) * 100
print("\n========= Persona Distribution =========")
for persona, percentage in persona_counts.items():
    print(f"{persona}: {percentage:.2f}%")

# Provide summary feedback on the processing
num_updated_prospects = len(final_result)
num_skipped_prospects = total_rows - num_updated_prospects
print("\n========= Processing Results =========")
print(f"{num_updated_prospects} prospects updated")
print(f"{num_skipped_prospects} prospects skipped")

# Optionally, print a summary of skip reasons
if not skipped_df.empty:
    skip_reason_counts = skipped_df["Skip Reason"].value_counts()
    print("\n========= Skipped Prospects Reasons =========")
    for reason, count in skip_reason_counts.items():
        print(f"{reason}: {count}")

# ===== Optional checkpoint saves to mitigate restarts =====
checkpoint_ts = datetime.now().strftime("%Y-%m-%d %H %M %S")
try:
    tmp_dir = "/Users/Jaime/Documents/Classified Persona Output/_checkpoints"
    os.makedirs(tmp_dir, exist_ok=True)
    final_result.to_csv(os.path.join(tmp_dir, f"accepted_checkpoint_{checkpoint_ts}.csv"), index=False)
    skipped_df.to_csv(os.path.join(tmp_dir, f"skipped_checkpoint_{checkpoint_ts}.csv"), index=False)
    print(f"Checkpoint written ({checkpoint_ts}).")
except Exception as _e:
    print(f"Checkpoint save failed: {_e}")

# Define output paths for accepted and skipped prospects
save_path = "/Users/Jaime/Documents/Classified Persona Output"  # Adjust for your OS
save_path_skipped = "/Users/Jaime/Documents/Classified Persona Output/Skipped prospects"  # Adjust for your OS

output_filename = os.path.join(save_path, datetime.now().strftime("Personas %Y-%m-%d %H %M %S.csv"))
skipped_filename = os.path.join(save_path_skipped, datetime.now().strftime("Skipped prospects %Y-%m-%d %H %M %S.csv"))

# Save the accepted prospects to a CSV file
final_result.to_csv(output_filename, index=False)
# Save the skipped prospects along with their skip reasons to a separate CSV file
skipped_df.to_csv(skipped_filename, index=False)

print("\n")
print(f"Accepted output written to {output_filename}")
print(f"Skipped prospects (with reasons) written to {skipped_filename}")