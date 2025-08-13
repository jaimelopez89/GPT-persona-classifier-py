import re
import io
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
    model="gpt-4o-mini"  # or "gpt-3.5-turbo-16k", "gpt-4", etc.
)

# Process the prospects in chunks for enrichment
CHUNK_SIZE = 150  # Adjust as needed
total_rows = len(df_filtered)
chunks = [df_filtered[i:i+CHUNK_SIZE] for i in range(0, total_rows, CHUNK_SIZE)]
print("Number of iterations: ", len(chunks))

# Collect responses from the LLM API
results = []
for chunk in tqdm(chunks):
    # Clean Job Title values
    chunk.loc[:, 'Job Title'] = chunk['Job Title'].apply(lambda x: re.sub(",", " ", x))
    # Prepare data in the required format for the API call
    job_titles_table = "\n".join([f"{row['Prospect Id']},{row['Job Title']}" for index, row in chunk.iterrows()])
    response = ask_chat_session(
        session=session,
        user_message=job_titles_table
    )
    if response:
        results.append(response)

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