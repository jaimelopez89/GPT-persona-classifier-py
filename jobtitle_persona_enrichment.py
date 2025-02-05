import pandas as pd
import re
import io
import os.path
from tqdm import tqdm
from pathlib import Path
from ask_chatgpt import *
from ask_gpt_v2 import *
from gpt_functions import *
# from pypardot.client import PardotAPI


# Initialize Pardot API wrapper
# p = PardotAPI(version=4)
# p.setup_salesforce_auth_keys()


# Define LLM to use
# model = "gpt-4o"

# Function to elect only rows that contain non-ververicans and no test emails
def filter_emails(df, column_name):
    df[column_name] = df[column_name].fillna('').astype(str)
    df = df[~df[column_name].str.contains("@ververica|test", regex=True)]
    return df

# Load the CSV file
path = input("Input the absolute path of the input file with prospects and no persona: ")

# Ensure the path string is treated correctly (removing quotes if any)
path = path.replace('"', '')

# Read the CSV file into a DataFrame
df = pd.read_csv(path, dtype={'First Name':str, 'Last Name':str, 'Email':str, 'Company':str, 'Job Title':str, 'Prospect Id':str})

# Filter out emails from Ververica and test emails
df_filtered = filter_emails(df, 'Email')

# Filter rows where 'Job.Title' is not empty
df_filtered = df_filtered[df_filtered['Job Title'].notna()]

# Keep only useful columns and drop the rest
cols_to_keep = ["Prospect Id","Email","Job Title"]

df_filtered = df_filtered[cols_to_keep]

# Load frame instructions from external file. This should set the character and capabilities of the LLM agent
frame_instructions_path = "frame_instructions.txt"  # <-- Adjust path if needed
with open(frame_instructions_path, "r", encoding="utf-8") as f:
    frame_instructions = f.read()


# Load persona definitions from external file. This should describe in detail which personas are allowed, their characteristics and how to assign them.
persona_definitions_path = "persona_definitions.txt"  # <-- Adjust path if needed
with open(persona_definitions_path, "r", encoding="utf-8") as f:
    persona_definitions = f.read()
    
# Build full system instructions by joining the two pieces above
complete_system_instructions = frame_instructions + persona_definitions

# Prepare LLM session (system instructions loaded only once) and store it in the var session
# This also defines the model to use throughout!
# This sets the system message in an internal conversation object
session = create_chat_session(
    system_message=complete_system_instructions, 
    model="gpt-3.5-turbo"  # or "gpt-3.5-turbo-16k", "gpt-4", etc.
)


# Main logic for processing and enriching data
chunk_size = 150  # Modify this based on rate limits or for debugging, 150 fits inside current rate limit
total_rows = len(df_filtered)
chunks = [df_filtered[i:i+chunk_size] for i in range(0, total_rows, chunk_size)]

#Print number of chunks
print("Number of iterations: ", len(chunks))

results = []

for chunk in tqdm(chunks):
    # Clean and prepare data for API call. Using .loc attribute on df slice to replace in-situ
    # chunk['Job Title'] = chunk['Job Title'].apply(lambda x: re.sub(",", " ", x))
    chunk.loc[:, 'Job Title'] = chunk['Job Title'].apply(lambda x: re.sub(",", " ", x))

    
    # Prepare the data in the required format for the API call
    job_titles_table = "\n".join([f"{row['Prospect Id']},{row['Job Title']}" for index, row in chunk.iterrows()])

    # Ask the LLM using the previously created session. Model is defined on session creation
    response = ask_chat_session(
        session=session,
        user_message=job_titles_table
    )

    if response:
        results.append(response)
   
      
    # Process response and add to results
    results.append(response)

# Define valid personas
valid_personas = ['Executive Sponsor', 'Economic Buyer', 'Data Product Owner/Manager', 'Data User', 'Application Developer', 'Real-time Specialist', 'Operator/System Administrator', 'Technical Decision Maker', 'Not a target']

# Filter out None and empty string values from the results list
filtered_results = [result for result in results if result]

# Combine all results and perform any necessary cleaning or formatting
enriched_result = "\n".join(filtered_results)

# Combine all results into a single DataFrame
# formatted_results = pd.read_csv(io.StringIO(enriched_result), header=None, names=["Prospect Id", "Job Title", "Persona", "Persona Certainty"], dtype={'Prospect Id':str, 'Persona Certainty':str, 'Persona':str, 'Job Title':str})

try:
    # Attempt to read without column restrictions
    df_test = pd.read_csv(io.StringIO(enriched_result), header=None,
                                    on_bad_lines='warn')
    
    # Check if there are more than the expected columns
    if df_test.shape[1] > 4:
        print("Warning: Extra columns detected. Only the first four columns will be used.")

    # Proceed with using only the required columns
    formatted_results = pd.read_csv(io.StringIO(enriched_result), 
                                    header=None, 
                                    names=["Prospect Id", "Job Title", "Persona", "Persona Certainty"],
                                    usecols=[0, 1, 2, 3], 
                                    dtype={'Prospect Id': str, 'Job Title': str, 'Persona': str, 'Persona Certainty': str},
                                    on_bad_lines='warn',
                                    )

except Exception as e:
    print(f"Error processing CSV data: {e}")
    # Further error handling or recovery actions



# Test step for visual check
# print(formatted_results.head()) 
# wait = input("Check the head of the formatted results till now. Press Enter to continue.")

# Perform an inner join between formatted_results and df_filtered
final_result = pd.merge(df_filtered, formatted_results, on="Prospect Id", how="inner")

# For debugging & visual inspection
# print(final_result.head()) 
# wait = input("Check the head of the merged output till now. Press Enter to continue.")

# Drop duplicate column for Job Title and rename the original to remove the _x
final_result = final_result.drop(columns="Job Title_y")
final_result.rename(columns={'Job Title_x': 'Job Title'}, inplace=True)

#The merge a few lines above now somehow produces duplicate rows after the merge (didn't happen on previous versions)
# Drop duplicate rows in postprocessing
final_result.drop_duplicates(subset=["Prospect Id"], keep="first", inplace=True)

# Sanitize output and keep only valid rows assigned to one of the five correct personas
final_result = final_result[final_result['Persona'].isin(valid_personas)]

# Print the first few rows to check
print(final_result.head())

# Display percentage counts per persona for validation
persona_counts = final_result['Persona'].value_counts(normalize=True) * 100
print("\n========= Persona Distribution =========")
for persona, percentage in persona_counts.items():
    print(f"{persona}: {percentage:.2f}%")

# Define path of dir to save to
# save_path = "C:/Users/Jaime/Documents/Marketing analytics/Classified persona output" #Windows directory
save_path = "/Users/Jaime/Documents/Classified Persona Output" #Mac directory

# Output results to a file with current date and time in the filename
from datetime import datetime
output_filename = os.path.join(save_path, datetime.now().strftime("Personas %Y-%m-%d %H %M %S.csv"))

# Save file to CSV, omitting indices
final_result.to_csv(output_filename, index=False)

# Provide feedback on how many prospects were enriched and skipped
num_updated_prospects = len(final_result)
num_skipped_prospects = total_rows - num_updated_prospects

print("\n========= Processing Results =========")
print(f"{num_updated_prospects} prospects updated")
print(f"{num_skipped_prospects} prospects skipped")

print(f"Output written to {output_filename}")