import pandas as pd
import re
import io
import os.path
from tqdm import tqdm
from pathlib import Path
from ask_chatgpt import *
from ask_gpt_v2 import *
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

# Load frame instructions from external file
frame_instructions_path = "frame_instructions.txt"  # <-- Adjust path if needed
with open(frame_instructions_path, "r", encoding="utf-8") as f:
    frame_instructions = f.read()


# Load persona definitions from external file
persona_definitions_path = "persona_definitions.txt"  # <-- Adjust path if needed
with open(persona_definitions_path, "r", encoding="utf-8") as f:
    persona_definitions = f.read()

# Persona definition to pass to the API
# definition = """You are an assistant with a strong machine learning capability who understands multiple languages including Japanese, and designed to efficiently categorize job titles into one of eight distinct customer personas using a sophisticated machine learning approach.
# You leverage techniques like fuzzy matching and similarity search to analyze job titles, focusing on attributes such as industry knowledge, required skills, and typical responsibilities.
# You operate by receiving a 2-column table: Prospect ID, Job title. You always return data in a comma-separated, four-column table: Prospect ID, Job title, Persona, Persona Certainty.
# The Prospect ID for each output row must be the same one that was fed as input. This is crucial.
# Persona Certainty must be a number from 0 to 1 with 2 decimals.
# The classification is based on the initial input only, without requiring further interaction.
# The output columns must be comma-separated and have no leading or trailing symbols. No context will be provided for the output.

# The eight available personas are:
# - Executive Sponsor: A key decision-maker for smaller organizations. They focus on aligning technology with business goals to drive efficiency, real-time insights, and ROI within the company's overall strategic vision. They prioritize scalability, strategic fit, and competitive advantage when adopting solutions.
# Crucial to securing large software deals. Their titles often contain Founder, Owner, Chief, President, Vice President or Officer, also abbreviated as three- or four-letter acronyms like CEO, CTO, SVP, CISO, VP, CIO, CITO etc. Some other example titles are Chief Digital Officer (CDO), SVP of Corporate Strategy, VP of Innovation & Transformation

# - Economic Buyer: Manages financial resources and ensures that projects deliver the desired business results.  Responsible for making decisions that maximize return on investment (ROI) and minimize total cost of ownership (TCO). Gives the final approval as to whether a software deal/purchase will go ahead or not. Critical to get on your side. Some sample titles may be Chief Financial Officer (CFO), VP of Budget & Procurement, Head of Business Operations, Financial Analyst

# - Data Product Owner/Manager: Responsible to design, build and deliver data products that achieve business goals. They prioritize user needs, scalability, and seamless real-time data integration, acting as the bridge between business and technical teams. Ultimate responsible for the stack and roadmap behind their data product. Can control budget and select vendors. Common titles include Product Owner, Product Manager, Director of Data Solutions, Data Platform & Services Manager, Data & Analytics Product Lead, Product Manager/Owner.

# - Data User: End-users of data products, leveraging them to solve immediate business challenges. Might act as citizen data engineers (who create derivative applications) or analysts (purely consume). They use data to provide fast, actionable insights to meet evolving demands. They work closely with Data Product Managers, using and developing these products to meet specific business needs. Sample titles include Business Intelligence Analyst, Customer Experience Citizen Developer, Financial Risk Analyst

# - Application Developer: Involved in research, usage, testing, evaluation, implementation, and/or migration of business- critical applications. They focus on general application development, not exclusively on stream processing. Prioritize ease of use and seamless integration of new technologies into their workflows, allowing them to focus more on development rather than maintenance. Sample job titles include Developer, Full-Stack Software Engineer, Microservices Developer, Cloud Application Developer, Software Engineer, SRE, SWE, Site Reliability Engineer, etc.

# - Real-time Specialist: Focused on building and maintaining applications specifically designed for real-time data processing. Deeply involved in the operational aspects of stream processing, including scaling applications, ensuring system reliability, and implementing fault- tolerant architectures. Some sample job titles include Streaming Data Engineer, Data Pipeline Architect, Real-Time SRE, Flink Engineer

# - Operator/System Administrator: Focused on ensuring that systems, applications, and networks run smoothly, with a focus on performance, uptime, security, and reliability. Responsible for the day-to-day management, maintenance, and optimization of an organization's IT infrastructure.  Typically not responsible for an application's roadmap. SOme example titles may be IT Infrastructure Manager, Administrator, Cloud & Systems Engineer, Senior Systems Administrator, Cloud Infrastructure Admin.

# - Technical Decision Maker: Defines the technical requirements and standards for solutions and develops technical strategies. They create technical solution strategies within their team and/or organization and will have broad technical expertise. Some example titles may be Director of Platform Engineering, Chief Architect, Head of Enterprise Architecture, Manager, etc.

# Job titles that do not conform to any of these eight classes (e.g. Consultant, Student, Unemployed, and many more) should be classified as Not a target.
# On the basis of those definitions, please classify these individuals job titles by whether they refer to an Executive Sponsor, an Economic Buyer, a Data Product Owner/Manager, a Data User, an Application Developer, a Real-Time Specialist, an Operator/System Administrator, a Technical Decision Maker or Not a target. Only 1 category is possible for each job title."""


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
    
    # Construct the full prompt with 'definition' and job titles table, then call the API
    # prompt = definition + job_titles_table
    # response = ask_chatgpt(prompt)
    
    complete_system_instructions = frame_instructions + persona_definitions
    
    response = ask_gpt_v2(
    system_message = complete_system_instructions,   # The persona definitions and instructions
    user_message=job_titles_table,      # The chunk of data you want classified
    model="gpt-4o-mini"           # Define mdoel to use 
    )
       
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

print(f"{num_updated_prospects} prospects updated")
print(f"{num_skipped_prospects} prospects skipped")

print(f"Output written to {output_filename}")