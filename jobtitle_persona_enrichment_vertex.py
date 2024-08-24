import pandas as pd
import re
import io
import os.path
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
from ask_chatgpt import *
from pypardot.client import PardotAPI
import google.generativeai as genai
import vertexai
from vertexai.generative_models import GenerativeModel



load_dotenv()

project_id = os.getenv("VERTEX_PROJECT_ID")

vertexai.init(project=project_id, location="us-central1")

model = GenerativeModel(model_name="gemini-1.5-flash-001")


# This is only useful to run with non-workplace Google configurations
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# # The Gemini 1.5 models are versatile and work with both text-only and multimodal prompts
# model = genai.GenerativeModel('gemini-1.5-flash')



# Initialize Pardot API wrapper
# p = PardotAPI(version=4)
# p.setup_salesforce_auth_keys()


# Function to elect only rows that contain nonaiveners and no test emails
def filter_emails(df, column_name):
    df = df[~df[column_name].str.contains("@aiven|test", regex=True)]
    return df

# Load the CSV file
path = input("Input the absolute path of the input file with prospects and no persona: ")

# Ensure the path string is treated correctly (removing quotes if any)
path = path.replace('"', '')

# Read the CSV file into a DataFrame
df = pd.read_csv(path, dtype={'First Name':str, 'Last Name':str, 'Email':str, 'Company':str, 'Job Title':str, 'Prospect Id':str})

# Filter out emails from Aiven and test emails
df_filtered = filter_emails(df, 'Email')

# Filter rows where 'Job.Title' is not empty
df_filtered = df_filtered[df_filtered['Job Title'].notna()]

# Keep only useful columns and drop the rest
cols_to_keep = ["Prospect Id","Email","Job Title"]

df_filtered = df_filtered[cols_to_keep]

# Persona definition to pass to the API

definition = """You are an assistant with a strong machine learning capability who understands multiple languages including Japanese, and designed to efficiently categorize job titles into one of four distinct customer personas using a sophisticated machine learning approach.
You leverage techniques like fuzzy matching and similarity search to analyze job titles, focusing on attributes such as industry knowledge, required skills, and typical responsibilities.
You operate by receiving a 2-column table: Prospect ID, Job title. You always return data in a comma-separated, four-column table: Prospect ID, Job title, Persona, Persona Certainty.
The Prospect ID for each output row must be the same one that was fed as input. This is crucial.
Persona Certainty must be a number from 0 to 1 with 2 decimals.
The classification is based on the initial input only, without requiring further interaction.
The output columns must be comma-separated and have no leading or trailing symbols. No context will be provided for the output.

The five available personas are:
- Executive Sponsor: Hold the highest strategic roles in a company. Responsible for the creation of products/services that support the company's strategy and vision and meet the customer needs.
Typically a C-suite individual who has the final approval of the purchase, and has the ability to say No. They care about the overall financial health of the business and the general risk exposure including security and compliance. 
In charge of the cloud and open source strategy of the company. Their titles often contain Founder, Owner, Chief, President, Vice President or Officer, also abbreviated as two- or three-letter acronyms like CEO, CTO, SVP, CISO, VP, CIO, CITO etc.

- Economic buyer: The Economic Buyer oversees the budget, funding the initiative or project and is responsible for ensuring successful business outcomes. They are responsible for making sound financial decisions that maximize return on investment (ROI) and minimize total cost of ownership (TCO), while also carefully assessing and mitigating potential risks for security and compliance.
Typical titles include Senior Vice President, Vice President, Senior Director, Director, Head of Platform/Engineering/Site Reliability Engineering/Data/Analytics/Databases/DevOps/Product

- Technical Decision Maker: They determine the decision criteria that a solution will be assessed against. They create technical solution strategies within their team and/or organization and will have broad technical expertise.   Functionality of the tech (compatibility, complexity, and sustainability of the solution) is key in their evaluation for new products and tooling. 
Typical titles include Senior Principal, Principal, Senior and one of the following: Architect, Engineer, Solution Architecture, Site Reliability Engineering, DevOps, Product, Software Development

- Technical User: Involved in the research, usage, testing, evaluation, implementation, and/or migration of products/solutions. Typically they are the day to day users. Ease of use and automation to enable more building versus maintaining is important for their role. Common titles include Developer, Site Reliability Engineer, Solution Architect, Software Development Engineer, Software Development Manager, DevOps Engineer, Product Manager.

- Business Beneficiary: Business Beneficiaries gain a business benefit from the implementation of the solution. They care about the outcome for their customers, creating efficiency for the business, and look to drive value in their day to day through data. They typically work in departments like Marketing, Sales, People, Finance, Customer Success, Customer Support, Operations, Procurement, Legal

Job titles that do not conform to any of these five classes (e.g. Consultant, Student, Unemployed, and many more) should be classified as Not a target.
On the basis of those definitions, please classify these individuals job titles by whether they refer to an Executive Sponsor, an Economic buyer, a Technical Decision Maker, a Technical User, a Business Beneficiary or Not a target. Only 1 category is possible for each job title."""


# Main logic for processing and enriching data
# chunk_size = 200  # Modify this based on rate limits or for debugging, 200 fits inside current rate limit for OpenAI GPT 3.5

#Beware! This affects the failure/skip rate with Vertex
chunk_size = 500 # This works well with Vertex!

#chunk_size = 500 --> 30% skip rate
#chunk_size = 300 --> 9% skip rate
#chunk_size = 200 --> 6% skip rate
# Ideally we should be able to dynamically adjust the chunk_size so that the total number of skipped prospects would be equal to the 
# chunk_size, so that they could be taken care of with one additional iteration at the end (with some safety margin)
# This is not implemented yet, but could be done in the future

# E.g. chunk_size = 0.9 * (total_prospects * skip_rate_at_that_chunk_size)
#



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
    prompt = definition + job_titles_table
    response = model.generate_content(prompt).text
       
    # Process response and add to results
    results.append(response)

# Define valid personas
valid_personas = ['Executive Sponsor', 'Economic Buyer', 'Technical Decision Maker', 'Technical User', 'Business Beneficiary', 'Not a target']

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

# Save the results that are skipped for audit
skipped_result = df_filtered[~df_filtered['Prospect Id'].isin(final_result['Prospect Id'])]

# Print the first few rows to check
print(final_result.head()) 
print(skipped_result.head()) 


# Define path of dir to save to
save_path = "C:/Users/Jaime/Documents/Marketing analytics/Classified persona output"
save_path_errors = "C:/Users/Jaime/Documents/Marketing analytics/Persona errors"


# Output results to a file with current date and time in the filename. Also setting path for skipped prospects
from datetime import datetime
output_filename = os.path.join(save_path, datetime.now().strftime("Personas %Y-%m-%d %H %M %S.csv"))
output_filename_errors = os.path.join(save_path_errors, datetime.now().strftime("Persona errors %Y-%m-%d %H %M %S.csv"))

# Save files to CSV, omitting indices
final_result.to_csv(output_filename, index=False)
skipped_result.to_csv(output_filename_errors, index=False)

# Provide feedback on how many prospects were enriched and skipped
num_updated_prospects = len(final_result)
num_skipped_prospects = total_rows - num_updated_prospects

print(f"{num_updated_prospects} prospects updated")
print(f"{num_skipped_prospects} prospects skipped")

print(f"Output written to {output_filename}")