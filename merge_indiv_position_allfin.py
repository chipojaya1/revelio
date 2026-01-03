import pandas as pd
import re
from tqdm import tqdm
import os

# File paths
file1 = r"D:\Dropbox\LinkedInData\individual_position\individual_position_all.csv"
file2 = r"D:\Dropbox\LinkedInData\company_ref\company_ref_all.csv"
user_file = r"D:\Dropbox\LinkedInData\individual_user\individual_user_csv\individual_user_all.csv"
user_education_file = r"D:\Dropbox\LinkedInData\individual_user_education\individual_user_education_all.csv"
output_file = r"D:\Dropbox\LinkedInData\final_combined_data.csv"

# Validate file existence
for f in [file1, file2]:
    if not os.path.exists(f):
        print(f"Error: File not found at {f}")
        exit()

# Keywords and regex pattern
keywords = ["Finance", "Insurance", "Bank", "Credit", "Trust", "Investment", 
            "Securities", "Loans", "Lending", "Mortgage", "Portfolio", "Asset", "Risk"]
pattern = re.compile(r'\b(' + '|'.join(keywords) + r')\b', re.IGNORECASE)

# Step 1: Filter company_ref_all.csv
chunksize = 10**6
filtered_chunks = []

print("Filtering company_ref_all.csv...")
try:
    total_rows_company = sum(1 for _ in open(file2, encoding='utf-8'))  # Fix here
except UnicodeDecodeError:
    total_rows_company = sum(1 for _ in open(file2, encoding='ISO-8859-1'))

for chunk in tqdm(pd.read_csv(file2, chunksize=chunksize, encoding='utf-8'), 
                 total=total_rows_company//chunksize):
    chunk['naics_code'] = chunk['naics_code'].astype(str).fillna('')
    chunk_filtered = chunk[
        (chunk['naics_code'].str.startswith('52')) |
        (chunk['company'].str.contains(pattern, na=False))
    ]
    filtered_chunks.append(chunk_filtered)

filtered_company_data = pd.concat(filtered_chunks, ignore_index=True)

# Step 2: Merge with individual_position_all.csv 
print("\nMerging position data...")
try:
    total_rows_individual = sum(1 for _ in open(file1, encoding='utf-8'))  # Fix here
except UnicodeDecodeError:
    total_rows_individual = sum(1 for _ in open(file1, encoding='ISO-8859-1'))

merged_chunks = []
for chunk in tqdm(pd.read_csv(file1, chunksize=chunksize, encoding='utf-8'),
                 total=total_rows_individual//chunksize):
    merged = chunk.merge(filtered_company_data, on='rcid', how='inner')
    merged_chunks.append(merged)

final_merged_data = pd.concat(merged_chunks, ignore_index=True)

# Step 3: Merge with individual_user_all.csv
print("Loading and merging individual_user_all.csv...")
try:
    individual_user_all = pd.read_csv(user_file)
    final_merged_with_users = pd.merge(
        final_merged_data,
        individual_user_all,
        on="user_id",
        how="left",
        suffixes=("_comp", "_user")  # Unique suffixes here
    )
    print("Merged with individual_user_all.")
except FileNotFoundError:
    print(f"Error: {user_file} not found.")
    exit()

# Step 4: Merge with individual_user_education_all.csv
print("Loading and merging individual_user_education_all.csv...")
try:
    individual_user_education_all = pd.read_csv(user_education_file)
    final_combined_data = pd.merge(
        final_merged_with_users,
        individual_user_education_all,
        on="user_id",
        how="left",
        suffixes=("_merged", "_edu")  # New unique suffixes here
    )
    print("Merged with individual_user_education_all.")
except FileNotFoundError:
    print(f"Error: {user_education_file} not found.")
    exit()

# Step 5: Save the final data
final_combined_data.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"Final combined data saved to {output_file}.")
