import pandas as pd
from tqdm import tqdm

# File paths
final_combined_file = r"D:\Dropbox\LinkedInData\final_combined_data.csv"
voter_file = r"F:\L2\combined_voter_data_renamed.csv"
output_file = r"D:\Dropbox\LinkedInData\user_party_match.csv"

# Load and preprocess final_combined_data.csv
final_combined = pd.read_csv(final_combined_file)

# Rename 'ïuser_id' to 'user_id'
final_combined.rename(columns={'ïuser_id': 'user_id'}, inplace=True)

# State name to abbreviation mapping
state_mapping = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
"Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY"
}

# Filter enddate >= 2019
final_combined['enddate_year'] = final_combined['enddate'].astype(str).str[:4]
final_combined = final_combined[final_combined['enddate_year'] != 'nan']
final_combined['enddate_year'] = final_combined['enddate_year'].astype(int)
final_combined = final_combined[final_combined['enddate_year'] >= 2019]

# Convert state names to abbreviations
final_combined['state_abbr'] = final_combined['state'].map(state_mapping)

# Process voter data in chunks
chunksize = 500000
all_matches = []

for chunk in tqdm(pd.read_csv(voter_file, chunksize=chunksize,
                             usecols=['Voters_FirstName', 'Voters_LastName',
                                     'state_abbrev', 'VoterParties_Change_Changed_Party',
                                     'politicalparty', 'LALVOTERID'])):
    # Preprocess voter chunk
    chunk['birth_year'] = pd.to_datetime(chunk['VoterParties_Change_Changed_Party']).dt.year
    
    # Merge data
    merged = pd.merge(
        final_combined,
        chunk,
        left_on=['firstname', 'lastname', 'state_abbr'],
        right_on=['Voters_FirstName', 'Voters_LastName', 'state_abbrev'],
        how='inner',  # Changed to inner join for phase 0
        suffixes=('', '_voter')
    )
    
    # ----- NEW PHASE 0: Party Consensus Without Age Validation -----
    phase0_matches = []
    resolved_users = set()
    
    # Group all merged records
    for user_id, group in merged.groupby('user_id'):
        if group['politicalparty'].nunique() == 1:
            phase0_matches.append(group.iloc[0])
            resolved_users.add(user_id)
    
    if phase0_matches:
        all_matches.append(pd.DataFrame(phase0_matches))
    
    # Filter out resolved users
    remaining = merged[~merged.user_id.isin(resolved_users)]
    
    # Calculate year difference for remaining records
    remaining['year_diff'] = remaining['collegeyear'] - remaining['birth_year']
    remaining = remaining.dropna(subset=['year_diff'])
    
    # ----- Original Age Validation Phases -----
    chunk_matches = []

    # Phase 1: 12-30 years
    phase1 = remaining[(remaining.year_diff >= 12) & (remaining.year_diff <= 30)]
    for user_id, group in phase1.groupby('user_id'):
        if group['politicalparty'].nunique() == 1:
            chunk_matches.append(group.iloc[0])
            resolved_users.add(user_id)

    # Phase 2: 15-25 years (remaining users)
    remaining = remaining[~remaining.user_id.isin(resolved_users)]
    phase2 = remaining[(remaining.year_diff >= 15) & (remaining.year_diff <= 25)]
    for user_id, group in phase2.groupby('user_id'):
        if group['politicalparty'].nunique() == 1:
            chunk_matches.append(group.iloc[0])
            resolved_users.add(user_id)

    # Phase 3: 15-24 years (remaining users)
    remaining = remaining[~remaining.user_id.isin(resolved_users)]
    phase3 = remaining[(remaining.year_diff >= 15) & (remaining.year_diff <= 24)]
    for user_id, group in phase3.groupby('user_id'):
        if group['politicalparty'].nunique() == 1:
            chunk_matches.append(group.iloc[0])
            resolved_users.add(user_id)

    # Phase 4: 15-23 years (remaining users)
    remaining = remaining[~remaining.user_id.isin(resolved_users)]
    phase4 = remaining[(remaining.year_diff >= 15) & (remaining.year_diff <= 23)]
    for user_id, group in phase4.groupby('user_id'):
        if group['politicalparty'].nunique() == 1:
            chunk_matches.append(group.iloc[0])

    if chunk_matches:
        all_matches.append(pd.DataFrame(chunk_matches))

# Combine matches and deduplicate
if all_matches:
    final_matches = pd.concat(all_matches, ignore_index=True)
    final_matches = final_matches.sort_values(['year_diff'], ascending=[False])
    final_output = final_matches.drop_duplicates('user_id', keep='first')
else:
    final_output = pd.DataFrame()

# Preserve unmatched users
final_combined = final_combined.merge(
    final_output,
    on='user_id',
    how='left',
    suffixes=('', '_matched')
)

# Save results
final_combined.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"Final dataset saved to {output_file}")
