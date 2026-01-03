import os
import zipfile
import pandas as pd
from io import BytesIO
from tqdm import tqdm
import chardet
import logging

# Configuration
MAIN_ZIP_PATH = r"D:\Dropbox\L2 data\2019_uniform.zip"
OUTPUT_CSV = r"F:\L2\combined_voter_data.csv"
ENCODING_FALLBACKS = ['utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
CHUNK_SIZE = 50000

REQUIRED_COLUMNS = [
    'LALVOTERID', 'Voters_FirstName', 'Voters_MiddleName',
    'Voters_LastName', 'Voters_BirthDate', 'Parties_Description',
    'VoterParties_Change_Changed_Party', 'Residence_Addresses_State',
    'Residence_Addresses_Zip', 'County'
]

# Initialize logging
logging.basicConfig(filename='voter_processing.log', level=logging.INFO)

def detect_encoding(file_obj):
    """Enhanced encoding detection with statistical analysis"""
    raw_data = file_obj.read(50000)
    file_obj.seek(0)
    
    try:
        result = chardet.detect(raw_data)
        confidence_threshold = 0.85
        detected_encoding = result['encoding'] if result['confidence'] > confidence_threshold else None
        
        # Statistical analysis for common voter file patterns
        if b'\x00' in raw_data:
            return 'utf-16-le'
        if b'\xa3' in raw_data:  # Pound sign in Latin-1
            return 'latin-1'
            
        return detected_encoding or 'latin-1'
    
    except Exception as e:
        logging.error(f"Encoding detection failed: {str(e)}")
        return 'latin-1'

def process_tab_file(data_file, filename):
    """Robust CSV processing with encoding fallbacks"""
    for encoding in [detect_encoding(data_file)] + ENCODING_FALLBACKS:
        try:
            data_file.seek(0)
            reader = pd.read_csv(
                data_file, sep='\t', usecols=REQUIRED_COLUMNS,
                dtype='string', encoding=encoding,
                engine='python', on_bad_lines='warn',
                chunksize=CHUNK_SIZE, 
                encoding_errors='backslashreplace'
            )
            
            for chunk in reader:
                clean_chunk = (
                    chunk
                    .drop_duplicates('LALVOTERID')
                    .apply(lambda x: x.str.strip() if x.dtype == "object" else x)
                )
                clean_chunk.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
            
            logging.info(f"Successfully processed {filename} with {encoding}")
            return True
            
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            logging.warning(f"Retrying {filename} with {encoding}: {str(e)[:100]}")
            continue
    
    logging.error(f"Failed all encoding attempts for {filename}")
    return False

def process_zip(zip_bytes):
    """ZIP processing with election data validation"""
    with zipfile.ZipFile(BytesIO(zip_bytes)) as nested_zip:
        for data_file_name in nested_zip.namelist():
            if not data_file_name.endswith('.tab'):
                continue
                
            with nested_zip.open(data_file_name) as data_file:
                if process_tab_file(data_file, data_file_name):
                    print(f"✓ Processed {data_file_name}")
                else:
                    print(f"✗ Failed {data_file_name}")

# Initialize output with header
pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(OUTPUT_CSV, index=False)

# Main processing loop
with zipfile.ZipFile(MAIN_ZIP_PATH) as main_zip:
    nested_zips = [f for f in main_zip.namelist() if f.endswith('.zip')]
    
    for zip_name in tqdm(nested_zips, desc="Processing state archives"):
        try:
            with main_zip.open(zip_name) as zip_file:
                process_zip(zip_file.read())
                
        except Exception as e:
            logging.error(f"Critical error in {zip_name}: {str(e)[:200]}")
            print(f"! Critical error in {zip_name} - see logs")

print(f"\nProcessing complete. Output saved to: {OUTPUT_CSV}")
