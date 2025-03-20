import pandas as pd
import numpy as np
import re
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Consistent variable naming
FILE_PATH = "users.csv"

def validate_email(email):
    """Validate email format"""
    if not isinstance(email, str):
        return False
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

try:
    # Load the file
    df = pd.read_csv(FILE_PATH)
    logger.info(f"File loaded successfully: {FILE_PATH}")
    
    # Remove duplicates
    rows_before = len(df)
    df.drop_duplicates(inplace=True)
    removed = rows_before - len(df)
    logger.info(f"Removed {removed} duplicates")
    
    # Clean age column - more efficient approach using vectorized operations
    if 'age' in df.columns:
        # Replace invalid ages with NaN
        df.loc[(df['age'] < 0) | (df['age'] > 120), 'age'] = np.nan
        # Fill NaNs with median
        median_age = df['age'].median()
        df['age'].fillna(value=median_age, inplace=True)
        logger.info(f"Cleaned age column and filled NaNs with median: {median_age}")
    else:
        logger.warning("No age column found in the dataset")
    
    # Email validation - more efficient with vectorized operations
    if 'email' in df.columns:
        # Store original count for logging
        original_count = len(df)
        
        # Apply validation function to all emails
        df['valid_email'] = df['email'].apply(validate_email)
        df = df[df['valid_email']].drop(columns=['valid_email'])
        
        # Replace empty emails
        df['email'] = df['email'].replace('', 'unknown@example.com')
        
        logger.info(f"Valid emails: {len(df)} out of {original_count}")
    else:
        logger.warning("No email column found in the dataset")
    
    # Country filtering - more efficient approach
    if 'country' in df.columns:
        valid_countries = ['USA', 'India', 'Canada', 'UK', 'Australia']
        original_count = len(df)
        df = df[df['country'].isin(valid_countries)]
        logger.info(f"Valid countries: {len(df)} out of {original_count}")
    else:
        logger.warning("No country column found in the dataset")
    
    # Export cleaned data
    OUTPUT_PATH = "cleaned_users.csv"
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"File saved successfully: {OUTPUT_PATH}")
    
except FileNotFoundError:
    logger.error(f"Could not find the file: {FILE_PATH}")
    exit(1)
except pd.errors.EmptyDataError:
    logger.error(f"The file {FILE_PATH} is empty")
    exit(1)
except pd.errors.ParserError:
    logger.error(f"Error parsing the file: {FILE_PATH}")
    exit(1)
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}")
    exit(1)