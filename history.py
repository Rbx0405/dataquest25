import pandas as pd
import sqlite3
import os
import logging
from typing import Set, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Database file (Change if using MySQL/PostgreSQL)
DATABASE_FILE = "watch_history.db"

def create_connection():
    """Create a database connection."""
    conn = sqlite3.connect(DATABASE_FILE)
    return conn

def load_csv_to_db(csv_file: str, table_name: str):
    """Load a CSV file into the SQLite database."""
    conn = create_connection()
    df = pd.read_csv(csv_file)

    # Save DataFrame to the database (replace existing table)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    
    conn.commit()
    conn.close()
    logger.info(f"Loaded {len(df)} records from {csv_file} into {table_name} table.")

def fetch_data_from_db(query: str) -> pd.DataFrame:
    """Fetch data from the database."""
    conn = create_connection()
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def validate_watch_history(known_devices: Set[str]) -> Dict[str, pd.DataFrame]:
    """
    Validate watch history data against database and known devices.
    
    Returns:
        Dictionary containing dataframes with inconsistencies
    """
    results = {}

    # Fetch watch history from database
    viewing_df = fetch_data_from_db("SELECT * FROM watch_history")
    logger.info(f"Loaded {len(viewing_df)} records from database.")

    # Fetch users from database
    database_df = fetch_data_from_db("SELECT user_id FROM users")
    logger.info(f"Loaded {len(database_df)} user records from database.")

    # Check if required columns exist
    required_columns = {'user_id', 'device'}
    missing_columns = required_columns - set(viewing_df.columns)

    if missing_columns:
        logger.error(f"Missing required columns {missing_columns} in watch history data.")
        return results

    # Find unexpected devices (if 'device' column exists)
    if 'device' in viewing_df.columns:
        viewing_df['device'] = viewing_df['device'].fillna('Unknown')
        unexpected_devices = viewing_df[~viewing_df['device'].isin(known_devices)].copy()
        if not unexpected_devices.empty:
            results['unexpected_devices'] = unexpected_devices
            logger.info(f"Found {len(unexpected_devices)} records with unexpected devices.")

    # Find invalid user IDs
    invalid_ids = viewing_df[~viewing_df['user_id'].isin(database_df['user_id'])].copy()
    if not invalid_ids.empty:
        results['invalid_ids'] = invalid_ids
        logger.info(f"Found {len(invalid_ids)} records with invalid user IDs.")

    # Combine inconsistencies
    if results:
        all_inconsistencies = pd.concat(results.values()).drop_duplicates()
        results['all_inconsistencies'] = all_inconsistencies
        logger.info(f"Total of {len(all_inconsistencies)} inconsistencies identified.")

    return results

def save_results(results: Dict[str, pd.DataFrame], output_dir: str = '.') -> None:
    """Save inconsistency results to CSV files."""
    if not results:
        logger.info("No inconsistencies found, nothing to save.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    for name, df in results.items():
        if not df.empty:
            output_file = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(df)} records to {output_file}")

def main():
    """Main function to run validation."""
    viewing_csv = "watch_history.csv"
    users_csv = "users.csv"

    # Load CSV data into the database
    load_csv_to_db(viewing_csv, "watch_history")
    load_csv_to_db(users_csv, "users")

    # Define known devices
    known_devices = {'Smartphone', 'Laptop', 'Tablet', 'Smart TV', 'Desktop'}

    # Validate data
    results = validate_watch_history(known_devices)

    # Save results
    save_results(results)

    # Print summary
    if results:
        unexpected_count = len(results.get('unexpected_devices', pd.DataFrame()))
        invalid_id_count = len(results.get('invalid_ids', pd.DataFrame()))
        total_count = len(results.get('all_inconsistencies', pd.DataFrame()))

        print("\nInconsistencies detected and saved:")
        print(f"- {unexpected_count} unexpected devices found")
        print(f"- {invalid_id_count} records with invalid IDs")
        print(f"- {total_count} total inconsistencies logged")
    else:
        print("\nNo inconsistencies found. Data validation successful.")

if __name__ == "__main__":
    main()