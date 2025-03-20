import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_ratings(df: pd.DataFrame, 
                  valid_users: Optional[pd.DataFrame] = None, 
                  valid_movies: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Clean movie ratings data by removing duplicates, filtering for valid users and movies,
    and handling multiple ratings from the same user for the same movie.
    
    Args:
        df (pd.DataFrame): Raw ratings data
        valid_users (pd.DataFrame, optional): DataFrame containing valid user IDs
        valid_movies (pd.DataFrame, optional): DataFrame containing valid movie IDs
        
    Returns:
        pd.DataFrame: Cleaned ratings data
    """
    # Store original count for logging
    original_count = len(df)
    logger.info(f"Starting with {original_count} rating records")
    
    # 1. Remove duplicates
    df = df.drop_duplicates()
    logger.info(f"Removed {original_count - len(df)} duplicate ratings")
    
    # 2. Keep only valid user_id and movie_id if provided
    if valid_users is not None:
        if 'user_id' in valid_users.columns:
            before_count = len(df)
            df = df[df['user_id'].isin(valid_users['user_id'])]
            logger.info(f"Filtered out {before_count - len(df)} ratings from invalid users")
        else:
            logger.warning("Valid users DataFrame doesn't contain 'user_id' column")
    else:
        logger.info("No valid users provided, skipping user filtering")
    
    if valid_movies is not None:
        if 'movie_id' in valid_movies.columns:
            before_count = len(df)
            df = df[df['movie_id'].isin(valid_movies['movie_id'])]
            logger.info(f"Filtered out {before_count - len(df)} ratings for invalid movies")
        else:
            logger.warning("Valid movies DataFrame doesn't contain 'movie_id' column")
    else:
        logger.info("No valid movies provided, skipping movie filtering")
    
    # 3. Remove impossible scores (outside 0-5 range)
    invalid_scores = (~df['rating'].between(0, 5)).sum()
    df = df[df['rating'].between(0, 5)]
    logger.info(f"Removed {invalid_scores} ratings with impossible scores")
    
    # 4. Handle conflicting multiple ratings from the same user for the same movie
    before_count = len(df)
    if 'timestamp' in df.columns:
        # Convert timestamps to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Check for missing timestamps after conversion
        missing_timestamps = df['timestamp'].isna().sum()
        if missing_timestamps > 0:
            logger.warning(f"Found {missing_timestamps} records with invalid timestamps")
            # Drop rows with invalid timestamps to avoid issues in sorting
            df = df.dropna(subset=['timestamp'])
        
        # Keep only the most recent rating
        df = df.sort_values('timestamp').drop_duplicates(
            subset=['user_id', 'movie_id'], keep='last'
        )
        logger.info("Kept only the most recent rating for each user-movie pair")
    else:
        # No timestamp? Take average per user/movie
        df = df.groupby(['user_id', 'movie_id'], as_index=False)['rating'].mean()
        logger.info("Averaged ratings for each user-movie pair (no timestamps available)")
    
    logger.info(f"Resolved {before_count - len(df)} conflicting ratings")
    logger.info(f"Final clean dataset contains {len(df)} ratings")
    
    return df

def main():
    try:
        # Load datasets
        ratings_file = "ratings.csv"
        users_file = "users.csv"  # Assuming these files exist
        movies_file = "movies.csv"
        
        ratings = pd.read_csv(ratings_file, encoding='latin1')
        logger.info(f"Loaded {len(ratings)} rating records from {ratings_file}")
        
        try:
            users = pd.read_csv(users_file, encoding='latin1')
            logger.info(f"Loaded {len(users)} user records from {users_file}")
        except FileNotFoundError:
            logger.warning(f"Users file {users_file} not found. Will clean ratings without user filtering.")
            users = None
        
        try:
            movies = pd.read_csv(movies_file, encoding='latin1')
            logger.info(f"Loaded {len(movies)} movie records from {movies_file}")
        except FileNotFoundError:
            logger.warning(f"Movies file {movies_file} not found. Will clean ratings without movie filtering.")
            movies = None
        
        # Apply cleaning
        cleaned_ratings = clean_ratings(ratings, users, movies)
        
        # Save cleaned data
        output_file = "ratings_cleaned.csv"
        cleaned_ratings.to_csv(output_file, index=False)
        logger.info(f"Saved cleaned ratings data to {output_file}")
        
        # Basic statistics
        logger.info(f"Average rating: {cleaned_ratings['rating'].mean():.2f}")
        logger.info(f"Rating distribution: {cleaned_ratings['rating'].value_counts().sort_index().to_dict()}")
        logger.info(f"Number of unique users: {cleaned_ratings['user_id'].nunique()}")
        logger.info(f"Number of unique movies: {cleaned_ratings['movie_id'].nunique()}")
        
        return cleaned_ratings
        
    except Exception as e:
        logger.error(f"Error processing ratings data: {str(e)}")
        raise

if __name__ == "__main__":
    main()