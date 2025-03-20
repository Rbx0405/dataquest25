import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_subscriptions(df):
    """
    Clean subscription data by removing duplicates, converting dates, removing negative amounts,
    and merging overlapping subscription periods for the same user.
    
    Args:
        df (pandas.DataFrame): Raw subscription data
        
    Returns:
        pandas.DataFrame: Cleaned subscription data
    """
    # Store original row count for logging
    original_count = len(df)
    
    # 1. Remove duplicate rows
    df = df.drop_duplicates()
    logger.info(f"Removed {original_count - len(df)} duplicate rows")
    
    # 2. Convert dates to datetime
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
    
    # Drop rows with invalid dates
    valid_dates_count = len(df)
    df = df.dropna(subset=['start_date', 'end_date'])
    logger.info(f"Removed {valid_dates_count - len(df)} rows with invalid dates")
    
    # 3. Remove subscriptions with negative amounts
    if 'amount_paid' in df.columns:
        negative_count = (df['amount_paid'] < 0).sum()
        df = df[df['amount_paid'] >= 0]
        logger.info(f"Removed {negative_count} subscriptions with negative amounts")
    
    # 4. Merge overlapping subscriptions per user
    logger.info("Merging overlapping subscriptions by user...")
    merged_rows = []
    
    # Sort by user and start date first to optimize the merging process
    df = df.sort_values(['user_id', 'start_date'])
    
    for user_id, group in df.groupby('user_id'):
        user_rows = []
        current = None
        
        for _, row in group.iterrows():
            if current is None:
                current = row.copy()
            elif row['start_date'] <= current['end_date']:
                # Merge overlapping dates
                current['end_date'] = max(current['end_date'], row['end_date'])
                if 'amount_paid' in df.columns:
                    current['amount_paid'] += row['amount_paid']
            else:
                # No overlap, add current to results and start a new current
                user_rows.append(current)
                current = row.copy()
                
        # Don't forget the last subscription period
        if current is not None:
            user_rows.append(current)
            
        merged_rows.extend(user_rows)
    
    cleaned_df = pd.DataFrame(merged_rows)
    logger.info(f"Final dataset contains {len(cleaned_df)} subscription records")
    
    return cleaned_df

def main():
    try:
        # Load subscription data
        input_file = "subscriptions.csv"
        subscriptions = pd.read_csv(input_file, encoding='latin1')
        logger.info(f"Loaded {len(subscriptions)} subscription records from {input_file}")
        
        # Apply cleaning
        cleaned_subscriptions = clean_subscriptions(subscriptions)
        
        # Save cleaned data
        output_file = "subscriptions_cleaned.csv"
        cleaned_subscriptions.to_csv(output_file, index=False)
        logger.info(f"Saved cleaned subscription data to {output_file}")
        
        # Optional: Generate basic statistics
        subscription_duration = (cleaned_subscriptions['end_date'] - 
                                 cleaned_subscriptions['start_date']).dt.days
        
        logger.info(f"Average subscription duration: {subscription_duration.mean():.2f} days")
        if 'amount_paid' in cleaned_subscriptions.columns:
            logger.info(f"Average amount paid: ${cleaned_subscriptions['amount_paid'].mean():.2f}")
        
        return cleaned_subscriptions
        
    except FileNotFoundError:
        logger.error(f"File not found: {input_file}")
    except Exception as e:
        logger.error(f"Error processing subscription data: {str(e)}")

if __name__ == "__main__":
    main()