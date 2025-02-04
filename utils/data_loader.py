import pandas as pd
from datetime import timedelta

def load_data(filepath):
    """Load transaction data from a CSV file."""
    return pd.read_csv(filepath, parse_dates=['PurchaseDate'])

def create_rfm(transactions_df, reference_date=None):
    """Create Customer Frequency Matrix (RFM) for BG/NBD modeling."""
    
    # Set reference date as the day after the last purchase if not provided
    if reference_date is None:
        reference_date = transactions_df['PurchaseDate'].max() + timedelta(days=1)
    
    rfm = transactions_df.groupby('CustomerID').agg({
        'PurchaseDate': [
            lambda x: (x.max() - x.min()).days,  # Recency: Last purchase - first purchase
            lambda x: (reference_date - x.min()).days  # T: End of observation period - first purchase
        ],
        'CustomerID': 'count',  # Frequency: Total purchases
        'MonetaryValue': 'mean'  # Average purchase value
    }).reset_index()
    
    # Rename columns
    rfm.columns = ['CustomerID', 'Recency', 'T', 'Frequency', 'MonetaryValue']
    
    # Adjust Frequency to reflect repeat purchases
    rfm['Frequency'] = rfm['Frequency'] - 1
    
    # Ensure Recency is not greater than T
    rfm = rfm[rfm['Recency'] <= rfm['T']]
    
    # Filter out customers with no repeat purchases
    rfm = rfm[rfm['Frequency'] > 0]
    
    return rfm
