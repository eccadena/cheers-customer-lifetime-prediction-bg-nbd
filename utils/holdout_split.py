import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import os

def calculate_rfm(data, reference_date):
    """Calculate RFM metrics for the provided dataset."""
    rfm = data.groupby('CustomerID').agg({
        'PurchaseDate': [
            lambda x: (x.max() - x.min()).days,  # Recency
            lambda x: (reference_date - x.max()).days  # T (customer age)
        ],
        'CustomerID': 'count',  # Frequency
        'MonetaryValue': 'mean'  # Average monetary value
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'T', 'Frequency', 'MonetaryValue']
    rfm['Frequency'] = rfm['Frequency'] - 1  # Adjust frequency for repeat purchases
    return rfm

def stratified_holdout_split(transactions_df, holdout_fraction=0.2):
    """
    Splits the dataset into calibration and holdout sets using stratified sampling 
    based on frequency bins to ensure balanced RFM characteristics.
    """
    # Ensure PurchaseDate is in datetime format
    transactions_df['PurchaseDate'] = pd.to_datetime(transactions_df['PurchaseDate'])
    
    # Reference date for RFM calculation
    latest_date = transactions_df['PurchaseDate'].max()
    reference_date = latest_date + timedelta(days=1)

    # Calculate RFM for the full dataset
    full_rfm = calculate_rfm(transactions_df, reference_date)
    
    # Create frequency bins (Low, Medium, High) for stratification
    full_rfm['FrequencyBin'] = pd.qcut(full_rfm['Frequency'], q=3, labels=['Low', 'Medium', 'High'])

    # Stratified sampling: Select holdout customers from each frequency bin
    holdout_customers = full_rfm.groupby('FrequencyBin', group_keys=False).apply(
        lambda x: x.sample(frac=holdout_fraction, random_state=42)
    )['CustomerID']

    # Split transactions into calibration and holdout sets
    calibration_df = transactions_df[~transactions_df['CustomerID'].isin(holdout_customers)]
    holdout_df = transactions_df[transactions_df['CustomerID'].isin(holdout_customers)]

    # Recalculate RFM for both sets
    calibration_rfm = calculate_rfm(calibration_df, reference_date)
    holdout_rfm = calculate_rfm(holdout_df, reference_date)

    # Plot RFM distributions for verification
    plot_rfm_distributions(calibration_rfm, holdout_rfm)

    return calibration_df, holdout_df

def plot_rfm_distributions(calibration_rfm, holdout_rfm):
    """Plot RFM distributions to verify similarity between calibration and holdout sets."""
    
    # Create output directory if it doesn't exist
    output_dir = 'outputs/eda_visualizations'
    os.makedirs(output_dir, exist_ok=True)

    # Plotting
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Recency Distribution
    sns.histplot(calibration_rfm['Recency'], bins=30, kde=True, color='blue', ax=axes[0, 0])
    axes[0, 0].set_title('Calibration Recency Distribution')
    sns.histplot(holdout_rfm['Recency'], bins=30, kde=True, color='orange', ax=axes[0, 1])
    axes[0, 1].set_title('Holdout Recency Distribution')

    # Frequency Distribution
    sns.histplot(calibration_rfm['Frequency'], bins=30, kde=True, color='blue', ax=axes[1, 0])
    axes[1, 0].set_title('Calibration Frequency Distribution')
    sns.histplot(holdout_rfm['Frequency'], bins=30, kde=True, color='orange', ax=axes[1, 1])
    axes[1, 1].set_title('Holdout Frequency Distribution')

    # Monetary Value Distribution
    sns.histplot(calibration_rfm['MonetaryValue'], bins=30, kde=True, color='blue', ax=axes[2, 0])
    axes[2, 0].set_title('Calibration Monetary Value Distribution')
    sns.histplot(holdout_rfm['MonetaryValue'], bins=30, kde=True, color='orange', ax=axes[2, 1])
    axes[2, 1].set_title('Holdout Monetary Value Distribution')

    plt.tight_layout()
    
    # Save the figure
    plot_filename = os.path.join(output_dir, 'rfm_distributions_comparison.png')
    plt.savefig(plot_filename)
    
    # Display the plot
    plt.show()

    print(f"RFM distribution plots saved to: {plot_filename}")