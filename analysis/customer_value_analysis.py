import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_clv(simulation_file, rfm_file):
    """
    Calculate Customer Lifetime Value (CLV) based on simulated purchases and RFM data.

    Parameters:
        simulation_file (str): Path to the Monte Carlo simulation output CSV.
        rfm_file (str): Path to the RFM data CSV.
    
    Returns:
        clv_data (DataFrame): DataFrame with CLV and customer segmentation.
    """
    import numpy as np
    import pandas as pd

    # Load simulation and RFM data
    simulation_results = pd.read_csv(simulation_file)
    rfm_data = pd.read_csv(rfm_file)

    # Aggregate total purchases per customer across all simulations
    total_purchases_per_customer = (
        simulation_results.groupby('CustomerID')['PurchasesToday'].sum().reset_index()
    )

    # Calculate CLV: Multiply total purchases by average monetary value from RFM
    clv_data = pd.merge(rfm_data, total_purchases_per_customer, on='CustomerID', how='inner')
    clv_data['CLV'] = clv_data['PurchasesToday'] * clv_data['MonetaryValue']

    # Separate customers with zero CLV
    clv_zero = clv_data[clv_data['CLV'] == 0].copy()
    clv_positive = clv_data[clv_data['CLV'] > 0].copy()

    # Debugging CLV values
    print("CLV Value Counts:")
    print(clv_data['CLV'].value_counts().head(10))
    print(f"Number of unique CLV values: {clv_data['CLV'].nunique()}")

    # Only apply binning to positive CLV values
    if not clv_positive.empty:
        q33 = clv_positive['CLV'].quantile(0.33)
        q66 = clv_positive['CLV'].quantile(0.66)
        max_clv = clv_positive['CLV'].max()

        # Ensure bins are unique
        bins = [0, q33, q66, max_clv]
        labels = ['Low Predictive Buyers', 'Moderate Buyers', 'High-Value Customers']

        # Ensure unique bins
        if len(np.unique(bins)) == len(bins):
            clv_positive['Segment'] = pd.cut(clv_positive['CLV'], bins=bins, labels=labels, include_lowest=True)
        else:
            print("Warning: Non-unique positive CLV bins detected. Adjusting to fewer segments.")
            clv_positive['Segment'] = 'Unsegmented'
    else:
        print("Warning: No positive CLV values found.")
        clv_positive['Segment'] = 'Unsegmented'

    # Assign 'No Purchases' to customers with zero CLV
    clv_zero['Segment'] = 'No Purchases'

    # Combine back the datasets
    clv_data = pd.concat([clv_zero, clv_positive]).sort_values('CustomerID').reset_index(drop=True)

    return clv_data


def plot_customer_segments(clv_data):
    """
    Plot the number of customers in each segment (High-Value, Moderate, Low Predictive Buyers).

    Parameters:
        clv_data (DataFrame): DataFrame containing customer segmentation.
    """
    # Count customers in each segment
    segment_counts = clv_data['Segment'].value_counts().sort_index()

    # Plot the customer segments
    plt.figure(figsize=(8, 6))
    segment_counts.plot(kind='bar', color=['green', 'orange', 'red'], alpha=0.7)
    
    plt.title('Number of Customers by Segment')
    plt.xlabel('Customer Segment')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.savefig('outputs/eda_visualizations/customer_segments_distribution.png')
    plt.show()

def plot_clv_separate_boxplots(clv_data):
    """
    Plot separate boxplots of CLV for each customer segment to handle different scales.

    Parameters:
        clv_data (DataFrame): DataFrame containing customer segmentation and CLV.
    """
    segments = clv_data['Segment'].unique()

    for segment in segments:
        segment_data = clv_data[clv_data['Segment'] == segment]

        plt.figure(figsize=(6, 4))
        plt.boxplot(segment_data['CLV'], patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red'))

        plt.title(f'CLV Distribution: {segment}')
        plt.ylabel('Customer Lifetime Value (CLV)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save each segment's box plot
        filename = f'outputs/eda_visualizations/clv_boxplot_{segment.replace(" ", "_").lower()}.png'
        plt.savefig(filename)
        plt.show()

def plot_clv_boxplot(clv_data):
    """
    Plot a boxplot of CLV distributions across customer segments.

    Parameters:
        clv_data (DataFrame): DataFrame containing customer segmentation and CLV.
    """
    plt.figure(figsize=(10, 6))
    clv_data.boxplot(column='CLV', by='Segment', grid=False, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', color='blue'),
                     medianprops=dict(color='red'))

    plt.title('CLV Distribution by Customer Segment')
    plt.suptitle('')  # Suppress the default title
    plt.xlabel('Customer Segment')
    plt.ylabel('Customer Lifetime Value (CLV)')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig('outputs/eda_visualizations/clv_boxplot_by_segment.png')
    plt.show()

def plot_purchase_trends(daily_simulation_file):
    """
    Plot total predicted purchases per day and cumulative purchases over time using actual simulated events.

    Parameters:
        daily_simulation_file (str): Path to the iterative daily purchase simulation results CSV.
    """
    # Load daily simulation results
    daily_purchases_df = pd.read_csv(daily_simulation_file)

    # Aggregate total purchases per day across all simulations and customers
    daily_trends = daily_purchases_df.groupby('Day')['PurchasesToday'].sum().reset_index()

    # Plot total predicted purchases per day
    plt.figure(figsize=(12, 6))
    plt.plot(daily_trends['Day'], daily_trends['PurchasesToday'], marker='o', linestyle='-', color='blue')
    plt.title('Total Predicted Purchases per Day Over 180 Days')
    plt.xlabel('Day')
    plt.ylabel('Number of Purchases')
    plt.grid(True)
    plt.savefig('outputs/eda_visualizations/daily_purchase_trends_actual.png')
    plt.show()

    # Plot cumulative purchases over time
    daily_trends['CumulativePurchases'] = daily_trends['PurchasesToday'].cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(daily_trends['Day'], daily_trends['CumulativePurchases'], marker='o', linestyle='-', color='green')
    plt.title('Cumulative Predicted Purchases Over 180 Days')
    plt.xlabel('Day')
    plt.ylabel('Cumulative Purchases')
    plt.grid(True)
    plt.savefig('outputs/eda_visualizations/cumulative_purchase_trends_actual.png')
    plt.show()