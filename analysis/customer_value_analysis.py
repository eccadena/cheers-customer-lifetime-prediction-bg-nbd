import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_clv(simulation_file, rfm_file):
    """
    Calculate Customer Lifetime Value (CLV) by combining predicted purchases with monetary value.

    Parameters:
        simulation_file (str): Path to the Monte Carlo simulation results CSV.
        rfm_file (str): Path to the RFM data CSV.
    
    Returns:
        clv_data (DataFrame): DataFrame containing CLV and customer segments.
    """
    # Load simulation results and RFM data
    simulation_results = pd.read_csv(simulation_file)
    rfm_data = pd.read_csv(rfm_file)

    # Merge simulation results with RFM to calculate CLV
    clv_data = simulation_results.merge(rfm_data[['CustomerID', 'MonetaryValue']], on='CustomerID', how='left')
    clv_data['CLV'] = clv_data['AverageSimulatedPurchases'] * clv_data['MonetaryValue']

    # Segment customers based on predicted purchases
    quantiles = clv_data['AverageSimulatedPurchases'].quantile([0.5, 0.9])
    clv_data['Segment'] = pd.cut(clv_data['AverageSimulatedPurchases'],
                                 bins=[-np.inf, quantiles[0.5], quantiles[0.9], np.inf],
                                 labels=['Low Predictive Buyers', 'Moderate Buyers', 'High-Value Customers'])
    
    # Save CLV data
    clv_data.to_csv('outputs/customer_clv_segments.csv', index=False)
    
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
