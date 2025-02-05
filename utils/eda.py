import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_rfm_distributions(rfm_df, output_path='outputs/eda_visualizations/rfm_distributions.png'):
    """
    Plot the distributions of Recency, Frequency, and Monetary values from the RFM data.
    
    Parameters:
        rfm_df (DataFrame): DataFrame containing 'Recency', 'Frequency', and 'MonetaryValue' columns.
        output_path (str): Path to save the generated plot.
    """
    if rfm_df.empty:
        print("RFM Data is empty. Cannot generate distribution plots.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        plt.figure(figsize=(15, 5))

        # Recency distribution
        plt.subplot(1, 3, 1)
        sns.histplot(rfm_df['Recency'], bins=30, kde=True)
        plt.title('Recency Distribution')

        # Frequency distribution
        plt.subplot(1, 3, 2)
        sns.histplot(rfm_df['Frequency'], bins=30, kde=True)
        plt.title('Frequency Distribution')

        # Monetary value distribution
        plt.subplot(1, 3, 3)
        sns.histplot(rfm_df['MonetaryValue'], bins=30, kde=True)
        plt.title('Monetary Value Distribution')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.show()

        print(f"RFM Distributions saved to: {output_path}")

    except Exception as e:
        print(f"Error while generating RFM distributions: {e}")