import matplotlib.pyplot as plt
import seaborn as sns

def plot_rfm_distributions(rfm_df):
    """Plot distributions of Recency, Frequency, and Monetary Value."""
    plt.figure(figsize=(15, 5))

    # Recency Distribution
    plt.subplot(1, 3, 1)
    sns.histplot(rfm_df['Recency'], bins=30, kde=True)
    plt.title('Recency Distribution')
    
    # Frequency Distribution
    plt.subplot(1, 3, 2)
    sns.histplot(rfm_df['Frequency'], bins=30, kde=True)
    plt.title('Frequency Distribution')
    
    # Monetary Value Distribution
    plt.subplot(1, 3, 3)
    sns.histplot(rfm_df['MonetaryValue'], bins=30, kde=True)
    plt.title('Monetary Value Distribution')
    
    plt.tight_layout()
    plt.savefig('outputs/eda_visualizations/rfm_distributions.png')
    plt.show()