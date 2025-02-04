import pandas as pd
from lifetimes import BetaGeoFitter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(holdout_rfm, holdout_period=90):
    """
    Evaluate the BG/NBD model using holdout data by reloading saved parameters.
    
    Parameters:
        holdout_rfm (DataFrame): Holdout dataset with 'Frequency', 'Recency', and 'T' columns.
        holdout_period (int): Duration of the holdout period in days.
    """
    # Load model parameters
    params_df = pd.read_csv('outputs/model_parameters.csv', index_col=0)
    params = params_df['parameters'].to_dict()
    
    # Reinitialize the model and fit it with dummy data
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    
    # Fit the model with dummy data to initialize internal structures
    dummy_data = pd.DataFrame({
        'frequency': [0],
        'recency': [0],
        'T': [1]
    })
    bgf.fit(dummy_data['frequency'], dummy_data['recency'], dummy_data['T'])
    
    # Assign the loaded parameters
    bgf.params_ = params
    
    # Now predict expected purchases for the holdout period
    holdout_rfm['PredictedPurchases'] = bgf.predict(
        holdout_period,
        holdout_rfm['Frequency'],
        holdout_rfm['Recency'],
        holdout_rfm['T']
    )

    # Visualize predicted vs actual purchases
    plot_predicted_vs_actual(holdout_rfm)

    # Calculate and print Mean Absolute Error
    mae = (holdout_rfm['Frequency'] - holdout_rfm['PredictedPurchases']).abs().mean()
    print(f"Mean Absolute Error (MAE) of Predictions: {mae:.2f}")

def plot_predicted_vs_actual(holdout_rfm):
    """Plot predicted vs actual purchases for holdout data."""
    plt.figure(figsize=(10, 6))
    sns.histplot(holdout_rfm['PredictedPurchases'], color='blue', label='Predicted', kde=True, stat="density")
    sns.histplot(holdout_rfm['Frequency'], color='orange', label='Actual', kde=True, stat="density")
    plt.legend()
    plt.title('Predicted vs Actual Purchases in Holdout Period')
    plt.xlabel('Number of Purchases')
    plt.ylabel('Density')
    
    # Save plot
    plt.savefig('outputs/eda_visualizations/predicted_vs_actual.png')
    plt.show()
