'''trouble pickling'''

import pandas as pd
from lifetimes import BetaGeoFitter

def train_bgnbd_model(rfm_df, penalizer_coef=0.01):
    """
    Train the BG/NBD model and save model parameters instead of the entire object.
    
    Parameters:
        rfm_df (DataFrame): DataFrame containing 'Frequency', 'Recency', and 'T' columns.
        penalizer_coef (float): Penalization coefficient to avoid overfitting.
    
    Returns:
        bgf (BetaGeoFitter): Trained BG/NBD model.
    """
    bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)
    bgf.fit(rfm_df['Frequency'], rfm_df['Recency'], rfm_df['T'])
    
    # Save model parameters to a CSV file
    params = pd.Series(bgf.params_, name='parameters')
    params.to_csv('outputs/model_parameters.csv')
    
    print("BG/NBD model trained and parameters saved successfully.")
    return bgf
