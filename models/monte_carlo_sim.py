# models/monte_carlo_sim.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
import sys

def monte_carlo_simulation(rfm_df, days=180, num_simulations=1000):
    """
    Perform Monte Carlo simulations to forecast future purchases with real-time progress updates.
    
    Parameters:
        rfm_df (DataFrame): DataFrame with 'Frequency', 'Recency', and 'T' columns.
        days (int): Number of days to simulate future purchases.
        num_simulations (int): Number of simulation runs.
    
    Returns:
        simulation_results (DataFrame): Simulated total purchases per customer.
    """
    # Load model parameters
    params_df = pd.read_csv('outputs/model_parameters.csv', index_col=0)
    params = params_df['parameters'].to_dict()

    # Reinitialize BG/NBD model and fit dummy data to set internal structures
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    dummy_data = pd.DataFrame({'frequency': [0], 'recency': [0], 'T': [1]})
    bgf.fit(dummy_data['frequency'], dummy_data['recency'], dummy_data['T'])
    bgf.params_ = params

    # Initialize a list to store simulation results
    simulation_list = []

    print(f"Starting Monte Carlo Simulation: Simulating purchases {days} days into the future.")
    print(f"Total Simulations: {num_simulations}\n")

    for i in range(1, num_simulations + 1):
        # Simulate purchases for each customer
        simulated_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(
            days,
            rfm_df['Frequency'],
            rfm_df['Recency'],
            rfm_df['T']
        )
        simulation_list.append(simulated_purchases.rename(f'Simulation_{i}'))

        # Inline progress update
        sys.stdout.write(f'\rSimulation {i}/{num_simulations} in progress...')
        sys.stdout.flush()

    print("\nSimulation complete. Processing results...")

    # Concatenate all simulations at once to avoid fragmentation
    simulation_results = pd.concat(simulation_list, axis=1)
    simulation_results.insert(0, 'CustomerID', rfm_df['CustomerID'])

    # Calculate average simulated purchases
    simulation_results['AverageSimulatedPurchases'] = simulation_results.iloc[:, 1:].mean(axis=1)
    
    # Save simulation results
    simulation_results.to_csv('outputs/purchase_simulations.csv', index=False)
    
    print("Simulation results saved to 'outputs/purchase_simulations.csv'.")
    
    return simulation_results

def plot_simulation_results(simulation_results):
    """Plot the distribution of average simulated purchases."""
    plt.figure(figsize=(10, 6))
    plt.hist(simulation_results['AverageSimulatedPurchases'], bins=30, color='purple', alpha=0.7)
    plt.title('Monte Carlo Simulation: Average Future Purchases')
    plt.xlabel('Average Number of Purchases')
    plt.ylabel('Number of Customers')
    
    # Save plot
    plt.savefig('outputs/eda_visualizations/monte_carlo_simulation.png')
    plt.show()
