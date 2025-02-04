import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
import sys

def monte_carlo_simulation(rfm_df, days=180, num_simulations=1000):
    """
    Perform Monte Carlo simulations to forecast future purchases with enhanced variability.
    
    Parameters:
        rfm_df (DataFrame): DataFrame with 'Frequency', 'Recency', and 'T' columns.
        days (int): Number of days to simulate future purchases.
        num_simulations (int): Number of simulation runs.
    
    Returns:
        daily_simulation_results (DataFrame): Simulated daily purchases for all customers.
    """
    # Load model parameters
    params_df = pd.read_csv('outputs/model_parameters.csv', index_col=0)
    params = params_df['parameters'].to_dict()

    # Reinitialize BG/NBD model
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    dummy_data = pd.DataFrame({'frequency': [0], 'recency': [0], 'T': [1]})
    bgf.fit(dummy_data['frequency'], dummy_data['recency'], dummy_data['T'])
    bgf.params_ = params

    print(f"Starting Monte Carlo Simulation: Simulating purchases {days} days into the future.")
    print(f"Total Simulations: {num_simulations}\n")

    # Initialize DataFrame to store daily purchases
    all_simulations = []

    for sim in range(1, num_simulations + 1):
        # Simulate expected total purchases for each customer over the period
        total_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(
            days,
            rfm_df['Frequency'],
            rfm_df['Recency'],
            rfm_df['T']
        )

        # For each customer, simulate when purchases occur using Poisson distribution with added variability
        daily_purchases = []
        for customer_id, expected_purchases in zip(rfm_df['CustomerID'], total_purchases):
            # Introduce variability with a Beta distribution scaling factor
            variability_factor = np.random.beta(a=2, b=5)  # Skews towards lower values but allows for bursts
            adjusted_lambda = (expected_purchases / days) * (1 + variability_factor)

            # Simulate number of purchases per day using Poisson process with noise
            purchases_per_day = np.random.poisson(lam=adjusted_lambda, size=days)

            customer_data = pd.DataFrame({
                'CustomerID': customer_id,
                'Day': np.arange(1, days + 1),
                'PurchasesToday': purchases_per_day
            })
            daily_purchases.append(customer_data)

        # Concatenate all customers' data for this simulation
        simulation_df = pd.concat(daily_purchases, ignore_index=True)
        simulation_df['Simulation'] = sim
        all_simulations.append(simulation_df)

        # Inline progress update
        sys.stdout.write(f'\rSimulation {sim}/{num_simulations} in progress...')
        sys.stdout.flush()

    print("\nSimulation complete. Processing results...")

    # Combine all simulation results into one DataFrame
    daily_simulation_results = pd.concat(all_simulations, ignore_index=True)

    # Save daily simulation results
    daily_simulation_results.to_csv('outputs/iterative_purchase_simulations.csv', index=False)

    print("Daily simulation results saved to 'outputs/iterative_purchase_simulations.csv'.")

    return daily_simulation_results

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
