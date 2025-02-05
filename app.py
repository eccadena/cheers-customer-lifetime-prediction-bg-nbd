import os
import pandas as pd
from utils.data_loader import load_data, create_rfm
from utils.holdout_split import stratified_holdout_split
from models.train_model import train_bgnbd_model
from models.evaluate_model import evaluate_model
from models.monte_carlo_sim import monte_carlo_simulation
from analysis.customer_value_analysis import calculate_clv, plot_customer_segments, plot_clv_boxplot, plot_clv_separate_boxplots
from analysis.customer_value_analysis import plot_purchase_trends
from utils.eda import plot_rfm_distributions

def main():
    # Load Transaction Data
    transactions_df = load_data('data/synthetic_customer_transactions.csv')
    
    # Generate and Save RFM Data if it doesn't exist
    rfm_file_path = 'data/synthetic_customer_rfm.csv'
    if not os.path.exists(rfm_file_path):
        print("RFM file not found. Generating RFM data...")
        rfm_df = create_rfm(transactions_df)
        rfm_df.to_csv(rfm_file_path, index=False)
    else:
        rfm_df = pd.read_csv(rfm_file_path)
     # Plot RFM distributions
    plot_rfm_distributions(rfm_df)
    
    # Split Data into Calibration and Holdout Sets
    calibration_df, holdout_df = stratified_holdout_split(transactions_df)
    
    # Create RFM for Holdout Set
    holdout_rfm = create_rfm(holdout_df)
    
    # Train BG/NBD Model
    bgf = train_bgnbd_model(rfm_df)
    
    # Evaluate Model on Holdout Data
    evaluate_model(holdout_rfm)
    
    # Monte Carlo Simulation for Future Forecasting (180 Days)
    monte_carlo_simulation(rfm_df, days=180, num_simulations=1000)
    
    # Customer Value Analysis
    clv_data = calculate_clv('outputs/iterative_purchase_simulations.csv', 'data/synthetic_customer_rfm.csv')
    
    # Display Top 10 Customers by CLV
    print("\nTop 10 Customers by CLV:")
    print(clv_data[['CustomerID', 'CLV', 'Segment']].sort_values(by='CLV', ascending=False).head(10))
    
    # Plot Customer Segments
    plot_customer_segments(clv_data)

    # Plot CLV Boxplot
    plot_clv_boxplot(clv_data)

    # Plot Separate CLV Boxplots for Each Segment
    plot_clv_separate_boxplots(clv_data)

    # Plot actual daily purchase trends
    plot_purchase_trends('outputs/iterative_purchase_simulations.csv')

if __name__ == '__main__':
    main()
