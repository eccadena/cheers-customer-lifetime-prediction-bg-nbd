import pandas as pd
import numpy as np
from datetime import timedelta, datetime

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
num_customers = 1000
start_date = datetime(2021, 1, 1)
end_date = datetime(2022, 12, 31)
total_days = (end_date - start_date).days

# Initialize list to store transactions
transactions = []

# Generate synthetic data
for customer_id in range(1, num_customers + 1):
    # Determine number of purchases (0 to 50)
    num_purchases = np.random.poisson(lam=10)
    num_purchases = min(max(num_purchases, 0), 50)
    
    # Churn probability (Beta distribution)
    churn_prob = np.random.beta(a=2, b=5)
    
    purchase_dates = []
    current_date = start_date
    
    for i in range(num_purchases):
        if np.random.rand() < churn_prob and i > 0:
            break  # Customer churned
        
        # Generate next purchase date
        days_until_next_purchase = np.random.poisson(lam=30)
        current_date += timedelta(days=days_until_next_purchase)
        
        if current_date > end_date:
            break  # Stop if date exceeds 2 years
        
        purchase_dates.append(current_date)
    
    # Add transactions to the list
    for purchase_date in purchase_dates:
        monetary_value = round(np.random.uniform(10, 500), 2)
        transactions.append([customer_id, purchase_date, monetary_value])

# Create DataFrame
transactions_df = pd.DataFrame(transactions, columns=["CustomerID", "PurchaseDate", "MonetaryValue"])

# Save as CSV
transactions_df.to_csv('synthetic_customer_transactions.csv', index=False)

# Save as Excel
transactions_df.to_excel('synthetic_customer_transactions.xlsx', index=False)