# Purpose
Portfolio building - this is a sample project similar to what was done at Cheers Health inc in 2020.

# **Customer Lifetime Prediction with BG/NBD**

This repository is designed to **showcase my machine learning skills** by modeling and forecasting customer purchasing behavior using the **BG/NBD (Beta-Geometric/Negative Binomial Distribution)** model. It simulates real-world client solutions, focusing on practical implementation and deployment workflows **without incurring cloud service costs**.

## **Purpose of This Repository**

As part of my preparation for **Machine Learning and Data Science interviews**, this project serves as a hands-on demonstration of my ability to:
- **Model customer behavior** using established statistical techniques.
- **Forecast future purchases** through Monte Carlo simulations.
- **Simulate ML workflows** similar to cloud services (like AWS SageMaker) using a modular, local setup. This project was originally deployed on AWS SageMaker (Azure ML Studio or Databricks env would be similar).
- **Deploy and manage models** in a scalable manner, showcasing an understanding of both the technical and operational aspects of machine learning.

While my work with clients is confidential, this project represents the **same level of complexity and structure** I apply in real-world scenarios, offering a transparent look into how I approach data science problems from start to finish.

## **Key Features**
- **Customer Purchase Modeling:** Predict customer transactions using the BG/NBD model.
- **Monte Carlo Simulations:** Generate future purchase scenarios for robust forecasting.
- **Simulated Deployment Workflow:** Mimic cloud service workflows with local modular scripts.
- **Visualization & Evaluation:** Analyze model performance and insights with visual outputs.

## **Technologies Used**
- **Python:** Data manipulation, modeling, and simulations.
- **Pandas, NumPy:** Data processing and analysis.
- **Lifetimes:** BG/NBD model implementation.
- **Matplotlib, Seaborn:** Visualization of results.

## **Key Questions for the Analysis**

### **Customer Behavior Insights**
- How often do customers make purchases?
- What proportion of customers churn after a few purchases versus remaining active?
- What is the average time between purchases?

### **Revenue and Purchase Trends**
- How much revenue do frequent buyers contribute compared to infrequent buyers?
- Are there any seasonal trends in purchasing behavior?

### **Churn Analysis**
- What percentage of customers are likely to churn based on historical behavior?
- How soon after their first purchase do customers typically stop buying?

### **Forecasting Future Purchases**
- How many purchases can we expect from each customer in the next 6-12 months?
- Which customers are most likely to make repeat purchases in the future?

### **Model Performance**
- How accurate is the BG/NBD model in predicting future transactions?
- How well does the model generalize to the holdout data?