📌 Project Overview
Customer churn occurs when customers stop doing business with a company. For telecommunications companies, retaining existing customers is often far more cost-effective than acquiring new ones.

This project analyzes a dataset of 7,043 customers to identify the key behavioral and demographic drivers of churn. By understanding these patterns, the business can implement targeted retention strategies.

🛠️ Tech Stack & Tools
Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Environment: Jupyter Notebook / Google Colab

📂 Dataset Description
The dataset includes 21 columns covering:

Demographics: Gender, Senior Citizen status, Partners, and Dependents.

Services: Phone, Multiple Lines, Internet (DSL/Fiber Optic), Online Security, Tech Support, Streaming, etc.

Account Info: Tenure, Contract type (Month-to-month, 1-year, 2-year), Payment Method, Monthly Charges, and Total Charges.

Target: Churn (Yes/No).

🚀 Key Insights from Analysis
1. High-Risk Segments
Contract Type: Customers on Month-to-Month contracts are the most volatile, showing a churn rate of 42.7%.

Internet Service: Fiber Optic users churn at a much higher rate (41.8%) compared to DSL users. This suggests possible pricing or service stability issues with the fiber product.

2. The "Loyalty Threshold"
Tenure: Churn is heavily concentrated in the first 6 months. Once a customer stays past the 24-month mark, their likelihood of leaving drops significantly.

3. Financial Impact
Monthly Charges: Churned customers had a higher median monthly charge (approx. $80) compared to retained customers (approx. $65), indicating price sensitivity.

🧪 Methodology
Data Cleaning: Handled missing values in TotalCharges and converted data types.

Exploratory Data Analysis (EDA): Univariate and bivariate analysis to find correlations between services and churn.

Feature Observation: Analyzed the impact of "Add-on" services (like Tech Support) on retention.

💡 Business Recommendations
Incentivize Long-Term Contracts: Offer discounts to transition Month-to-Month users to 1-Year plans.

Tech Support Bundling: Since customers with Tech Support churn less, bundle this service for high-risk demographics.

Early Intervention: Launch a "New Customer Success" program specifically for users in their first 3–6 months.
