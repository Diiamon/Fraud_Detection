#--Filters (Sidebar)--
Filters_Tooltip = (
    "Filters allow you to refine the dataset based on specific criteria, directly impacting the insights shown on the dashboard. "
    "By selecting different filters, you can focus on particular transaction types, time periods, or characteristics, making it easier to identify trends and anomalies. "
    "Since all metrics, charts, and analyses are based on the filtered data, applying different filters can lead to varying conclusions. "
    "For example, filtering by nighttime transactions might reveal higher fraud rates, whereas filtering by transaction type could highlight which categories are most targeted by fraudsters. "
    "Carefully adjusting filters helps uncover meaningful patterns and improves decision-making."
)

Step_Filter_Tooltip = "Filters transactions based on the time step, which represents the transaction's sequence in the dataset."

Time_of_Day_Filter_Tooltip = "Filters transactions by the time of day when they occurred, such as morning, afternoon, evening, or night."

Transaction_Type_Filter_Tooltip = "Filters transactions by type, such as Cash Out, Payment, or Transfer."

#--Metrics Section--

Total_Transactions_Tooltip = "The total number of transactions in the selected dataset based on the applied filters."

Total_Fraudulent_Transactions_Tooltip = "The total number of transactions flagged as fraudulent in the selected dataset."

Fraud_Rate_Tooltip = "The percentage of transactions that are fraudulent out of the total transactions."

Avg_Transaction_Amount_Tooltip = "The average monetary value of all transactions in the selected dataset."

Avg_Fraudulent_Transaction_Amount_Tooltip = "The average monetary value of transactions that were marked as fraudulent."

Total_Loss_Due_to_Fraud_Tooltip = "The total amount of money lost due to fraudulent transactions."

#--Charts (Fraud vs Legitimate Transactions)--

Fraud_vs_Legitimate_Transactions_Tooltip = (
    "This pie chart visualizes the proportion of fraudulent transactions versus legitimate ones. "
    "A high percentage of fraudulent transactions may indicate a compromised dataset or a vulnerability in the system. "
    "If fraud makes up a small percentage, it suggests that most transactions are legitimate but still require monitoring."
)

Fraud_Count_by_Transaction_Type_Tooltip = (
    "This bar chart displays the number of fraudulent transactions for each transaction type. "
    "Certain transaction types, such as 'Cash Out' or 'Transfer,' may be more susceptible to fraud. "
    "A high fraud count in a specific category suggests that fraudsters may be targeting that type of transaction."
)


#--Fraud Trends (Tab 1)--

Fraudulent_Transactions_in_Large_Transactions_by_Hour_Tooltip = (
    "This bar chart shows the number of fraudulent large transactions across different hours of the day. "
    "A high fraud rate during off-peak hours (e.g., late at night) may indicate that fraudsters exploit reduced monitoring. "
    "Alternatively, high fraud during business hours could mean fraud is disguised among legitimate transactions."
)

Fraudulent_Transactions_in_Non_Large_Transactions_by_Hour_Tooltip = (
    "This bar chart displays the number of fraudulent non-large transactions at different times of the day. "
    "If fraud is concentrated in smaller transactions, fraudsters may be testing stolen accounts with low-value transfers. "
    "Patterns emerging during specific hours could suggest automated fraud attempts."
)

Transaction_Amount_Distribution_for_Fraudulent_Transactions_Tooltip = (
    "This histogram illustrates how fraudulent transactions are distributed by amount. "
    "A concentration of fraud at low values may indicate small-scale, repeated fraud attempts. "
    "Conversely, fraud clustered at high amounts could signal organized fraud operations."
)

Fraud_Cases_Over_Time_Tooltip = (
    "This line chart tracks the number of fraud cases occurring each day over time. "
    "Sudden spikes could indicate a targeted fraud campaign, a new security vulnerability, or a failure in fraud detection. "
    "A steady rise in fraud cases may suggest that fraudsters are adapting to detection methods."
)


#--Transaction Insights (Tab 2)--

Large_Transaction_Distribution_Tooltip = (
    "This pie chart shows the proportion of large transactions compared to non-large transactions. "
    "A higher proportion of large transactions could mean a high-value client base, but it also increases the risk of significant fraud losses."
)

Rapid_Transactions_Distribution_Tooltip = (
    "This pie chart displays the percentage of transactions occurring in rapid succession. "
    "A high concentration of rapid transactions could indicate bot activity, account takeovers, or laundering attempts. "
    "Monitoring these transactions helps identify unusual bursts of activity."
)

Low_Balance_Transactions_by_Day_Tooltip = (
    "This line chart tracks the number of low-balance transactions over time. "
    "Frequent low-balance transactions may indicate financially unstable users, increased withdrawal activity, or attempts to drain compromised accounts."
)

Total_Transactions_Per_Time_of_Day_Tooltip = (
    "This bar chart displays the total number of transactions occurring at different times of the day. "
    "Understanding transaction patterns helps identify peak activity periods, which can assist in fraud detection and resource allocation. "
    "If fraud is concentrated during specific hours, adjusting monitoring strategies accordingly may reduce risk."
)


#--Correlation Heatmaps--

Feature_Correlations_Tooltip = (
    "This heatmap shows the relationships between various transaction features, such as large transactions, low balance, transaction frequency, and rapid transactions. "
    "Strong correlations may indicate that certain features contribute significantly to fraud risk."
)

Fraud_Correlations_Tooltip = (
    "This heatmap displays how fraud is correlated with large and non-large transactions, as well as low balance. "
    "Identifying strong correlations can help refine fraud detection models and improve risk assessment strategies."
)

#--Fraud Prediction Data Input--

Low_Balance_Tooltip = (
    "Indicates whether the account had a low balance before the transaction. "
    "Enter **1 for Yes (low balance)** and **0 for No (sufficient balance)**. "
    "A low balance could indicate a higher risk of fraudulent activity if the account is frequently drained."
)

Avg_Transaction_Amount_Tooltip = (
    "The average monetary value of past transactions associated with the account. "
    "A sudden increase or decrease from the usual average could indicate suspicious activity. "
    "Enter a numerical value (e.g., 100.50 for £100.50)."
)

Hour_Tooltip = (
    "The hour of the day when the transaction occurred (0-23). "
    "Fraudulent transactions often happen at unusual hours, such as late at night, when users are less likely to monitor their accounts."
)

New_Balance_Destination_Tooltip = (
    "The balance of the recipient’s account after the transaction is completed. "
    "Enter a numerical value (e.g., 5000.00 for £5000.00). "
    "A zero or unusually high balance could indicate fraudulent transfers."
)

Old_Balance_Destination_Tooltip = (
    "The balance of the recipient’s account before the transaction. "
    "Enter a numerical value (e.g., 5000.00 for £5000.00). "
    "Comparing this with the new balance helps detect inconsistencies, which may suggest fraud."
)

Amount_Tooltip = (
    "The monetary value of the transaction. "
    "Enter a numerical value (e.g., 250.00 for £250.00). "
    "Fraudulent transactions may involve unusually large amounts or multiple small transactions designed to bypass detection."
)

Step_Tooltip = (
    "Represents the time step of the transaction in the dataset, acting as a sequence number. "
    "It helps track when transactions occur and identify patterns over time. "
    "Enter a numerical value (e.g., 200 for step 200 in the dataset)."
)
