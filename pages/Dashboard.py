import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from built_functions import * 
import tooltip_messages
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset and Perform feature engineering
engineered_data = load_and_preprocess_data()


# Streamlit app interface
st.title("Fraud Detection Dashboard")

# --- Sidebar for Filters ---
st.sidebar.title("Filters", help = tooltip_messages.Filters_Tooltip)

# Step filter (Numeric)
step_filter = st.sidebar.slider("Step", 
                                int(engineered_data['step'].min()), int(engineered_data['step'].max()),
                                (int(engineered_data['step'].min()), int(engineered_data['step'].max())),
                               help = tooltip_messages.Step_Filter_Tooltip)

# Multi-select for categorical filters
time_of_day_filter = st.sidebar.multiselect(
                                "Time of Day", 
                                engineered_data['time_of_day'].dropna().unique().tolist(),  # Ensure it's a list for the options
                                default=engineered_data['time_of_day'].dropna().unique().tolist(),  # Default selected values should be a list
                                help = tooltip_messages.Time_of_Day_Filter_Tooltip
                            )

type_filter = st.sidebar.multiselect(
                                "Transaction Type", 
                                engineered_data['type'].dropna().unique().tolist(),  # Ensure it's a list for the options
                                default=engineered_data['type'].dropna().unique().tolist(),  # Default selected values should be a list
                                help = tooltip_messages.Transaction_Type_Filter_Tooltip
                            )

# Apply filters
filtered_df = engineered_data[
    (engineered_data['step'] >= step_filter[0]) & 
    (engineered_data['step'] <= step_filter[1]) & 
    (engineered_data['time_of_day'].isin(time_of_day_filter)) &
    (engineered_data['type'].isin(type_filter))
]

# # Display the filtered DataFrame
# st.subheader("Filtered Transactions")
# st.write(filtered_df)


# --- Metrics Section ---
st.subheader("Key Fraud Insights")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    total_transactions = len(filtered_df)
    st.metric(label="Total Transactions", value=f"{total_transactions:,}", help = tooltip_messages.Total_Transactions_Tooltip)

with metric_col2:
    total_fraud = filtered_df['isFraud'].sum()  # Use the numeric column for sum
    st.metric(label="Total Fraudulent Transactions", value=f"{total_fraud:,}", delta_color="inverse", 
              help = tooltip_messages.Total_Fraudulent_Transactions_Tooltip)

with metric_col3:
    fraud_rate = (total_fraud / total_transactions) * 100
    st.metric(label="Fraud Rate (%)", value=f"{fraud_rate:.2f}%", delta_color="inverse",
             help = tooltip_messages.Fraud_Rate_Tooltip)

with metric_col4:
    avg_amount = filtered_df['amount'].mean()
    st.metric(label="Avg Transaction Amount", value=f"${avg_amount:,.2f}",
             help = tooltip_messages.Avg_Transaction_Amount_Tooltip)

metric_col5, metric_col6 = st.columns(2)

with metric_col5:
    avg_fraud_amount = filtered_df[filtered_df['isFraud'] == 1]['amount'].mean()  # Use the numeric column
    st.metric(label="Avg Fraudulent Transaction", value=f"${avg_fraud_amount:,.2f}", delta_color="inverse",
             help = tooltip_messages.Avg_Fraudulent_Transaction_Amount_Tooltip)

with metric_col6:
    total_loss = filtered_df[filtered_df['isFraud'] == 1]['amount'].sum()  # Use the numeric column
    st.metric(label="Total Loss Due to Fraud", value=f"${total_loss:,.2f}", delta_color="inverse",
             help = tooltip_messages.Total_Loss_Due_to_Fraud_Tooltip)

# --- Complementary Charts Section ---
st.subheader("Fraud vs Legitimate Transactions")

metric_chart_col7, metric_chart_col8 = st.columns([1.5, 1])  # Chart on the left, Data on the right

with metric_chart_col7:

    fraud_label_map = {0: 'Non-Fraud', 1: 'Fraud'}
    filtered_df['isFraud_label'] = filtered_df['isFraud'].map(fraud_label_map)
    
    fraud_pie = px.pie(filtered_df, names='isFraud_label', title="Fraud vs. Legitimate Transactions",
                        color_discrete_sequence=['#007BFF', '#00A2E8'], hole=0.4,
                      category_orders={'isFraud_label': ['Non-Large', 'Large']})  # Custom category order)
    st.plotly_chart(fraud_pie, use_container_width=True)
    with st.expander("Analysis"):
        st.caption(tooltip_messages.Fraud_vs_Legitimate_Transactions_Tooltip)

with metric_chart_col8:
    fraud_type_bar = px.bar(filtered_df.groupby('type')['isFraud'].sum().reset_index(),  # Use numeric column
                            x='type', y='isFraud', 
                            title="Fraud Count by Transaction Type",
                            labels={'isFraud': 'Number of Frauds'},
                            color='isFraud', color_continuous_scale='Blues')
    
    st.plotly_chart(fraud_type_bar, use_container_width=True)
    with st.expander("Analysis"):
        st.caption(tooltip_messages.Fraud_Count_by_Transaction_Type_Tooltip)
    


# --- Analysis Visualizations ---


# Set the title of the app
st.subheader('Transaction Analysis and Fraud Trends')

# Fraud Trends Tab
tab1, tab2 = st.tabs(["Fraud Trends", "Transaction Insights"])

with tab1:
    st.subheader("Fraud Trends")
    
    # Fraud vs Non-Fraud Transactions
    # Create two columns for side-by-side charts
    chart_tab_1_col1, chart_tab_1_col2 = st.columns(2)
    
    # Chart for fraud in large transactions
    with chart_tab_1_col1:
        fraud_large = px.bar(
        filtered_df[filtered_df['large_transaction'] == 1].groupby('hour')['isFraud'].sum().reset_index(), 
        x='hour', 
        y='isFraud', 
        title="Number of Fraudulent Transactions in Large Transactions by Hour", 
        color='isFraud', 
        color_discrete_sequence=['#FF5733'], 
        labels={'hour': 'Hour of Day', 'isFraud': 'Number of Frauds'},
        category_orders={"hour": sorted(filtered_df['hour'].unique())}  # Sort the hours
    )
        st.plotly_chart(fraud_large)
        with st.expander("Analysis"):
            st.caption(tooltip_messages.Fraudulent_Transactions_in_Large_Transactions_by_Hour_Tooltip)

    # Chart for fraud in non-large transactions
    with chart_tab_1_col2:
        fraud_non_large = px.bar(
        filtered_df[filtered_df['large_transaction'] == 0].groupby('hour')['isFraud'].sum().reset_index(), 
        x='hour', 
        y='isFraud', 
        title="Number of Fraudulent Transactions in Non-Large Transactions by Hour", 
        color='isFraud', 
        color_discrete_sequence=['#33C1FF'], 
        labels={'hour': 'Hour of Day', 'isFraud': 'Number of Frauds'},
        category_orders={"hour": sorted(filtered_df['hour'].unique())}  # Sort the hours
    )
        st.plotly_chart(fraud_non_large)
        with st.expander("Analysis"):
            st.caption(tooltip_messages.Fraudulent_Transactions_in_Non_Large_Transactions_by_Hour_Tooltip)
    
    # Create a histogram for fraudulent transactions' amounts
    fraud_hist = px.histogram(
        filtered_df[filtered_df['isFraud'] == 1],  # Filter for fraud cases only
        x='amount',  # Plot the transaction amount
        title="Transaction Amount Distribution for Fraudulent Transactions", 
        color_discrete_sequence=['#00A2E8'],  # Choose color for the bars
        nbins=100,  # Adjust number of bins as needed
        labels={'amount': 'Transaction Amount'},  # Label for the x-axis
    )
    
    st.plotly_chart(fraud_hist)
    with st.expander("Analysis"):
        st.caption(tooltip_messages.Transaction_Amount_Distribution_for_Fraudulent_Transactions_Tooltip)

    
    # Group by 'day' and count fraud cases (isFraud == 1)
    fraud_cases_per_day = filtered_df[filtered_df['isFraud'] == 1].groupby('day').size().reset_index(name='fraud_count')
    
    # Create a line chart showing the number of fraud cases per day
    fraud_line = px.line(
        fraud_cases_per_day, 
        x='day', 
        y='fraud_count', 
        title="Fraud Cases Over Time (Per Day)", 
        color_discrete_sequence=['#007BFF'], 
        labels={'day': 'Day', 'fraud_count': 'Number of Fraud Cases'}
    )
    
    st.plotly_chart(fraud_line)
    with st.expander("Analysis"):
        st.caption(tooltip_messages.Fraud_Cases_Over_Time_Tooltip)


with tab2:
    st.subheader("Transaction Insights")
    
    # Create two columns to display the charts side by side
    chart_tab_2_col1, chart_tab_2_col2 = st.columns(2)

    # Assuming 'large_transaction' and 'rapid_transactions' are numeric (0 and 1), we can map them to labels
    label_map = {0: 'Non-Large', 1: 'Large'}
    filtered_df['large_transaction_label'] = filtered_df['large_transaction'].map(label_map)
    
    label_map_rapid = {0: 'Non-Rapid', 1: 'Rapid'}
    filtered_df['rapid_transaction_label'] = filtered_df['rapid_transactions'].map(label_map_rapid)

    
    # Large Transaction Distribution - First column
    with chart_tab_2_col1:
        lt_dist = px.pie(
        filtered_df, 
        names='large_transaction_label',  # Use the mapped label for large transactions
        title="Large Transaction Distribution", 
        color_discrete_sequence=['#007BFF', '#00A2E8'],
        hole=0.4,  # Adding a hole to make it a donut chart
        category_orders={'large_transaction_label': ['Non-Large', 'Large']}  # Custom category order
    )
        st.plotly_chart(lt_dist)
        with st.expander("Analysis"):
            st.caption(tooltip_messages.Large_Transaction_Distribution_Tooltip)
    
    # Rapid Transactions Distribution - Second column
    with chart_tab_2_col2:
        rt_dist = px.pie(
        filtered_df, 
        names='rapid_transaction_label',  # Use the mapped label for rapid transactions
        title="Rapid Transactions Distribution", 
        color_discrete_sequence=['#007BFF', '#00A2E8'],
        hole=0.4,  # Adding a hole to make it a donut chart
        category_orders={'rapid_transaction_label': ['Non-Rapid', 'Rapid']}  # Custom category order
    )
        st.plotly_chart(rt_dist)
        with st.expander("Analysis"):
            st.caption(tooltip_messages.Rapid_Transactions_Distribution_Tooltip)


    # Low Balance Transaction Count by Day (Line Chart)
    # Filter for low balance transactions (where low_balance = 1)
    low_balance_df = filtered_df[filtered_df['low_balance'] == 1]
    
    # Group by day and count low balance transactions
    lb_line = px.line(
        low_balance_df.groupby('day').size().reset_index(name='count'),
        x='day',
        y='count',
        title="Low Balance Transactions by Day",
        labels={'count': 'Low Balance Transaction Count'},
        line_shape='linear',  # To ensure it's a line chart
        color_discrete_sequence=['#007BFF']
    )
    
    st.plotly_chart(lb_line)
    with st.expander("Analysis"):
        st.caption(tooltip_messages.Low_Balance_Transactions_by_Day_Tooltip)

    # Group by time_of_day, then count the number of transactions
    tod_bar = px.bar(
        filtered_df.groupby('time_of_day').size().reset_index(name='count'),
        x='time_of_day',
        y='count',
        title="Total Transactions Per Time of Day",
        labels={'time_of_day': 'Time of Day', 'count': 'Transaction Count'},
        color_discrete_sequence=['#00A2E8']
    )
    
    st.plotly_chart(tod_bar)
    with st.expander("Analysis"):
        st.caption(tooltip_messages.Total_Transactions_Per_Time_of_Day_Tooltip)
    
    # Heatmap of Feature Correlations

    # Calculate the first correlation matrix (large_transaction, low_balance, transactions_per_hour, rapid_transactions)
    corr_df1 = filtered_df[['large_transaction', 'low_balance', 'transactions_per_hour', 'rapid_transactions']].corr()
    
    # Calculate the second correlation matrix (large_transaction, non_large_transaction, fraud)
    corr_df2 = filtered_df[['large_transaction', 'low_balance', 'isFraud']].copy()
    corr_df2['non_large_transaction'] = 1 - corr_df2['large_transaction']  # Creating the non_large_transaction column
    corr_df2 = corr_df2.corr()
    
    # Create two columns in Streamlit
    st.subheader('Correlations')
    heat_tab_2_col1, heat_tab_2_col2 = st.columns(2)
    
    # Plot the first heatmap (correlations for large_transaction, low_balance, etc.) in the first column
    with heat_tab_2_col1:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_df1,
            annot=True,
            cmap='Blues',
            fmt='.2f',
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            linecolor='none',
            square=True,
            ax=ax1
        )
        #ax1.set_title("Feature Correlations (Large Transaction, Low Balance, etc.)")
        # Change axis labels and title text color to white
        ax1.set_xticklabels(ax1.get_xticklabels(), color='white')
        ax1.set_yticklabels(ax1.get_yticklabels(), color='white')

        # Adjust tick labels size
        ax1.tick_params(axis='x', labelsize=8, labelcolor='white')  # Change the font size for x-axis labels
        ax1.tick_params(axis='y', labelsize=8, labelcolor='white')  # Change the font size for y-axis labels

        # Adjust colorbar tick labels to white
        cbar = ax1.collections[0].colorbar
        cbar.ax.tick_params(labelcolor='white')
        
        # Remove background color of the plot
        fig1.patch.set_visible(False)  # This hides the background of the figure
        ax1.set_facecolor('none')  # Make the axes background transparent
        
        st.pyplot(fig1)
        with st.expander("Analysis"):
            st.caption(tooltip_messages.Feature_Correlations_Tooltip)
    
    # Plot the second heatmap (correlations for large_transaction, non_large_transaction, and fraud) in the second column
    with heat_tab_2_col2:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_df2,
            annot=True,
            cmap='Blues',
            fmt='.2f',
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            linecolor='none',
            square=True,
            ax=ax2
        )
        #ax2.set_title("Correlations (Large Transaction, Non-Large Transaction, Fraud)")
        # Change axis labels and title text color to white
        ax2.set_xticklabels(ax2.get_xticklabels(), color='white')
        ax2.set_yticklabels(ax2.get_yticklabels(), color='white')

        # Adjust tick labels size
        ax2.tick_params(axis='x', labelsize=8, labelcolor='white')  # Change the font size for x-axis labels
        ax2.tick_params(axis='y', labelsize=8, labelcolor='white')  # Change the font size for y-axis labels

        # Adjust colorbar tick labels to white
        cbar = ax2.collections[0].colorbar
        cbar.ax.tick_params(labelcolor='white')
        
        # Remove background color of the plot
        fig2.patch.set_visible(False)  # This hides the background of the figure
        ax2.set_facecolor('none')  # Make the axes background transparent
        
        st.pyplot(fig2)
        with st.expander("Analysis"):
            st.caption(tooltip_messages.Fraud_Correlations_Tooltip)



