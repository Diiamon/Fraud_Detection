from built_functions import *
import tooltip_messages
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load dataset and Perform feature engineering
engineered_data = load_and_preprocess_data()

# Perform transformations
Transformed_data = tranformation(engineered_data)

# Balance the data (assuming resampling is done here)
X_resampled_td, y_resampled_td = balancing_data(Transformed_data)

# Train Random Forest model
rf_model_td, rf_model_fit_td, y_pred_td, report_td = random_forest_model(X_resampled_td, y_resampled_td)

# Streamlit app layout
st.title('Predict with Random Forest Model')

# Sidebar for input data
st.sidebar.header('Input Data for Prediction')


# Add input fields on the sidebar for each feature
low_balance = st.sidebar.number_input('Low Balance', value=0.0, help = tooltip_messages.Low_Balance_Tooltip)
avg_transaction_amount = st.sidebar.number_input('Average Transaction Amount', value=0.0, help = tooltip_messages.Avg_Transaction_Amount_Tooltip)
hour = st.sidebar.number_input('Hour', value=0.0, help = tooltip_messages.Hour_Tooltip)
newbalanceDest = st.sidebar.number_input('New Balance Destination', value=0.0, help = tooltip_messages.New_Balance_Destination_Tooltip)
oldbalanceDest = st.sidebar.number_input('Old Balance Destination', value=0.0, help = tooltip_messages.Old_Balance_Destination_Tooltip)
amount = st.sidebar.number_input('Amount', value=0.0, help = tooltip_messages.Amount_Tooltip)
step = st.sidebar.number_input('Step', value=0.0, help = tooltip_messages.Step_Tooltip)
# Add more input fields based on your dataframe features

# Create a DataFrame from the input values
input_data = pd.DataFrame({
    'step': [step],
    'amount': [amount],
    'oldbalanceDest': [oldbalanceDest],
    'newbalanceDest': [newbalanceDest],
    'hour': [hour],
    'avg_transaction_amount': [avg_transaction_amount],
    'low_balance': [low_balance] 
    # Add more columns as needed
})

# When the user presses the "Predict" button
if st.sidebar.button('Predict'):
    # Directly use input data without feature engineering
    prediction = rf_model_td.predict(input_data)  # Model expects data in the same format it was trained on
    prediction_proba = rf_model_td.predict_proba(input_data)  # Get the probabilities

    # Display the prediction and its meaning
    # Create a single row with 3 buttons
    col1, col2, col3 = st.columns(3)
    
    # Prediction button
    with col1:
        if prediction[0] == 1:
            st.metric(label = "Fraudulent Transaction", value = 1)
        else:
            st.metric(label = "Legitimate Transaction", value = 0)
    
    # Fraud probability button
    risk_label = "High Risk" if prediction_proba[0][1] > 0.5 else "Low Risk"
    delta_value = prediction_proba[0][1] if prediction_proba[0][1] > 0.5 else -(prediction_proba[0][1])
    delta_color = "inverse" if prediction_proba[0][1] > 0.5 else "normal"  # Red for high, green for low

    with col2:
        st.metric(label = "Fraud Probability:", value = round(prediction_proba[0][1],2), delta=risk_label, delta_color=delta_color)
    
    # Legitimate probability button
    with col3:
        st.metric(label = "Legit Probability:", value = round(prediction_proba[0][0],2))

    st.markdown("""
                ### Prediction Output:
                
                - **Prediction:** After the user inputs data, the model makes a prediction (either **0** or **1**).
                  - **0**: Legitimate transaction
                  - **1**: Fraudulent transaction
                  
                - **Probability Information:** We also show the probabilities for each class (fraudulent and legitimate).
                  - This helps understand the model's **confidence** in its prediction.
                """)


    # Feature importance
    st.subheader('Feature Importance')
    feature_importance = rf_model_td.feature_importances_
    feature_names = input_data.columns
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    col4, col5 = st.columns([2,1])
    # Plot feature importance using Plotly (Left side - 3x size)
    with col4:
        st.subheader("Feature Importance Chart")
        fig = px.bar(
            feature_df, 
            x='Feature', 
            y='Importance', 
            title="Feature Importance",
            labels={'Importance': 'Feature Importance'},
            color='Importance',  # Adds a color gradient based on importance
            color_continuous_scale='Blues'  # Optional: Change color theme
        )
        st.plotly_chart(fig, use_container_width=True)  # Ensures it scales well
    
    # Show DataFrame (Right side - 1x size)
    with col5:
        st.subheader("Feature Importance Table")
        st.dataframe(feature_df, height=300)  # Adjust height for better display
    
    st.markdown("""
    ### Feature Importance Explained:

    - The model provides a **feature_importances_** attribute, which tells us how important each feature was for making the decision.
    - We display this as a bar chart using Plotly to visualize which features the model considers most influential.
    """)

    # Display the classification report (train set report)
    st.subheader("Model Classification Report")
    st.text(report_td)
    st.markdown("""
    ### Classification Report Explained:

    - The classification_report gives us a breakdown of the model's performance, including metrics like:
        - **Precision**: How many positive predictions were actually correct.
        - **Recall**: How many actual positives were correctly identified.
        - **F1-Score**: A balance between precision and recall.
        - **Support**: The number of occurrences of each class in the dataset.
    
    """)