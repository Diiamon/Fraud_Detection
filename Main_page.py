import streamlit as st

# Set the title of the app
st.title('Fraud Detection Analysis')

# Introduction section
st.header("Welcome to the Fraud Detection Analysis App")
st.write("""
    This application is designed to showcase the results and insights from the fraud detection analysis. 
    Using advanced machine learning techniques, I aim to identify fraudulent patterns and provide 
    actionable insights to prevent and detect fraud effectively.
""")

# Project Overview section
st.header("Project Overview")
st.write("""
    In this analysis, I utilised the Synthetic Financial Datasets for Fraud Detection from Kaggle to build a machine learning model 
    that can detect fraudulent activities. The model is trained using historical transaction data 
    and incorporates multiple features, including transaction amount, user behavior, and more.
""")

# Key Metrics Section (You can replace these with your actual metrics)
st.header("Key Metrics")
st.write("""
    - **Accuracy** 
    - **Precision**
    - **Recall**
    - **F1 Score**
""")

# Insights Section
st.header("Key Insights")
st.write("""
    - High-risk transactions tend to have higher amounts and unusual user behaviors.
    - Certain patterns like rapid transactions in different geographical locations 
      are a strong indicator of potential fraud.
    - The model's predictions are constantly improving with more data and fine-tuning.
""")

# Call to Action Section
st.header("Explore the Analysis")
st.write("""
    Dive deeper into the detailed fraud detection analysis below. 
    You can explore the interactive visualisations, data exploration tools, and model performance results.
    Explore the Dashboard and Fraud prediction pages.
""")


# Footer (Optional)
st.markdown("---")
st.write("Created by Olukemi Alake - Data Analyst ")
