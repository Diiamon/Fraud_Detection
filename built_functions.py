import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os
import kaggle
import zipfile

# Define the path to save the dataset
dataset_path = 'Fraud_Detection/fraud_data.zip'

# Function to download the dataset using the Kaggle API
def download_dataset():
    # If the dataset is already downloaded, skip downloading
    if not os.path.exists(dataset_path):
        print("Downloading the dataset...")
        # Adjusting the dataset download command with correct path for your dataset
        kaggle.api.dataset_download_file('ealaxi/paysim1', 'PS_20174392719_1491204439457_log.csv', path='Fraud_Detection/data/')
        print("Download complete!")
    else:
        print("Dataset already downloaded.")

# Function to extract the dataset (if it's a zip file)
def extract_dataset():
    if os.path.exists(dataset_path):
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall('Fraud_Detection/data/')
        print("Dataset extracted.")

# Feature Engineering function
@st.cache_resource
def feature_engeneering(df):
    df = df.copy()  # Create a copy to avoid modifying the original data
    
    # Feature engineering steps
    df['day'] = df['step'] // 24  # Convert hours to days
    df['hour'] = df['step'] % 24  # Extract the hour of the day
    df['time_of_day'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    # Transaction-related features
    df['transactions_per_day'] = df.groupby(['nameOrig', 'day'])['step'].transform('count')
    df['avg_transaction_amount'] = df.groupby('nameOrig')['amount'].transform('mean')
    
    # Flag large transactions
    threshold = df['amount'].quantile(0.95)
    df['large_transaction'] = (df['amount'] > threshold).astype(int)
    
    # Flag low balance after transaction
    df['low_balance'] = (df['newbalanceOrig'] < df['amount'] * 0.1).astype(int)
    
    # Count transactions per user per hour
    df['transactions_per_hour'] = df.groupby(['nameOrig', 'hour'])['step'].transform('count')
    
    # Flag users who made more than 2 transactions in the same hour
    df['rapid_transactions'] = (df['transactions_per_hour'] >= 2).astype(int)
    
    # Ratio of amount transferred compared to original balance
    df['balance_change_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)  # Add 1 to avoid division by zero
    
    # Flag if the recipient name looks like a business (usually starts with "M")
    df['is_business_dest'] = df['nameDest'].str.startswith('M').astype(int)

    return df

# Function to load and preprocess the data
@st.cache_resource
def load_and_preprocess_data():
    # Download and extract the dataset if it's not already present
    download_dataset()
    extract_dataset()
    
    # Load data only once and cache it
    df = pd.read_csv('Fraud_Detection/data/PS_20174392719_1491204439457_log.csv')
    
    # Apply feature engineering to the data
    df = feature_engeneering(df)
    return df

@st.cache_resource
def tranformation(df):
    df = df.copy()  # Create a copy to avoid modifying original data
    df.drop(["nameOrig", "nameDest"], axis=1, inplace=True)  # Remove sender/receiver names since fraud detection is based on transaction behavior, not user IDs.
    
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])  # Encode transaction type
    df['time_of_day'] = le.fit_transform(df['time_of_day'])  # Encode time of day
    df['is_business_dest'] = df['is_business_dest'].astype(int)  # Convert to binary
    # Drop rows with missing values in 'isFraud' and 'isFlaggedFraud'
    df = df.dropna(subset=['isFraud', 'isFlaggedFraud'])
    return df

@st.cache_resource
def balancing_data(df):
    df = df.copy()  # Create a copy to avoid modifying original data
    X = df.drop("isFraud", axis=1)  # Features
    y = df["isFraud"]  # Target
    
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)  # Synthetic minority oversampling
    return X_resampled, y_resampled

@st.cache_resource
def anomaly_detection(df):
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    df["anomaly_score"] = iso_forest.fit_predict(df.drop("isFraud", axis=1))  # Detect anomalies
    return df

@st.cache_resource
def random_forest_model(X_resampled, y_resampled):
    X_resampled = X_resampled.drop(['balance_change_ratio','newbalanceOrig','oldbalanceOrg','transactions_per_hour',
           'rapid_transactions','day', 'time_of_day', 'transactions_per_day','large_transaction','isFlaggedFraud','type','is_business_dest'], axis=1)
    
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # Train model
    rf_model = RandomForestClassifier(n_estimators=3, random_state=42)
    rf_model_fit=rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    return rf_model, rf_model_fit, y_pred, report
    