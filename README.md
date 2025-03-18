# Fraud Detection Analysis

## Overview
This project demonstrates the application of machine learning models to detect fraudulent activities in transaction data. By analysing historical data, the model identifies patterns and characteristics that indicate potential fraud. The goal is to provide actionable insights that can help prevent and detect fraud in real-time transactions.

## Dataset Source

The data used in this project is from the **Synthetic Financial Datasets For Fraud Detection** available on Kaggle. The dataset contains synthetic transaction data generated to simulate fraud detection scenarios. It was used to build and evaluate the fraud detection models in this project.

You can access the dataset on Kaggle here: [Synthetic Financial Fraud Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)


## Project Structure
- **Data Preprocessing**: The initial phase includes data cleaning and feature engineering to prepare the dataset for modeling.
- **Modeling**: Various machine learning algorithms are used to build models, including Random Forest, Gradient Boosted Trees, and Neural Networks.
- **Evaluation**: The models are evaluated on key metrics such as accuracy, precision, recall, and F1 score to determine their effectiveness in detecting fraud.
- **Streamlit App**: An interactive web application is built using Streamlit to visualise the analysis results, display model performance, and provide a user-friendly interface for exploring fraud detection insights.

## Key Features
- **Fraud Detection**: The app uses a machine learning model to predict whether a transaction is fraudulent or not.
- **Interactive Visualizations**: Users can explore various visualizations such as confusion matrices, ROC curves, and feature importance charts.
- **Model Performance Metrics**: The app provides performance metrics like accuracy, precision, recall, and F1 score for each model.
- **Data Exploration**: Users can interact with the data to explore features like transaction amount, user behavior, and more.

## Technologies Used
- **Python**: Programming language used for data analysis and machine learning model development.
- **Streamlit**: Web application framework used to build the interactive dashboard.
- **Pandas**: Data manipulation and analysis library.
- **Scikit-learn**: Machine learning library for building and evaluating models.
- **Matplotlib/Seaborn**: Libraries used for creating visualizations.

## Setup and Installation

### Prerequisites
To run this project, you'll need Python 3.x installed on your system. You'll also need to install the required dependencies.

### Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/fraud-detection-analysis.git
    cd fraud-detection-analysis
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

The app will open in your web browser, where you can interact with the fraud detection analysis.

## Usage
Once the app is running, you can explore the following sections:
- **Project Overview**: An introduction to the fraud detection project and its goals.
- **Data Visualisation**: Interactive charts to explore features such as transaction amounts, user behaviors, and more.
- **Model Performance**: View key performance metrics and compare different models.
- **Fraud Detection**: Use the trained model to predict fraud in new, unseen transaction data.

## Performance Metrics
The following evaluation metrics are used to assess the models:
- **Accuracy**: The overall percentage of correct predictions.
- **Precision**: The percentage of positive predictions that are correct.
- **Recall**: The percentage of actual fraud cases that were detected.
- **F1 Score**: The harmonic mean of precision and recall, balancing the two.

## Insights
- **High Transaction Amounts**: Transactions with higher amounts are more likely to be flagged as fraud.
- **Unusual User Behavior**: Patterns like rapid, geographically diverse transactions are strong indicators of potential fraud.
- **Model Improvement**: The model’s performance continues to improve with additional data and fine-tuning.

---

For more information or questions, feel free to reach out to me via GitHub Issues or email.
