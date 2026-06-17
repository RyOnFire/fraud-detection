# Fraud Detection Dashboard

An interactive web application that detects fraudulent credit card transactions using machine learning. Built with Python, scikit-learn, XGBoost, and Streamlit.

## Overview

This project analyzes transaction data and flags potentially fraudulent activity using two trained models: XGBoost and Random Forest with SMOTE oversampling. Users can upload their own transaction CSV and get instant fraud predictions, visualizations, and a downloadable report of flagged transactions.

## Features

- Upload any CSV of credit card transactions
- Choose between XGBoost or Random Forest models
- View key metrics: total transactions, flagged fraud, normal transactions
- Interactive charts showing fraud distribution and probability scores
- Download flagged transactions as a CSV report
- Fallback mode: if your CSV doesn't match the expected format, map your own columns and train a custom model on the spot

## Dataset

Trained on the Credit Card Fraud Detection dataset from Kaggle, 284,807 real anonymized European transactions, with a fraud rate of 0.17 percent.

## Model Performance

| Model | Precision | Recall | F1 Score |
|---|---|---|---|
| XGBoost | 99% | 83% | 90% |
| Random Forest + SMOTE | 89% | 85% | 87% |

## Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Streamlit
- Plotly

## How to Run Locally

git clone https://github.com/RyOnFire/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
streamlit run app.py

You will need to download creditcard.csv from Kaggle separately and place it in the project folder, since it is too large for GitHub.

## How It Works

1. Transaction data is cleaned and scaled (Amount and Time columns)
2. SMOTE balances the dataset for Random Forest training, since fraud cases are rare
3. XGBoost handles class imbalance internally using scale_pos_weight
4. Both models output a fraud probability for each transaction
5. The dashboard visualizes results and lets users export flagged transactions

## Future Improvements

- Real-time transaction scoring via API
- Model monitoring and retraining pipeline
- Ensemble model combining XGBoost and Random Forest
- Expanded feature engineering (time of day, transaction velocity)
