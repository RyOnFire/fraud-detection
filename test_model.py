import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from data import load_new_data

# Load saved models
xgb_model = joblib.load('xgb_model.pkl')
rf_model = joblib.load('rf_model.pkl')

# Load real data to fit scaler
df = load_new_data()

# Create test transactions
test_transactions = pd.DataFrame({
    'distance_from_home': [5, 2000, 1, 500],
    'distance_from_last_transaction': [2, 1500, 0.5, 800],
    'ratio_to_median_purchase_price': [1.0, 8.5, 0.9, 6.2],
    'repeat_retailer': [1, 0, 1, 0],
    'used_chip': [1, 0, 1, 0],
    'used_pin_number': [1, 0, 1, 0],
    'online_order': [0, 1, 0, 1]
})

# Scale the numeric columns
scaler = StandardScaler()
cols_to_scale = ['distance_from_home', 
                 'distance_from_last_transaction',
                 'ratio_to_median_purchase_price']

scaler.fit(df[cols_to_scale])
test_transactions[cols_to_scale] = scaler.transform(test_transactions[cols_to_scale])

print("XGBoost Predictions:")
xgb_preds = xgb_model.predict(test_transactions)
print(xgb_preds)

print("\nRandom Forest Predictions:")
rf_preds = rf_model.predict(test_transactions)
print(rf_preds)