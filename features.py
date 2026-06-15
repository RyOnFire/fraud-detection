import pandas as pd
import numpy as np

def engineer_features(df):
    # Time of day (fraud often happens late at night)
    df['hour'] = (df['Time'] % 86400) // 3600
    
    # How different is this amount from the average?
    df['amount_zscore'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
    
    # Is this a small amount? (testing pattern)
    df['is_small_amount'] = (df['Amount'] < 10).astype(int)
    
    # Is this a large amount?
    df['is_large_amount'] = (df['Amount'] > 1000).astype(int)
    
    return df