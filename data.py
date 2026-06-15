import pandas as pd 
import plotly.express as px
from sklearn.preprocessing import StandardScaler

def load_new_data():
    df = pd.read_csv('card_transdata.csv')
    
    scaler = StandardScaler()
    cols_to_scale = ['distance_from_home', 
                     'distance_from_last_transaction',
                     'ratio_to_median_purchase_price']
    
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df
    


def load_data():
    df = pd.read_csv('creditcard.csv')
    
    # Engineer features before scaling and dropping
    df['hour'] = (df['Time'] % 86400) // 3600
    df['amount_zscore'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
    df['is_small_amount'] = (df['Amount'] < 10).astype(int)
    df['is_large_amount'] = (df['Amount'] > 1000).astype(int)
    
    # Scale Amount and Time
    scaler = StandardScaler()
    df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_Scaled'] = scaler.fit_transform(df[['Time']])
    df = df.drop(['Amount', 'Time'], axis=1)
    
    return df

def explore_data(df):
    print(df.head())

    print('length of df:', len(df))
    print('number of columns:', len(df.columns))
    print('columns:', df.columns)
    print('missing values:', df.isna().sum())

    fraudulent = df['Class'].value_counts()[1]
    print('fraudulent:', fraudulent)
    non_fraudulent = df['Class'].value_counts()[0]
    print('non_fraudulent:', non_fraudulent)
    average_transaction_amount = df['Amount_Scaled'].mean()
    print('average transaction amount:', average_transaction_amount)
    largest_transaction_amount = df['Amount_Scaled'].max()
    print('largest transaction amount:', largest_transaction_amount)

    counts = df['Class'].value_counts()
    fig = px.bar(x=['Normal', 'Fraud'], y=counts.values, title='Fraud vs Normal Transactions')
    fig.show()

    fig2 = px.histogram(df, x='Amount_Scaled', nbins=50, title='Distribution of Transaction Amounts')
    fig2.show()
