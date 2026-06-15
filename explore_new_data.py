import pandas as pd

df = pd.read_csv('card_transdata.csv')
print(df.head())
print(df.shape)
print(df.columns)
print(df['fraud'].value_counts())
print(df.isna().sum())