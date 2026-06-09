import pandas as pd 
import plotly.express as px
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('creditcard.csv')
print(df.head())

print('length of df:', len(df))
print('number of columns:', len(df.columns))
print('columns:', df.columns)
print('missing values:', df.isna().sum())

fraudulent = df['Class'].value_counts()[1]
print('fraudulent:', fraudulent)
non_fraudulent = df['Class'].value_counts()[0]
print('non_fraudulent:', non_fraudulent)
average_transaction_amount = df['Amount'].mean()
print('average transaction amount:', average_transaction_amount)
largest_transaction_amount = df['Amount'].max()
print('largest transaction amount:', largest_transaction_amount)

counts = df['Class'].value_counts()
fig = px.bar(x=['Normal', 'Fraud'], y=counts.values, title='Fraud vs Normal Transactions')
fig.show()

fig2 = px.histogram(df, x='Amount', nbins=50, title='Distribution of Transaction Amounts')
fig2.show()

scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_Scaled'] = scaler.fit_transform(df[['Time']])
df = df.drop(['Amount', 'Time'], axis=1)
print(df[['Amount_Scaled', 'Time_Scaled']].head())