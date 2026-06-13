import streamlit as st
import pandas as pd
import plotly.express as px
from data import load_data
from model import train_model, predict
from sklearn.preprocessing import StandardScaler
from ai_detector import detect_columns



st.title('🔍 Fraud Detection Dashboard')
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
sensitivity = st.sidebar.slider("Sensitivity", 0.001, 0.1, 0.00172)
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # AI column detection
    with st.spinner("AI is detecting columns..."):
        detected = detect_columns(df)

    # Pre-fill dropdowns with AI suggestions
    amount_col = st.selectbox("Select Amount Column", df.columns,
                              index=list(df.columns).index(detected['amount_col']))
    time_col = st.selectbox("Select Time/Date Column", df.columns,
                            index=list(df.columns).index(detected['time_col']))
    target_col = st.selectbox("Select Target Column", ['None'] + list(df.columns),
                              index=0 if detected['target_col'] == 'None'
                              else list(df.columns).index(detected['target_col']) + 1)

    scaler = StandardScaler()
    df['Amount_Scaled'] = scaler.fit_transform(df[[amount_col]])
    df['Time_Scaled'] = scaler.fit_transform(df[[time_col]])
    df = df.drop([amount_col, time_col], axis=1)

    # Separate labels from features
    if target_col != 'None':
        labels = df[target_col]
        features = df.drop(target_col, axis=1)
    else:
        labels = None
        features = df

    # Train model using sensitivity slider value
    model = train_model(features, contamination=sensitivity)

    # Make predictions
    predictions = predict(model, features)

    # KPI cards
    total = len(df)
    flagged = sum(predictions == -1)
    normal = sum(predictions == 1)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", total)
    col2.metric("Flagged as Fraud", flagged)
    col3.metric("Normal Transactions", normal)

    # Bar chart
    st.subheader("Fraud vs Normal Transactions")
    fig = px.bar(
        x=['Normal', 'Fraud'],
        y=[normal, flagged],
        color=['Normal', 'Fraud'],
        title='Transaction Distribution'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Transaction table
    st.subheader("Flagged Transactions")
    df['Prediction'] = predictions
    df['Prediction'] = df['Prediction'].map({1: 'Normal', -1: 'Fraud'})
    flagged_df = df[df['Prediction'] == 'Fraud']
    st.dataframe(flagged_df)

    st.download_button(
        label="Download Flagged Transactions",
        data=flagged_df.to_csv(index=False),
        file_name="flagged_transactions.csv",
        mime="text/csv"
    )