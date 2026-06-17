import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title('🔍 Fraud Detection Dashboard')

# Load models
xgb_model = joblib.load('xgb_model.pkl')
rf_model = joblib.load('rf_model.pkl')

# Sidebar
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
model_choice = st.sidebar.selectbox("Select Model", ["XGBoost", "Random Forest"])

st.sidebar.divider()
st.sidebar.subheader("Expected Columns")
st.sidebar.write("Time, V1-V28, Amount, Class (optional)")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    v_cols = [f'V{i}' for i in range(1, 29)]
    required_cols = v_cols + ['Amount', 'Time']
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        st.warning("We don't recognize this CSV format. Let's map your columns!")

        target_col = st.selectbox("Which column is the fraud/target label?", df.columns)
        exclude_from_features = ['Prediction', 'Fraud_Probability']
        feature_cols = st.multiselect(
            "Select columns to use for detection",
            [col for col in df.columns if col != target_col],
            default=[col for col in df.columns if col != target_col and col not in exclude_from_features]
        )

        if st.button("Train Model on My Data"):
            with st.spinner("Training model on your data..."):
                from model import train_model_with_smote

                numeric_cols = df[feature_cols].select_dtypes(
                    include=['float64', 'int64']).columns.tolist()
                scaler = StandardScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

                features = df[feature_cols]
                labels = df[target_col]

                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.2, random_state=42
                )

                custom_model = train_model_with_smote(X_train, y_train)
                predictions = custom_model.predict(features)
                probabilities = custom_model.predict_proba(features)[:, 1]

                df['Fraud_Probability'] = probabilities
                df['Prediction'] = predictions
                df['Prediction'] = df['Prediction'].map({1: 'Fraud', 0: 'Normal'})

                total = len(df)
                flagged = int((predictions == 1).sum())
                normal = int((predictions == 0).sum())

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Transactions", total)
                col2.metric("Flagged as Fraud", flagged)
                col3.metric("Normal Transactions", normal)

                fig = px.bar(x=['Normal', 'Fraud'], y=[normal, flagged],
                            color=['Normal', 'Fraud'], title='Transaction Distribution')
                st.plotly_chart(fig, use_container_width=True)

                flagged_df = df[df['Prediction'] == 'Fraud'].sort_values(
                    'Fraud_Probability', ascending=False)
                st.dataframe(flagged_df)

                st.download_button(
                    label="Download Flagged Transactions",
                    data=flagged_df.to_csv(index=False),
                    file_name="flagged_transactions.csv",
                    mime="text/csv"
                )
        st.stop()

    # Standard format - scale and predict
    scaler = StandardScaler()
    df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_Scaled'] = scaler.fit_transform(df[['Time']])
    df_clean = df.drop(['Amount', 'Time'], axis=1)

    if 'Class' in df_clean.columns:
        df_clean = df_clean.drop('Class', axis=1)

    model = xgb_model if model_choice == "XGBoost" else rf_model

    predictions = model.predict(df_clean)
    probabilities = model.predict_proba(df_clean)[:, 1]

    df['Fraud_Probability'] = probabilities
    df['Prediction'] = predictions
    df['Prediction'] = df['Prediction'].map({1: 'Fraud', 0: 'Normal'})

    total = len(df)
    flagged = int((predictions == 1).sum())
    normal = int((predictions == 0).sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", total)
    col2.metric("Flagged as Fraud", flagged)
    col3.metric("Normal Transactions", normal)

    st.divider()

    st.subheader("Fraud vs Normal Transactions")
    fig = px.bar(x=['Normal', 'Fraud'], y=[normal, flagged],
                color=['Normal', 'Fraud'], title='Transaction Distribution')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fraud Probability Distribution")
    fig2 = px.histogram(df, x='Fraud_Probability', nbins=50,
                        title='Distribution of Fraud Probabilities')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Flagged Transactions")
    flagged_df = df[df['Prediction'] == 'Fraud'].sort_values(
        'Fraud_Probability', ascending=False)
    st.dataframe(flagged_df)

    st.download_button(
        label="Download Flagged Transactions",
        data=flagged_df.to_csv(index=False),
        file_name="flagged_transactions.csv",
        mime="text/csv"
    )