from sklearn.ensemble import IsolationForest
from data import load_data
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def train_model(df, contamination=0.00172):
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(df)
    return model

def predict(model, df):
    predictions = model.predict(df)
    return predictions

print("Loading data...")
df = load_data()
labels = df['Class']
df = df.drop('Class', axis=1)

print("Training model...")
model = train_model(df)

print("Making predictions...")
predictions = predict(model, df)
print(predictions)

print("Total flagged as fraud:", np.sum(predictions == -1))
print("Total normal:", np.sum(predictions == 1))

# Convert predictions from -1/1 to 0/1 to match our Class column
predictions_converted = [1 if p == -1 else 0 for p in predictions]

print("Confusion Matrix:")
print(confusion_matrix(labels, predictions_converted))

print("\nClassification Report:")
print(classification_report(labels, predictions_converted))