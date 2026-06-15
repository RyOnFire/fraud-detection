from sklearn.ensemble import IsolationForest, RandomForestClassifier
from data import load_data, load_new_data
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
import joblib



def train_new_models(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # XGBoost
    print("Training XGBoost on new data...")
    xgb = XGBClassifier(
        scale_pos_weight=len(labels[labels==0])/len(labels[labels==1]),
        n_estimators=100,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    print("XGBoost Results:")
    print(classification_report(y_test, xgb_preds))
    
    # Random Forest with SMOTE
    print("Training Random Forest on new data...")
    rf_model = train_model_with_smote(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    print("Random Forest Results:")
    print(classification_report(y_test, rf_preds))
    
    return xgb, rf_model






def train_ensemble(features, labels):
    # Apply SMOTE first
    smote = SMOTE(random_state=42)
    features_balanced, labels_balanced = smote.fit_resample(features, labels)
    
    # Define individual models
    rf = RandomForestClassifier(
        n_estimators=50, 
        min_samples_split=5,
        random_state=42
    )
    xgb = XGBClassifier(
        scale_pos_weight=len(labels[labels==0])/len(labels[labels==1]),
        n_estimators=100,
        random_state=42
    )
    
    # Combine into voting ensemble
    ensemble = VotingClassifier(
        estimators=[('random_forest', rf), ('xgboost', xgb)],
        voting='soft'  # uses probabilities not just votes
    )
    
    ensemble.fit(features_balanced, labels_balanced)
    return ensemble

def tune_random_forest(features, labels):
    params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, params, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(features, labels)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best F1 score:", grid_search.best_score_)
    
    return grid_search.best_estimator_

def train_model_with_xgboost(features, labels):
    # Calculate class imbalance ratio
    scale_pos_weight = len(labels[labels == 0]) / len(labels[labels == 1])
    
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_estimators=100
    )
    model.fit(features, labels)
    return model

def train_model_with_smote(features, labels):
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    features_balanced, labels_balanced = smote.fit_resample(features, labels)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features_balanced, labels_balanced)
    
    return model

def train_model(df, contamination=0.00172):
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(df)
    return model

def predict(model, df):
    predictions = model.predict(df)
    return predictions




print("Loading new data...")
new_df = load_new_data()
new_labels = new_df['fraud']
new_features = new_df.drop('fraud', axis=1)

xgb_new, rf_new = train_new_models(new_features, new_labels)

# Save both models here
joblib.dump(xgb_new, 'xgb_model.pkl')
joblib.dump(rf_new, 'rf_model.pkl')
print("Models saved!")

#print("Loading data...")
#df = load_data()
#labels = df['Class']
#features = df.drop('Class', axis=1)




# X_train, X_test, y_train, y_test = train_test_split(
#     features, labels, test_size=0.2, random_state=42
# )
# print("Training XGBoost...")
# xgb_model = train_model_with_xgboost(X_train, y_train)
# xgb_predictions = xgb_model.predict(X_test)
# print("XGBoost Results:")
# print(classification_report(y_test, xgb_predictions))

# print("Training Random Forest with SMOTE...")
# model = train_model_with_smote(X_train, y_train)
# predictions = model.predict(X_test)
# print(classification_report(y_test, predictions))

# print("Tuning Random Forest...")
# best_rf = tune_random_forest(X_train, y_train)
# predictions = best_rf.predict(X_test)
# print("Tuned Random Forest Results:")
# print(classification_report(y_test, predictions))

# print("Training Ensemble...")
# ensemble = train_ensemble(X_train, y_train)
# ensemble_predictions = ensemble.predict(X_test)
# print("Ensemble Results:")
# print(classification_report(y_test, ensemble_predictions))



#print("Training model...")
#model = train_model(df)

#print("Making predictions...")
#predictions = predict(model, df)
#print(predictions)

#print("Total flagged as fraud:", np.sum(predictions == -1))
#print("Total normal:", np.sum(predictions == 1))

# Convert predictions from -1/1 to 0/1 to match our Class column
#predictions_converted = [1 if p == -1 else 0 for p in predictions]

#print("Confusion Matrix:")
#print(confusion_matrix(labels, predictions_converted))

#print("\nClassification Report:")
#print(classification_report(labels, predictions_converted))