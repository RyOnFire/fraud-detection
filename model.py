from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
from data import load_data


def train_model_with_xgboost(features, labels):
    scale_pos_weight = len(labels[labels == 0]) / len(labels[labels == 1])
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_estimators=100
    )
    model.fit(features, labels)
    return model


def train_model_with_smote(features, labels):
    smote = SMOTE(random_state=42)
    features_balanced, labels_balanced = smote.fit_resample(features, labels)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features_balanced, labels_balanced)
    return model


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    labels = df['Class']
    features = df.drop('Class', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    print("Training XGBoost...")
    xgb_model = train_model_with_xgboost(X_train, y_train)
    print("XGBoost Results:")
    print(classification_report(y_test, xgb_model.predict(X_test)))

    print("Training Random Forest with SMOTE...")
    rf_model = train_model_with_smote(X_train, y_train)
    print("Random Forest Results:")
    print(classification_report(y_test, rf_model.predict(X_test)))

    joblib.dump(xgb_model, 'xgb_model.pkl')
    joblib.dump(rf_model, 'rf_model.pkl')
    print("Models saved!")