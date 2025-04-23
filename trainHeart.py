import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# Load and preprocess data
heart = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Define feature columns
feature_columns = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
    'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium',
    'sex', 'smoking', 'time'
]

# Define ranges for input validation based on dataset
feature_ranges = {
    'age': (40.0, 95.0),
    'anaemia': (0, 1),
    'creatinine_phosphokinase': (23, 7861),
    'diabetes': (0, 1),
    'ejection_fraction': (14, 80),
    'high_blood_pressure': (0, 1),
    'platelets': (25100.0, 850000.0),
    'serum_creatinine': (0.5, 9.4),
    'serum_sodium': (113, 148),
    'sex': (0, 1),
    'smoking': (0, 1),
    'time': (4, 285)
}

# Feature scaling
scaler = StandardScaler()
X = heart[feature_columns]
y = heart['DEATH_EVENT']
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Save model and scaler
joblib.dump(model, 'heart_failure_model.pkl')
joblib.dump(scaler, 'scaler.pkl')