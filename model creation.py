# ==========================================
# FINANCIAL INVESTMENT ADVISOR ML MODEL
# Train + Save Model for Streamlit App
# ==========================================

# ========= IMPORT LIBRARIES =========
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ========= CREATE DATASET =========

# data = {
#     'Age': [
#         22, 25, 27, 30, 35, 40, 45, 50, 55, 60,
#         23, 28, 32, 38, 42, 48, 52, 58, 26, 33
#     ],

#     'Goal': [
#         2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
#         2, 2, 1, 1, 1, 1, 1, 1, 2, 2
#     ],

#     'Savings': [
#         25000, 30000, 35000, 20000, 15000,
#         12000, 10000, 9000, 8000, 7000,
#         40000, 45000, 18000, 16000, 14000,
#         11000, 9500, 8500, 50000, 28000
#     ],

#     # 0 = Low Risk
#     # 1 = Medium Risk
#     # 2 = High Risk

#     'RiskProfile': [
#         2, 2, 2, 1, 1,
#         1, 0, 0, 0, 0,
#         2, 2, 1, 1, 1,
#         0, 0, 0, 2, 1
#     ]
# }

# df = pd.DataFrame(data)

df=pd.read_csv('financial_dummy_dataset.csv')

# ========= DATA CLEANING =========

print("\nChecking Missing Values:\n")
print(df.isnull().sum())

print("\nChecking Duplicate Rows:\n")
print(df.duplicated().sum())

# ========= FEATURES & TARGET =========

X = df[['Age', 'Goal', 'Savings']]
y = df['RiskProfile']

# ========= TRAIN TEST SPLIT =========

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ========= FEATURE SCALING =========

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========= TRAIN KNN MODEL =========

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train_scaled, y_train)

# ========= MODEL EVALUATION =========

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# ========= SAVE MODEL & SCALER =========

joblib.dump(model, "financial_advisor_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved successfully!")
print("Files created:")
print("1. financial_advisor_model.pkl")
print("2. scaler.pkl")

# ========= TEST PREDICTION =========

# Example User Input
# Age = 26
# Goal = 2 (Long-Term)
# Savings = 30000

sample_input = [[26, 2, 30000]]

# Scale Input
sample_input_scaled = scaler.transform(sample_input)

# Predict
prediction = model.predict(sample_input_scaled)[0]

# ========= RISK LABEL =========

risk_labels = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}

print("\nPrediction Result:")
print("Risk Profile:", risk_labels[prediction])

# ========= INVESTMENT RECOMMENDATION =========

def get_recommendation(profile_code):

    if profile_code == 0:
        return """
Low Risk Investments:
- Fixed Deposits
- Government Bonds
- PPF
- Savings Schemes
"""

    elif profile_code == 1:
        return """
Medium Risk Investments:
- Mutual Funds
- Index Funds
- Hybrid Funds
"""

    else:
        return """
High Risk Investments:
- Stocks
- Equity Funds
- Cryptocurrency
- High Growth Investments
"""

print(get_recommendation(prediction))