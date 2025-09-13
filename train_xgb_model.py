import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle

# Load dataset
df = pd.read_csv("Employee-Attrition.csv")

# Drop redundant columns
df = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'])

# Encode target
df['Attrition'] = df['Attrition'].map({'Yes':1,'No':0})

# Encode categorical features
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Split features and target
X = df_encoded.drop(columns=['Attrition'])
y = df_encoded['Attrition']

# Save feature column order
feature_columns = X.columns.tolist()

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Feature scaling
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# XGBoost GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_res, y_train_res)
xgb_best = grid_search.best_estimator_

print("Best parameters:", grid_search.best_params_)

# Save model, scaler, and columns
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_best, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

print("Model, scaler, and columns saved successfully!")
