# Employee Attrition Prediction App

## Overview

This project predicts whether an employee is likely to leave the organization based on their work environment, satisfaction, salary, and other factors. The app uses **XGBoost** as the predictive model and **Streamlit** as the interactive web interface.

The dataset used is the **IBM HR Analytics Employee Attrition dataset** from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).

---

## Folder Structure

```
EmployeeAttritionApp/
│
├── Employee-Attrition.csv       # Dataset file
├── train_xgb_model.py           # Script to train XGBoost model, save scaler and columns
├── app.py                       # Streamlit app
├── xgb_model.pkl                # Saved trained XGBoost model (generated after training)
├── scaler.pkl                   # Saved StandardScaler (generated after training)
└── model_columns.pkl            # List of columns used during training (generated after training)
```

---

## Requirements

* Python 3.8+
* Libraries:

  * pandas
  * numpy
  * scikit-learn
  * xgboost
  * imbalanced-learn
  * streamlit
  * matplotlib (optional)
  * seaborn (optional)

Install dependencies using:

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn streamlit matplotlib seaborn
```

---

## Step 1: Train the Model

1. Place the dataset `Employee-Attrition.csv` in the project folder.
2. Run the training script:

```bash
python train_xgb_model.py
```

This will:

* Train an XGBoost classifier using all numeric and categorical features.
* Handle class imbalance with SMOTE.
* Scale numeric features using `StandardScaler`.
* Perform hyperparameter tuning using `GridSearchCV`.
* Save:

  * `xgb_model.pkl` → trained model
  * `scaler.pkl` → feature scaler
  * `model_columns.pkl` → column order used for training

---

## Step 2: Run the Streamlit App

After training, run the app:

```bash
streamlit run app.py
```

* The app will open in your browser.
* Fill employee details in the input fields.
* Click **Predict Attrition**.
* The app will display:

  * **Attrition Probability**
  * Prediction result (likely to leave / likely to stay)

---

## Features Used

The app uses a combination of numeric and categorical features including:

* Age, DistanceFromHome, Education, EnvironmentSatisfaction, JobSatisfaction, MonthlyIncome
* BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, OverTime
* Additional numeric features from dataset are automatically filled with default values.

---

## Notes

* **All input features** are aligned with the training data to avoid column mismatch errors.
* Categorical variables are **label encoded** as per training mappings.
* Missing numeric features are automatically filled with **default values**.

---

## Future Improvements

* Pre-fill default or median values for all numeric features to simplify the UI.
* Display **top features influencing prediction** using XGBoost feature importance.
* Deploy the app online using **Streamlit Cloud** or **Heroku**.

---

This README ensures that anyone can set up, train, and run your Employee Attrition Prediction app without errors.

---
