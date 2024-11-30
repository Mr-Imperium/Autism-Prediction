import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def load_model():
    models = {
        'Logistic Regression': joblib.load('logistic_regression_model.joblib'),
        'XGBoost': joblib.load('xgboost_model.joblib'),
        'SVM': joblib.load('svm_model.joblib')
    }
    feature_columns = joblib.load('feature_columns.joblib')
    scaler = joblib.load('feature_scaler.joblib')
    return models, feature_columns, scaler

def preprocess_input(input_data, feature_columns, scaler):
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])
    
    # Create a full DataFrame with all expected columns
    full_df = pd.DataFrame(columns=feature_columns)
    
    # Copy input data to full DataFrame
    for col in input_data.keys():
        if col in feature_columns:
            full_df[col] = input_data[col]
    
    # Fill any missing columns with 0
    full_df = full_df.fillna(0)
    
    # Perform necessary transformations
    # Log transform age
    full_df['age'] = np.log(full_df['age'] + 1)
