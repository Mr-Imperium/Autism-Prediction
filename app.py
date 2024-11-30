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
    # Create a dictionary to store processed input
    processed_input = {}
    
    # Explicitly handle each expected feature column
    for col in feature_columns:
        if col.startswith('A') and col.endswith('_Score'):
            # Handle score columns
            processed_input[col] = input_data.get(col, 0)
        elif col == 'age':
            processed_input[col] = np.log(float(input_data.get('age', 25.0)) + 1)
        elif col == 'gender':
            processed_input[col] = input_data.get('gender', 'unknown')
        elif col == 'jaundice':
            processed_input[col] = input_data.get('jaundice', 'no')
        elif col == 'austim':
            processed_input[col] = input_data.get('austim', 'no')
        elif col == 'used_app_before':
            processed_input[col] = input_data.get('used_app_before', 'no')
        elif col == 'ethnicity':
            processed_input[col] = input_data.get('ethnicity', 'unknown')
        elif col == 'contry_of_res':
            processed_input[col] = input_data.get('contry_of_res', 'unknown')
        elif col == 'relation':
            processed_input[col] = input_data.get('relation', 'unknown')
        elif col == 'result':
            processed_input[col] = float(input_data.get('result', 0.0))
        elif col == 'ageGroup':
            processed_input[col] = input_data.get('ageGroup', 'Young')
        elif col == 'sum_score':
            processed_input[col] = int(input_data.get('sum_score', 0))
        elif col == 'ind':
            processed_input[col] = int(input_data.get('ind', 0))
        else:
            # Default for any unexpected columns
            processed_input[col] = 0
    
    # Convert to DataFrame
    full_df = pd.DataFrame([processed_input])
    
    # Ensure correct column order
    full_df = full_df[feature_columns]
    
    # Label encoding for categorical variables
    categorical_columns = full_df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        full_df[col] = le.fit_transform(full_df[col].astype(str))
    
    # Impute and scale
    try:
        imputer = SimpleImputer(strategy='mean')
        df_imputed = imputer.fit_transform(full_df)
        df_scaled = scaler.transform(df_imputed)
        return df_scaled
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        st.error("Input data: " + str(input_data))
        st.error("Processed DataFrame: " + str(full_df))
        st.error("Feature Columns: " + str(feature_columns))
        raise

def main():
    st.title('Autism Spectrum Disorder Prediction')
    
    # Sidebar for model selection
    st.sidebar.header('Model Selection')
    model_choice = st.sidebar.selectbox(
        'Choose a Model',
        ['Logistic Regression', 'XGBoost', 'SVM']
    )
    
    # Load models
    models, feature_columns, scaler = load_model()
    
    # Create input fields dynamically based on feature columns
    input_data = {}
    
    # Divide inputs into columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Numeric inputs
        input_data['age'] = st.number_input('Age', min_value=0.0, max_value=100.0, value=25.0)
        input_data['result'] = st.number_input('Result', min_value=-5.0, max_value=100.0, value=0.0)
        
        # Score inputs
        for i in range(1, 6):
            input_data[f'A{i}_Score'] = st.number_input(f'A{i} Score', min_value=0, max_value=10, value=0)
    
    with col2:
        # Categorical inputs
        input_data['gender'] = st.selectbox('Gender', ['male', 'female'])
        input_data['jaundice'] = st.selectbox('Jaundice', ['yes', 'no'])
        input_data['austim'] = st.selectbox('Autism', ['yes', 'no'])
        input_data['used_app_before'] = st.selectbox('Used App Before', ['yes', 'no'])
        
        # Remaining score inputs
        for i in range(6, 11):
            input_data[f'A{i}_Score'] = st.number_input(f'A{i} Score', min_value=0, max_value=10, value=0)
    
    # Additional feature engineering
    input_data['sum_score'] = sum(input_data[f'A{i}_Score'] for i in range(1, 11))
    input_data['ind'] = (1 if input_data['austim'] == 'yes' else 0) + \
                        (1 if input_data['used_app_before'] == 'yes' else 0) + \
                        (1 if input_data['jaundice'] == 'yes' else 0)
    
    # Add ageGroup feature
    def convertAge(age):
        if age < 4:
            return 'Toddler'
        elif age < 12:
            return 'Kid'
        elif age < 18:
            return 'Teenager'
        elif age < 40:
            return 'Young'
        else:
            return 'Senior'
    
    input_data['ageGroup'] = convertAge(input_data['age'])
    
    # Prediction button
    if st.button('Predict'):
        # Preprocess input
        processed_input = preprocess_input(input_data, feature_columns, scaler)
        
        # Make prediction
        model_pipeline = models[model_choice]
        prediction = model_pipeline.predict(processed_input)
        prediction_proba = model_pipeline.predict_proba(processed_input)
        
        # Display results
        st.subheader('Prediction Results')
        if prediction[0] == 1:
            st.error('Potential Autism Spectrum Disorder Detected')
        else:
            st.success('No Autism Spectrum Disorder Detected')
        
        # Probability display
        st.write(f'Probability of ASD: {prediction_proba[0][1]:.2%}')
        st.write(f'Model Used: {model_choice}')

if __name__ == '__main__':
    main()
