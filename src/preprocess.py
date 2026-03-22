import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def clean_dependents(df):
    """Convert '3+' in Dependents to 3 and fill missing values."""
    df = df.copy()
    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].replace('3+', '3')
        # Fill missing values with mode
        mode_val = df['Dependents'].mode()[0] if not df['Dependents'].mode().empty else '0'
        df['Dependents'] = df['Dependents'].fillna(mode_val).astype(float)
    return df

def get_preprocessor():
    """Returns a scikit-learn ColumnTransformer for loan data."""
    
    # Define feature groups
    num_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Dependents']
    cat_features = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']

    # Numerical pipeline: Impute median then scale
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Impute mode then OneHot (handle unknown categories)
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features)
        ]
    )
    
    return preprocessor
