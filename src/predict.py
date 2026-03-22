import joblib
import pandas as pd
from pathlib import Path

def load_pipeline():
    """Load the trained ML pipeline."""
    model_path = Path(__file__).resolve().parents[1] / "models" / "loan_model.pkl"
    return joblib.load(model_path)

def predict_risk(input_df, pipeline=None):
    """
    Predict loan risk and probabilities.
    Args:
        input_df: pd.DataFrame with applicant details.
        pipeline: Pre-loaded pipeline (optional).
    Returns:
        prediction (int), approval_probability (float), risk_probability (float)
    """
    if pipeline is None:
        pipeline = load_pipeline()
    
    # Pre-cleaning for Dependents (must match training logic)
    from src.preprocess import clean_dependents
    input_df = clean_dependents(input_df)
    
    prediction = pipeline.predict(input_df)[0]
    probabilities = pipeline.predict_proba(input_df)[0]
    
    # 1 is Approve (Low Risk), 0 is Reject (High Risk) in our training mapping
    approval_probability = float(probabilities[1])
    risk_probability = float(probabilities[0])
    
    return prediction, approval_probability, risk_probability
