import pandas as pd
import numpy as np

def preprocess_input(input_df, label_encoders, scaler):
    """Preprocess user input for prediction"""
    # Feature engineering
    input_df['total_study_time'] = input_df['studytime'] + 0  # Placeholder for traveltime
    input_df['parent_education'] = (input_df['Medu'] + input_df['Fedu']) / 2
    
    # Encode categorical features
    for col in ['region', 'ethnicity', 'economic_status']:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])
    
    # Select and order features
    features = [
        'age', 'Medu', 'Fedu', 'failures', 'absences', 
        'G1', 'G2', 'total_study_time', 'parent_education',
        'region', 'ethnicity', 'economic_status'
    ]
    
    # Scale features
    scaled = scaler.transform(input_df[features])
    return pd.DataFrame(scaled, columns=features)