import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data():
    """Load and combine datasets from UCI repository"""
    math_df = pd.read_csv('data/raw/student-mat.csv', sep=';')
    portuguese_df = pd.read_csv('data/raw/student-por.csv', sep=';')
    return pd.concat([math_df, portuguese_df]).drop_duplicates(keep='first')

def explore_data(df):
    """Perform exploratory data analysis"""
    # Initial overview
    print("Dataset shape:", df.shape)
    print("\nData types:\n", df.dtypes.value_counts())
    print("\nMissing values:\n", df.isnull().sum())
    
    # Summary statistics
    print("\nSummary statistics:\n", df.describe())
    
    # Visualizations
    plt.figure(figsize=(10, 6))
    sns.countplot(x='G3', data=df)
    plt.title('Final Grade Distribution')
    plt.savefig('reports/grade_distribution.png')
    
    return df

def sri_lankan_context(df):
    """Add Sri Lankan-specific features"""
    np.random.seed(42)
    regions = ['Urban', 'Rural', 'Estate']
    ethnicities = ['Sinhala', 'Tamil', 'Muslim', 'Other']
    economic_status = ['Low', 'Middle', 'High']
    
    df['region'] = np.random.choice(regions, size=len(df), p=[0.18, 0.77, 0.05])
    df['ethnicity'] = np.random.choice(ethnicities, size=len(df), p=[0.75, 0.11, 0.09, 0.05])
    df['economic_status'] = np.random.choice(economic_status, size=len(df), p=[0.45, 0.50, 0.05])
    df['academic_risk'] = (df['G3'] < 10).astype(int)
    
    return df

def preprocess_data(df):
    """Clean and transform data"""
    # Handle missing values (median for numerical, mode for categorical)
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns
    
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Feature engineering
    df['total_study_time'] = df['studytime'] + df['traveltime']
    df['parent_education'] = (df['Medu'] + df['Fedu']) / 2
    
    # Encode categorical features
    label_encoders = {}
    cat_cols = ['region', 'ethnicity', 'economic_status']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Feature selection
    features = [
        'age', 'Medu', 'Fedu', 'failures', 'absences', 
        'G1', 'G2', 'total_study_time', 'parent_education',
        'region', 'ethnicity', 'economic_status'
    ]
    
    X = df[features]
    y = df['academic_risk']
    
    return X, y, label_encoders