from joblib import dump, load
def enhanced_correlation_feature_selection(processed_df):
    """Enhanced correlation analysis, feature selection, and scaling."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    # Dimensionality reduction - Enhanced correlation analysis
    plt.figure(figsize=(14, 12))
    corr_matrix = processed_df.corr(numeric_only=True)
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    # Improved heatmap with clustering
    sns.clustermap(
        corr_matrix,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        figsize=(14, 12),
        row_cluster=True,
        col_cluster=True,
        mask=mask
    )
    plt.title('Hierarchically Clustered Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('reports/clustered_correlation_matrix.png')
    plt.close()
    # Select features with highest correlation to academic risk
    if 'academic_risk' in processed_df.columns:
        risk_corr = corr_matrix['academic_risk'].abs().sort_values(ascending=False)[1:]  # Exclude self-correlation
        print("\nTop Features Correlated with Academic Risk:")
        print(risk_corr.head(10))
        # Feature selection based on multicollinearity
        selected_features = []
        high_corr_pairs = set()
        threshold = 0.7
        for feature in risk_corr.index:
            if feature in corr_matrix.columns:
                if feature not in selected_features:
                    redundant = False
                    for selected in selected_features:
                        if selected in corr_matrix.columns:
                            corr_val = abs(corr_matrix.loc[feature, selected])
                            if corr_val > threshold:
                                high_corr_pairs.add((feature, selected, corr_val))
                                redundant = True
                                break
                    if not redundant:
                        selected_features.append(feature)
        print(f"\nSelected {len(selected_features)} features after multicollinearity check:")
        print(selected_features)
        if high_corr_pairs:
            print("\nExcluded features due to high correlation (>0.7):")
            for pair in high_corr_pairs:
                print(f"{pair[0]} - {pair[1]} (r={pair[2]:.2f})")
        # Feature scaling with selected features
        try:
            X_selected = processed_df[selected_features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)
            dump(scaler, 'models/scaler.joblib')
            dump(selected_features, 'models/selected_features.joblib')
            print("\nScaler and feature list saved for future inference")
            return X_scaled, selected_features, scaler
        except KeyError as e:
            print(f"\nError during feature scaling: {e}")
            print("Please ensure the selected features exist in the 'processed_df' DataFrame.")
            return None, None, None
    else:
        print("Error: 'academic_risk' column not found in processed_df. Cannot perform correlation analysis.")
        print("Please ensure the preprocessing step was run successfully.")
        return None, None, None
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