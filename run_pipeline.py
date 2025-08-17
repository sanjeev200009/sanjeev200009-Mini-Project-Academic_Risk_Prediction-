from src.data_processing import load_data, explore_data, sri_lankan_context, preprocess_data
from src.modeling import train_models
from src.evaluation import evaluate_models
import joblib

def main(): 
    import os  
    import joblib

    print("🚀 Starting Academic Risk Prediction Pipeline") 
     
    # Data understanding 
    print("\n🔍 Loading and exploring data...") 
    df = load_data() 
    df = explore_data(df) 
     
    # Sri Lankan context 
    print("\n🇱🇰 Adding Sri Lankan features...") 
    df = sri_lankan_context(df) 
     
    # Preprocessing 
    print("\n⚙️ Preprocessing data...") 
    X, y, label_encoders = preprocess_data(df) 
    joblib.dump(label_encoders, 'models/label_encoders.joblib') 

    # Save actual training data for Streamlit exploration
    os.makedirs("data/processed", exist_ok=True)
    df['risk'] = y  # add the target column to dataframe
    df.to_csv("data/processed/training_data.csv", index=False)
    print("✅ training_data.csv saved successfully!")
     
    # Modeling 
    print("\n🤖 Training models...") 
    results, _, X_test, y_test = train_models(X, y) 
     
    # Evaluation 
    print("\n📊 Evaluating models...") 
    metrics_df = evaluate_models(results, X_test, y_test) 
    print("\nModel Performance:\n", metrics_df) 
     
    print("\n✅ Pipeline completed successfully!") 

if __name__ == "__main__":
    main()