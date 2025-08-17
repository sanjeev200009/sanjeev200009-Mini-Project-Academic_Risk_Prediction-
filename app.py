def show_about():
    st.title("📚 About: Academic Risk Prediction in Sri Lankan Schools")
    st.markdown("""
    ### Project Background
    Sri Lanka places high importance on educational outcomes, especially national-level exams like the G.C.E. O/L and A/L, which directly impact students' access to higher education and career opportunities. Many students face academic challenges such as absenteeism, insufficient study time, and socioeconomic limitations. Educators often detect underperformance only at the end of the academic year, by which point it is too late to intervene effectively.
    
    Recent advances in machine learning offer opportunities to support early intervention. By analyzing academic history, demographic details, and behavioral attributes, predictive models can flag students who are at risk of failing, allowing educators to provide timely assistance.
    
    ### Final Problem Statement
    While machine learning models have shown promise in predicting academic performance in international settings, their applicability in Sri Lankan school education remains underexplored. This project aims to develop a predictive model using available school data (e.g., midterm grades, attendance, demographics) to identify students at academic risk early and enable proactive intervention.
    
    #### Gap in Knowledge
    Most predictive models in education focus on datasets from Western contexts and higher education levels. There is a lack of predictive systems tailored to the Sri Lankan secondary school context.
    
    #### Justification
    Developing an ML-based early-warning system tailored to the Sri Lankan educational context can enable more timely academic interventions and reduce dropout rates.
    
    ### Research Questions
    1. Which features (attendance, study time, parental education, school type, etc.) most significantly affect academic risk among Sri Lankan school students?
    2. How accurately can machine learning models (e.g., Decision Tree, Random Forest, Logistic Regression) predict academic performance?
    3. What is the most interpretable and effective model that can be deployed in Sri Lankan schools?
    
    ### Research Objectives
    1. To identify and analyze the academic, demographic, and behavioral factors (e.g., midterm scores, attendance, parental education) that significantly influence students’ academic performance in Sri Lankan secondary schools.
    2. To develop and train multiple supervised machine learning models including Decision Tree, Random Forest, and Logistic Regression on a contextually relevant dataset to classify students based on their risk of academic underperformance.
    3. To evaluate and compare the predictive performance of the developed models using standard metrics (e.g., accuracy, precision, recall, F1-score), and to select the most suitable model for practical implementation in Sri Lankan schools.
    
    ### Data & Tools
    - **Primary Source:** UCI Student Performance Dataset (Cortez & Silva, 2008), which contains anonymized student-level academic and demographic data.
    - **Dataset Size:** Expected to consist of approximately 1000 records after preprocessing and augmentation.
    - **Adaptation Plan:** The dataset will be extended and locally contextualized by simulating Sri Lankan-specific attributes such as province, medium of instruction, and school type using domain knowledge and synthetic data generation techniques.
    - **Tools & Frameworks:** Python (Pandas, Scikit-learn, NumPy), Data Visualization (Matplotlib, Seaborn), Deployment (Streamlit).
    
    ### Group Members
    - SIVASUTHAKARAN SANJEEV (ITBNM-2211-0185)
    - VISHVALINGAM DESHANTH (ITBNM-2211-0121)
    - PIRAPAKARAN SAJEEVAN (ITBNM-2211-0183)
    - SIVASUBRAMANIYAM AINKARAN (ITBNM-2211-0103)
    - VIJAYAKUMAR MURALITHARAN (ITBNM-2211-0157)
    """)
import requests
import zipfile
import os
def show_data_understanding():
    st.title("🔬 Data Understanding: UCI Student Dataset")
    zip_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    zip_file_name = "student.zip"
    extract_path = "./student_data"
    math_file_path = os.path.join(extract_path, 'student-mat.csv')
    por_file_path = os.path.join(extract_path, 'student-por.csv')
    # Download and extract if not present
    if not (os.path.exists(math_file_path) and os.path.exists(por_file_path)):
        st.info(f"Downloading {zip_url}...")
        response = requests.get(zip_url)
        if response.status_code == 200:
            with open(zip_file_name, 'wb') as f:
                f.write(response.content)
            with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            st.success(f"Downloaded and extracted to {extract_path}")
        else:
            st.error(f"Error: Could not download the zip file. Status code: {response.status_code}")
            return
    # Load datasets
    math_df = pd.read_csv(math_file_path, sep=';')
    por_df = pd.read_csv(por_file_path, sep=';')
    st.write(f"Math dataset shape: {math_df.shape}")
    st.write(f"Portuguese dataset shape: {por_df.shape}")
    raw_df = pd.concat([math_df, por_df]).drop_duplicates(keep='first')
    st.write(f"Combined dataset shape: {raw_df.shape}")
    # Data overview
    st.subheader("=== Data Overview ===")
    st.write(f"Dataset shape: {raw_df.shape}")
    st.write("Data types:", raw_df.dtypes.value_counts())
    st.write("Missing values:", raw_df.isnull().sum())
    # Unique value counts for categorical columns
    st.subheader("=== Unique Values per Column (Categorical) ===")
    for col in raw_df.select_dtypes(exclude='number').columns:
        st.write(f"{col}: {raw_df[col].nunique()} unique values → {raw_df[col].unique()[:10]}")
    # Summary statistics
    st.subheader("=== Summary Statistics (Numerical Features) ===")
    st.dataframe(raw_df.describe().T)
    # Final grade distribution
    st.subheader("Final Grade (G3) Distribution")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(raw_df['G3'], bins=20, kde=True, color="skyblue", ax=ax1)
    ax1.set_title('Final Grade (G3) Distribution')
    ax1.set_xlabel('Final Grade')
    ax1.set_ylabel('Count')
    st.pyplot(fig1)
    # Correlation heatmap
    st.subheader("Correlation Heatmap (Numeric Features)")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    corr = raw_df.corr(numeric_only=True)
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax2)
    ax2.set_title("Correlation Heatmap (Numeric Features)")
    st.pyplot(fig2)
    # Top correlated features with final grade
    st.subheader("Features Most Correlated with Final Grade (G3)")
    st.write(corr['G3'].sort_values(ascending=False).head(10))
    # Boxplots: Study time, absences vs final grade
    st.subheader("Study Time & Absences vs Final Grade")
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(x='studytime', y='G3', data=raw_df, ax=axes3[0], palette="Set2")
    axes3[0].set_title("Study Time vs Final Grade")
    sns.boxplot(x='absences', y='G3', data=raw_df, ax=axes3[1], palette="Set3")
    axes3[1].set_title("Absences vs Final Grade")
    st.pyplot(fig3)
    # Outlier detection: Absences
    st.subheader("Outlier Detection - Absences")
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=raw_df['absences'], ax=ax4)
    ax4.set_title("Outlier Detection - Absences")
    st.pyplot(fig4)
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from src.utils import preprocess_input
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Academic Risk Prediction",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load trained model and preprocessing artifacts"""
    return {
        'model': joblib.load('models/best_model.joblib'),
        'scaler': joblib.load('models/scaler.joblib'),
        'label_encoders': joblib.load('models/label_encoders.joblib')
    }

def show_prediction():
    """Prediction Section"""
    st.title("🎓 Sri Lankan Academic Risk Prediction System")
    st.markdown("""
    **Predict students at risk of academic failure**  
    *A machine learning solution for educational institutions*
    """)

    # Load model
    artifacts = load_model()

    # Input form
    with st.sidebar:
        st.header("📝 Student Profile")
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 15, 22, 17)
            studytime = st.select_slider("Study Time (hrs/week)", [1, 2, 3, 4], 2)
            absences = st.slider("Absences", 0, 30, 5)
        with col2:
            g1 = st.slider("First Exam Score", 0, 20, 12)
            g2 = st.slider("Second Exam Score", 0, 20, 11)
            failures = st.slider("Past Failures", 0, 4, 0)

        st.subheader("Demographic Information")
        region = st.selectbox("Region", ["Urban", "Rural", "Estate"])
        ethnicity = st.selectbox("Ethnicity", ["Sinhala", "Tamil", "Muslim", "Other"])
        economic_status = st.selectbox("Economic Status", ["Low", "Middle", "High"])

        medu = st.slider("Mother's Education", 0, 4, 2, 
                        help="0: None, 1: Primary, 2: Middle, 3: Secondary, 4: Higher")
        fedu = st.slider("Father's Education", 0, 4, 2)

        predict_btn = st.button("Assess Academic Risk", type="primary", use_container_width=True)

    # Create input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'Medu': [medu],
        'Fedu': [fedu],
        'failures': [failures],
        'absences': [absences],
        'G1': [g1],
        'G2': [g2],
        'studytime': [studytime],
        'traveltime': [0],  # Placeholder
        'region': [region],
        'ethnicity': [ethnicity],
        'economic_status': [economic_status]
    })

    st.subheader("Student Information Summary")
    st.dataframe(input_data, hide_index=True)

    if predict_btn:
        processed_input = preprocess_input(
            input_data,
            artifacts['label_encoders'],
            artifacts['scaler']
        )

        model = artifacts['model']
        prediction = model.predict(processed_input)
        proba = model.predict_proba(processed_input)[0]

        st.subheader("📊 Risk Assessment Results")
        risk_status = "High Risk" if prediction[0] == 1 else "Low Risk"
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk Status", risk_status)
        with col2:
            st.metric("Probability", f"{proba[1]:.1%}")

        st.progress(int(proba[1] * 100), 
                   text=f"Risk Probability: {proba[1]:.1%}")

        st.subheader("🎯 Intervention Recommendations")
        if prediction[0] == 1:
            st.error("This student is at high risk of academic failure")
            st.markdown("""
            - **Academic Support**: Schedule weekly tutoring sessions
            - **Parental Engagement**: Arrange parent-teacher meeting
            - **Attendance Monitoring**: Track daily attendance
            - **Resource Allocation**: Provide study materials and resources
            - **Counseling**: Offer psychological support
            """)
        else:
            st.success("This student is performing adequately")
            st.markdown("""
            - **Maintain Progress**: Continue current support
            - **Goal Setting**: Set challenging academic targets
            - **Peer Support**: Encourage study groups
            - **Regular Check-ins**: Monthly academic reviews
            """)

def show_visualizations():
    st.title("📈 Model Visualizations & Reports")

    # Feature importance
    try:
        st.subheader("🔑 Feature Importance")
        st.image("reports/feature_importance.png", use_container_width=True)
    except:
        st.warning("Feature importance not available.")

    # Confusion matrix
    try:
        st.subheader("🧮 Confusion Matrix")
        st.image("reports/confusion_matrices.png", use_container_width=True)
    except:
        st.warning("Confusion matrix not available.")

    # ROC curve
    try:
        st.subheader("📉 ROC Curve")
        st.image("reports/roc_curves.png", use_container_width=True)
    except:
        st.warning("ROC curve not available.")

    # Grade distribution
    try:
        st.subheader("📊 Grade Distribution")
        st.image("reports/grade_distribution.png", use_container_width=True)
    except:
        st.warning("Grade distribution not available.")

    # Model performance
    try:
        st.subheader("📋 Model Performance Metrics")
        perf_df = pd.read_csv("reports/model_performance.csv")
        st.dataframe(perf_df, hide_index=True)
    except Exception as e:
        st.warning(f"Model performance not available. {e}")

    st.title("📈 Model Visualizations & Reports")

    # Feature importance
    try:
        st.subheader("🔑 Feature Importance")
        st.image("reports/feature_importance.png", use_container_width=True)
    except Exception as e:
        st.warning(f"Feature importance not available. {e}")

    # Confusion matrix
    try:
        st.subheader("🧮 Confusion Matrix")
        st.image("reports/confusion_matrices.png", use_container_width=True)
    except Exception as e:
        st.warning(f"Confusion matrix not available. {e}")

    # ROC curve
    try:
        st.subheader("📉 ROC Curve")
        st.image("reports/roc_curves.png", use_container_width=True)
    except Exception as e:
        st.warning(f"ROC curve not available. {e}")

    # Grade distribution
    try:
        st.subheader("📊 Grade Distribution")
        st.image("reports/grade_distribution.png", use_container_width=True)
    except Exception as e:
        st.warning(f"Grade distribution not available. {e}")

def show_data_exploration():
    st.title("🔍 Data Exploration & Feature Insights")
    st.markdown("""
    Explore the dataset and understand key factors influencing academic performance.
    Interact with the visualizations to discover patterns and relationships.
    """)
    
    # Load sample data (cached)
    @st.cache_data
    def load_data():
        return pd.read_csv("data/processed/training_data.csv")
    
    df = load_data()
    
    # Section 1: Dataset Overview
    st.header("📂 Dataset Overview")
    st.write(f"Dataset contains {df.shape[0]} student records with {df.shape[1]-1} features")
    
    if st.checkbox("Show raw data sample"):
        st.dataframe(df.sample(5, random_state=42))
    
    # Section 2: Feature Distribution Explorer
    st.header("📊 Feature Distribution Explorer")
    col1, col2 = st.columns(2)
    
    with col1:
        feature = st.selectbox("Select feature to visualize", 
                              options=['age', 'studytime', 'absences', 'G1', 'G2', 
                                       'failures', 'Medu', 'Fedu', 'traveltime'])
    with col2:
        plot_type = st.selectbox("Select visualization type", 
                                ["Histogram", "Box Plot", "Violin Plot"])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if plot_type == "Histogram":
        ax.hist(df[feature], bins=15, color='skyblue', edgecolor='black')
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
    elif plot_type == "Box Plot":
        ax.boxplot(df[feature], vert=False)
        ax.set_title(f"{feature} Distribution")
        ax.set_xlabel(feature)
    elif plot_type == "Violin Plot":
        ax.violinplot(df[feature], vert=False)
        ax.set_title(f"{feature} Density Distribution")
        ax.set_xlabel(feature)
    
    st.pyplot(fig)
    
    # Section 3: Feature Relationships
    st.header("🔗 Feature Relationships")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_feature = st.selectbox("X-axis Feature", 
                                options=['G1', 'G2', 'studytime', 'absences', 'age'])
    with col2:
        y_feature = st.selectbox("Y-axis Feature", 
                                options=['G2', 'G1', 'absences', 'failures', 'Medu'])
    with col3:
        hue_feature = st.selectbox("Color by", 
                                  ['risk', 'region', 'economic_status', 'ethnicity'], 
                                  index=0)
    
    rel_fig, rel_ax = plt.subplots(figsize=(10, 6))
    scatter = rel_ax.scatter(
        x=df[x_feature],
        y=df[y_feature],
        c=df[hue_feature].astype('category').cat.codes if hue_feature != 'risk' else df[hue_feature],
        cmap='viridis',
        alpha=0.7
    )
    rel_ax.set_xlabel(x_feature)
    rel_ax.set_ylabel(y_feature)
    rel_ax.set_title(f"{x_feature} vs {y_feature} colored by {hue_feature}")
    
    # Create legend for categorical coloring
    if hue_feature in ['region', 'economic_status', 'ethnicity', 'risk']:
        handles, labels = scatter.legend_elements()
        legend_labels = sorted(df[hue_feature].unique())
        rel_ax.legend(handles, legend_labels, title=hue_feature)
    
    st.pyplot(rel_fig)
    
    # Section 4: Risk Factor Analysis
    st.header("⚠️ Risk Factor Analysis")
    risk_factor = st.selectbox(
        "Analyze how this feature relates to academic risk",
        options=['studytime', 'absences', 'failures', 'Medu', 'Fedu', 'traveltime', 'region', 'economic_status']
    )
    
    fig2, ax2 = plt.subplots(1, 2, figsize=(15, 5))
    
    # Boxplot
    sns.boxplot(x='risk', y=risk_factor, data=df, ax=ax2[0])
    ax2[0].set_title(f"{risk_factor} vs Academic Risk")
    ax2[0].set_xlabel("Academic Risk")
    ax2[0].set_ylabel(risk_factor)
    
    # Countplot for categorical features
    if df[risk_factor].dtype == 'object' or df[risk_factor].nunique() < 5:
        sns.countplot(x=risk_factor, hue='risk', data=df, ax=ax2[1])
        ax2[1].set_title(f"Risk Distribution by {risk_factor}")
        ax2[1].set_xlabel(risk_factor)
        ax2[1].set_ylabel("Count")
        ax2[1].legend(title='Academic Risk', labels=['Low Risk', 'High Risk'])
    else:
        # Histogram for continuous features
        sns.histplot(data=df, x=risk_factor, hue='risk', kde=True, ax=ax2[1])
        ax2[1].set_title(f"Risk Distribution by {risk_factor}")
        ax2[1].set_xlabel(risk_factor)
        ax2[1].set_ylabel("Density")
        ax2[1].legend(title='Academic Risk', labels=['Low Risk', 'High Risk'])
    
    st.pyplot(fig2)
    
    # Section 5: Key Statistics
    st.header("📈 Key Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Academic Risk Distribution")
        risk_counts = df['risk'].value_counts()
        st.dataframe(risk_counts)
        
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.pie(risk_counts, labels=['Low Risk', 'High Risk'], autopct='%1.1f%%',
               colors=['#66b3ff','#ff9999'], startangle=90)
        ax3.set_title("Academic Risk Proportion")
        st.pyplot(fig3)
    
    with col2:
        st.subheader("Correlation with Academic Risk")
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()['risk'].sort_values(ascending=False)
        st.dataframe(corr.iloc[1:])  # Exclude risk itself
        
        st.subheader("Top Risk Indicators")
        st.markdown("""
        - **Past Failures**: Students with previous failures have 3x higher risk
        - **Low Exam Scores (G1, G2)**: Scores below 10 indicate high risk
        - **High Absences**: Students with >15 absences have 60% risk rate
        - **Low Parental Education**: Medu/Fedu ≤1 correlates with higher risk
        """)
 

def main():
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Go to:", ["About", "Prediction", "Visualizations", "Data Exploration", "Data Understanding"])

    if section == "About":
        show_about()
    elif section == "Prediction":
        show_prediction()
    elif section == "Visualizations":
        show_visualizations()
    elif section == "Data Exploration":
        show_data_exploration()
    elif section == "Data Understanding":
        show_data_understanding()
        
if __name__ == "__main__":
    main()
