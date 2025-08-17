# 🎓 Sri Lankan Academic Risk Prediction System

A **Streamlit web application** for predicting students at risk of academic failure in Sri Lankan schools. The system uses machine learning to identify high-risk students and provides actionable recommendations for improving academic performance.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Folder Structure](#folder-structure)  
- [Model & Data](#model--data)  
- [Visualization & Reporting](#visualization--reporting)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Project Overview

This project helps educational institutions identify students who may be at high risk of academic failure based on various demographic, academic, and behavioral features. The application allows teachers and administrators to:

- Predict student risk using a trained machine learning model
- Explore key data trends and insights
- Provide actionable recommendations for high-risk students
- Visualize model performance and feature importance

The system is tailored for **Sri Lankan students**, using relevant demographic and educational features.

---

## Features

### 1. Prediction
- Predict **High Risk** or **Low Risk** for students.
- Displays **probability of risk**.
- Provides actionable **intervention recommendations**.
- Inputs include:
  - Age, Study Time, Absences, Exam Scores (G1, G2)
  - Past Failures, Parental Education, Region, Ethnicity, Economic Status

### 2. Visualizations
- **Feature Importance**
- **Confusion Matrix**
- **ROC Curve**
- **Grade Distribution**
- **Model Performance Metrics**

### 3. Data Exploration
- Dataset overview & sample records
- Feature distribution explorer (Histogram, Box Plot, Violin Plot)
- Feature relationships (scatter plots by categorical variables)
- Risk factor analysis
- Key statistics & correlation with academic risk

---

## Installation

1. **Clone the repository**
```bash
git https://github.com/sanjeev200009/sanjeev200009-Mini-Project-Academic_Risk_Prediction-.git
cd academic-risk-prediction
