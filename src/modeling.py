from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from joblib import dump
import time

def train_models(X, y):
    """Train and tune multiple models with evaluation"""
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Define models and hyperparameters
    models = {
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        }
    }
    
    results = {}
    best_score = 0
    best_model = None
    
    # Train and tune models
    for name, config in models.items():
        print(f"\n=== Training {name} ===")
        start_time = time.time()
        
        grid = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Evaluate
        y_pred = grid.predict(X_test)
        y_proba = grid.predict_proba(X_test)[:, 1] if hasattr(grid.best_estimator_, "predict_proba") else None
        
        results[name] = {
            'best_params': grid.best_params_,
            'cv_best_f1': grid.best_score_,  # CV score
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
            'training_time': training_time,
            'model': grid.best_estimator_
        }
        
        # Track best model
        if results[name]['f1'] > best_score:
            best_score = results[name]['f1']
            best_model = grid.best_estimator_
    
    # Save best model and scaler
    dump(best_model, 'models/best_model.joblib')
    dump(scaler, 'models/scaler.joblib')
    
    return results, best_model, X_test, y_test
