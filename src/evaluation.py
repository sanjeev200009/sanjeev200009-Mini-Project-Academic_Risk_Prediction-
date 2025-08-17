import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

def evaluate_models_interpretation(model_results, X_test, y_test, X=None, best_model=None):
    """Advanced evaluation and interpretation of models (Colab style)"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    evaluation_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'])
    lw = 2
    # Create subplots for confusion matrices
    n_models = len(model_results)
    n_rows = int(np.ceil(n_models / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 6 * n_rows))
    axes = axes.flatten()
    plt.figure(1, figsize=(10, 8))
    for i, (name, result) in enumerate(model_results.items()):
        model = result.get('best_estimator', result.get('model'))
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
        # Add to results
        evaluation_df.loc[i] = [name, accuracy, precision, recall, f1, roc_auc]
        # ROC curve
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure(1)
            plt.plot(fpr, tpr, lw=lw, label=f'{name} (AUC = {roc_auc:.2f})')
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        # Classification report
        print(f"\n=== {name} Classification Report ===")
        print(classification_report(y_test, y_pred))
    # Plot ROC curves
    plt.figure(1)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('reports/roc_curves_interpretation.png')
    plt.close(1)
    # Save confusion matrices
    plt.tight_layout()
    fig.savefig('reports/confusion_matrices_interpretation.png')
    plt.close(fig)
    # Save evaluation results
    evaluation_df.to_csv('reports/model_evaluation_results.csv', index=False)
    # Feature importance for best model
    if best_model is not None and X is not None:
        if hasattr(best_model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
            feature_importances.nlargest(10).sort_values().plot(kind='barh')
            plt.title('Top 10 Feature Importances')
            plt.xlabel('Importance Score')
            plt.savefig('reports/feature_importance_interpretation.png')
            plt.close()
        elif isinstance(best_model, LogisticRegression):
            plt.figure(figsize=(10, 6))
            coefficients = pd.Series(best_model.coef_[0], index=X.columns)
            coefficients.sort_values().plot(kind='barh')
            plt.title('Feature Coefficients (Logistic Regression)')
            plt.xlabel('Coefficient Value')
            plt.savefig('reports/feature_coefficients_logreg.png')
            plt.close()
    return evaluation_df
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc

def evaluate_models(results, X_test, y_test):
    """Generate evaluation reports and visualizations"""
    # Performance comparison
    metrics_df = pd.DataFrame()
    for model, scores in results.items():
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'Model': model,
            'Accuracy': [scores['accuracy']],
            'Precision': [scores['precision']],
            'Recall': [scores['recall']],
            'F1-Score': [scores['f1']],
            'ROC AUC': [scores['roc_auc']]
        })], ignore_index=True)
    
    metrics_df.to_csv('reports/model_performance.csv', index=False)
    
    # Confusion matrices
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    # If only one model, axes is not a list, so wrap it
    if len(results) == 1:
        axes = [axes]
    for i, (model, scores) in enumerate(results.items()):
        cm = confusion_matrix(y_test, scores['model'].predict(X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{model} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrices.png')
    
    # ROC curves
    plt.figure(figsize=(10, 8))
    for model, scores in results.items():
        fpr, tpr, _ = roc_curve(y_test, scores['model'].predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.savefig('reports/roc_curves.png')
    
    # Feature importance for best model
    best_model = max(results, key=lambda x: results[x]['f1'])
    if hasattr(results[best_model]['model'], 'feature_importances_'):
        importances = results[best_model]['model'].feature_importances_
        features = X_test.columns if hasattr(X_test, 'columns') else range(len(importances))
        fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        fi_df = fi_df.sort_values('Importance', ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=fi_df)
        plt.title('Top 10 Important Features')
        plt.savefig('reports/feature_importance.png')
    
    return metrics_df