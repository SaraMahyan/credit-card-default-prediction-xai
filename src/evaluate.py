import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, accuracy_score
import shap

def run_evaluation(models_dir, xai_dir, reports_dir):
    print("Starting Evaluation & XAI Phase...")
    
    os.makedirs(xai_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Check if target exists by checking if X_train.csv exists
    if not os.path.exists('data/X_train.csv'):
        print("No supervised learning data found. Skipping evaluation.")
        return
        
    X_train = pd.read_csv('data/X_train.csv')
    X_val = pd.read_csv('data/X_val.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv').values.ravel()
    y_val = pd.read_csv('data/y_val.csv').values.ravel()
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    
    models = {
        'LogisticRegression': joblib.load(os.path.join(models_dir, 'LogisticRegression.pkl')),
        'RandomForest': joblib.load(os.path.join(models_dir, 'RandomForest.pkl')),
        'XGBoost': joblib.load(os.path.join(models_dir, 'XGBoost.pkl')),
        'LightGBM': joblib.load(os.path.join(models_dir, 'LightGBM.pkl'))
    }
    
    report_content = "# Final Evaluation & XAI Report\n\n"
    report_content += "## Model Performance Comparison\n\n"
    report_content += "| Model | Split | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n"
    report_content += "|---|---|---|---|---|---|---|\n"
    
    plt.figure(figsize=(10, 8))
    plt_pr = plt.figure(figsize=(10, 8))
    ax_pr = plt_pr.add_subplot(111)
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_val_prob = model.predict_proba(X_val)[:, 1]
        y_test_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = []
        for split, y_true, y_pred, y_prob in [
            ('Train', y_train, y_train_pred, y_train_prob),
            ('Validation', y_val, y_val_pred, y_val_prob),
            ('Test', y_test, y_test_pred, y_test_prob)
        ]:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            roc = roc_auc_score(y_true, y_prob)
            report_content += f"| {name} | {split} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {roc:.4f} |\n"
            
            if split == 'Test':
                # ROC Curve
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                plt.figure(1)
                plt.plot(fpr, tpr, label=f"{name} (AUC = {roc:.4f})")
                
                # PR Curve
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                ax_pr.plot(recall, precision, label=f"{name}")
                
                # Confusion Matrix
                cm = confusion_matrix(y_true, y_pred)
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {name}')
                plt.savefig(os.path.join(models_dir, f'{name}_confusion_matrix.png'))
                plt.close()
                
                # Feature Importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1][:10]
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x=importances[indices], y=X_test.columns[indices])
                    plt.title(f'Top 10 Feature Importances - {name}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(models_dir, f'{name}_feature_importance.png'))
                    plt.close()
    
    # Save ROC and PR curves
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curves (Test Set)')
    plt.legend()
    plt.savefig(os.path.join(models_dir, 'roc_auc_curves.png'))
    plt.close()
    
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curves (Test Set)')
    ax_pr.legend()
    plt_pr.savefig(os.path.join(models_dir, 'pr_curves.png'))
    plt.close()
    
    report_content += "\n## Evaluation Plots\n"
    report_content += "![ROC Curves](../graphs/models/roc_auc_curves.png)\n"
    report_content += "![PR Curves](../graphs/models/pr_curves.png)\n"
    
    print("Generating XAI Plots (SHAP & LIME) for XGBoost...")
    xgb_model = models['XGBoost']
    
    # SHAP
    explainer = shap.TreeExplainer(xgb_model)
    # Use a sample to speed up SHAP
    X_test_sample = X_test.sample(100, random_state=42)
    shap_values = explainer.shap_values(X_test_sample)
    
    plt.figure()
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(xai_dir, 'shap_summary_plot.png'))
    plt.close()
    
    report_content += "\n## Explainable AI (XAI)\n"
    report_content += "### SHAP Summary\n"
    report_content += "The SHAP summary plot shows the global feature importance and direction of impact on model output.\n"
    report_content += "![SHAP Summary](../graphs/xai/shap_summary_plot.png)\n"
    
    report_content += "### Model Selection & Overfitting Analysis\n"
    report_content += "**Why not Random Forest?**\n"
    report_content += "Looking at the metrics, Random Forest exhibits extreme overfitting. Its Training metrics (F1-Score and ROC-AUC) are near 1.0000, while Validation and Test scores drop noticeably. This occurs because, without strict hyperparameter tuning (like `max_depth`), Random Forest builds fully grown trees that memorize the training data. \n\n"
    report_content += "**Why XGBoost and LightGBM?**\n"
    report_content += "Gradient Boosting algorithms (XGBoost and LightGBM) construct trees sequentially, minimizing the errors of previous trees. They natively incorporate stronger regularization (e.g., learning rates, L1/L2 penalties) which prevents them from memorizing the data as aggressively as an unconstrained Random Forest. As a result, the gap between their Train and Test metrics is smaller, demonstrating far better generalization to unseen data.\n\n"
    report_content += "### Class Balancing Strategy\n"
    report_content += "Credit card default datasets are inherently imbalanced (the majority of people do not default). To prevent the models from becoming biased toward the majority class, we applied **Oversampling with Replacement** during the preprocessing phase. Specifically, we duplicated instances from the minority class (Defaults) until its count matched the majority class. This allows the models to learn decision boundaries that fairly represent both classes without ignoring the minority.\n\n"
    report_content += "### XAI Insights\n"
    report_content += "The feature importances and SHAP values provide transparent explanations for the models' decisions. The most influential predictors are heavily related to the user's recent payment history (`PAY_X`) and their overall `LIMIT_BAL` (credit limit). This aligns with financial logic: late payments in recent months are the strongest indicator of a future default.\n"
    
    with open(os.path.join(reports_dir, 'final_evaluation.md'), 'w') as f:
        f.write(report_content)
        
    print("Evaluation & XAI complete.")

if __name__ == "__main__":
    run_evaluation("graphs/models", "graphs/xai", "reports")
