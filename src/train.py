import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def run_training(data_path, models_dir, clustering_dir, reports_dir):
    print("Starting Training / Clustering Phase...")
    df = pd.read_csv(data_path)
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(clustering_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    target_col = None
    for col in df.columns:
        if 'default' in col.lower() or 'target' in col.lower() or 'class' in col.lower():
            target_col = col
            break
            
    if target_col is None:
        print("No target label exists. Switching to Unsupervised Learning...")
        # Unsupervised Learning
        X = df.values
        
        # KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans_labels = kmeans.fit_predict(X)
        kmeans_silhouette = silhouette_score(X, kmeans_labels)
        
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
        plt.title('KMeans Clustering')
        plt.savefig(os.path.join(clustering_dir, 'kmeans_scatter.png'))
        plt.close()
        
        # Report
        with open(os.path.join(reports_dir, 'clustering_report.md'), 'w') as f:
            f.write("# Clustering Report\n")
            f.write("Target label not found. Proceeded with unsupervised learning.\n")
            f.write(f"- KMeans Silhouette Score: {kmeans_silhouette:.4f}\n")
            f.write("\n![KMeans Scatter](../graphs/clustering/kmeans_scatter.png)\n")
        return
        
    print("Target label found. Proceeding with Supervised Learning...")
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Split Train (70%), Val (15%), Test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp) # 0.15/0.85 approx 0.1765
    
    # Save splits for evaluation step
    X_train.to_csv('data/X_train.csv', index=False)
    X_val.to_csv('data/X_val.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_val.to_csv('data/y_val.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': LGBMClassifier(random_state=42)
    }
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(models_dir, f'{name}.pkl'))
        
    print("Training complete. Models saved.")

if __name__ == "__main__":
    run_training("data/processed_data.csv", "graphs/models", "graphs/clustering", "reports")
