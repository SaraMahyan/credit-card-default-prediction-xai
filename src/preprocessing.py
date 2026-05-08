import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_preprocessing(data_path, output_dir, reports_dir):
    print("Starting Preprocessing...")
    df = pd.read_csv(data_path)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Identify target
    target_col = None
    for col in df.columns:
        if 'default' in col.lower() or 'target' in col.lower() or 'class' in col.lower():
            target_col = col
            break
            
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
        
    # Handle missing values
    missing_info = df.isnull().sum()
    df = df.fillna(df.median(numeric_only=True))
    
    report_content = f"# Preprocessing Report\n\n## Missing Values Handled\n```\n{missing_info}\n```\n\n"
    
    # Class Imbalance Check & SMOTE
    if target_col:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        
        class_counts = y.value_counts()
        report_content += f"## Original Class Distribution\n```\n{class_counts}\n```\n\n"
        
        # Manual oversampling
        df_scaled = pd.concat([X, y.reset_index(drop=True)], axis=1)
        majority_class = df_scaled[df_scaled[target_col] == 0]
        minority_class = df_scaled[df_scaled[target_col] == 1]
        
        # Check which class is majority
        if len(minority_class) > len(majority_class):
            majority_class, minority_class = minority_class, majority_class
            
        minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
        df_upsampled = pd.concat([majority_class, minority_upsampled])
        
        X_resampled = df_upsampled.drop(target_col, axis=1)
        y_resampled = df_upsampled[target_col]
        
        resampled_counts = y_resampled.value_counts()
        report_content += f"## Post-Balancing Class Distribution\n```\n{resampled_counts}\n```\n\n"
        
        plt.figure(figsize=(8, 6))
        sns.countplot(x=y_resampled, palette='viridis')
        plt.title('Target Variable Distribution (Post-Balancing)')
        plt.savefig(os.path.join(output_dir, 'post_balancing_distribution.png'))
        plt.close()
        
        report_content += "## Post-Balancing Graph\n![Post-Balancing Distribution](../graphs/preprocessing/post_balancing_distribution.png)\n"
        
        processed_df = pd.concat([X_resampled, y_resampled], axis=1)
        processed_df.to_csv('data/processed_data.csv', index=False)
        print("Preprocessing complete. Data saved to data/processed_data.csv")
    else:
        report_content += "No target column found. Proceeding with Unsupervised Learning setup.\n"
        # Scaling for unsupervised
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        processed_df = pd.DataFrame(X_scaled, columns=df.columns)
        processed_df.to_csv('data/processed_data.csv', index=False)
        
    with open(os.path.join(reports_dir, 'preprocessing_report.md'), 'w') as f:
        f.write(report_content)
        
if __name__ == "__main__":
    run_preprocessing("data/UCI_Credit_Card.csv", "graphs/preprocessing", "reports")
