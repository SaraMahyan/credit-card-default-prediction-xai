import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(data_path, output_dir):
    print("Starting EDA...")
    df = pd.read_csv(data_path)
    
    # Create output dir if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Target Distribution
    target_col = None
    for col in df.columns:
        if 'default' in col.lower() or 'target' in col.lower() or 'class' in col.lower():
            target_col = col
            break
            
    if target_col:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=target_col, data=df, palette='viridis')
        plt.title('Target Variable Distribution')
        plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
        plt.close()
        
    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # 3. Missing Value Matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title('Missing Value Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missing_values.png'))
    plt.close()
    
    # 4. Feature Distribution (LIMIT_BAL if exists, else first numeric)
    if 'LIMIT_BAL' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['LIMIT_BAL'], bins=50, kde=True, color='blue')
        plt.title('Distribution of LIMIT_BAL')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_distribution.png'))
        plt.close()
    
    print(f"EDA graphs saved to {output_dir}")

if __name__ == "__main__":
    run_eda("data/UCI_Credit_Card.csv", "graphs/raw_data")
