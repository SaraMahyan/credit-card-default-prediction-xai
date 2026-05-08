import os
import sys

# Ensure src is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import eda
import preprocessing
import train
import evaluate

def run_all():
    print("=========================================")
    print("STARTING ML PIPELINE")
    print("=========================================")
    
    data_path = "data/UCI_Credit_Card.csv"
    
    # Check if dataset exists
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    print("\n--- STEP 1: Exploratory Data Analysis ---")
    eda.run_eda(data_path, "graphs/raw_data")
    
    print("\n--- STEP 2: Preprocessing ---")
    preprocessing.run_preprocessing(data_path, "graphs/preprocessing", "reports")
    
    print("\n--- STEP 3 & 4: Dynamic Target Check & Model Training ---")
    train.run_training("data/processed_data.csv", "graphs/models", "graphs/clustering", "reports")
    
    print("\n--- STEP 5 & 6: Explainable AI & Final Evaluation ---")
    evaluate.run_evaluation("graphs/models", "graphs/xai", "reports")
    
    print("\n=========================================")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=========================================")

if __name__ == "__main__":
    run_all()
