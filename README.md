# Credit Card Default Prediction and Explainable AI

An end-to-end machine learning project for predicting credit card default risk using the UCI Credit Card dataset. The workflow covers exploratory data analysis, preprocessing, model training, evaluation, and explainable AI so the full pipeline can be reproduced from raw data to final reports and plots.

## Project Overview

This repository is designed to:

- explore the raw dataset with visualizations and summary plots,
- clean and preprocess the data,
- balance the target classes with oversampling,
- train multiple supervised models,
- evaluate the models on train, validation, and test splits,
- generate explainability artifacts with SHAP,
- save reports and figures for further review or presentation.

The target column used by the pipeline is `default.payment.next.month`.

## Dataset

The project uses the UCI Credit Card dataset located at:

- `data/UCI_Credit_Card.csv`

Key fields include:

- `LIMIT_BAL`
- `PAY_0` to `PAY_6`
- `BILL_AMT1` to `BILL_AMT6`
- `PAY_AMT1` to `PAY_AMT6`
- `default.payment.next.month`

## Workflow

The main pipeline is orchestrated by [`src/pipeline.py`](src/pipeline.py).

1. **EDA** - Generates initial plots from the raw dataset.
2. **Preprocessing** - Handles missing values, removes `ID`, scales numeric features, and balances the target class by oversampling.
3. **Training** - Splits the processed data into train, validation, and test sets, then trains multiple models.
4. **Evaluation** - Compares model performance, produces ROC and precision-recall curves, saves confusion matrices, and generates SHAP-based explainability plots.

## Models

The training step compares the following models:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

## Results

The current experiment shows a clear trade-off between raw training performance and generalization. The table below summarizes the key metrics reported for the train and test splits.

| Model | Split | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | Train | 0.6720 | 0.6831 | 0.6418 | 0.6618 | 0.7247 |
| Logistic Regression | Test | 0.6680 | 0.6774 | 0.6417 | 0.6590 | 0.7235 |
| Random Forest | Train | 0.9995 | 0.9991 | 0.9999 | 0.9995 | 1.0000 |
| Random Forest | Test | 0.9294 | 0.9070 | 0.9569 | 0.9313 | 0.9779 |
| XGBoost | Train | 0.8706 | 0.8733 | 0.8671 | 0.8702 | 0.9475 |
| XGBoost | Test | 0.8066 | 0.8083 | 0.8037 | 0.8060 | 0.8915 |
| LightGBM | Train | 0.7904 | 0.8180 | 0.7472 | 0.7810 | 0.8841 |
| LightGBM | Test | 0.7598 | 0.7899 | 0.7078 | 0.7466 | 0.8503 |

Key findings:

- Random Forest achieved the strongest raw scores but also showed the largest train-test gap, which indicates overfitting.
- XGBoost offered the best balance between predictive power and generalization in this experiment.
- Recall is especially important in credit risk, because missing a default is more costly than raising a false alarm.
- SHAP analysis identified recent repayment behavior, especially `PAY_0`, and credit limit `LIMIT_BAL` as the most influential features.

## Project Structure

```text
data/
  UCI_Credit_Card.csv
  processed_data.csv
  X_train.csv
  X_val.csv
  X_test.csv
  y_train.csv
  y_val.csv
  y_test.csv
graphs/
  raw_data/
  preprocessing/
  models/
  xai/
models/
preprocessing/
raw_data/
xai/
reports/
src/
  eda.py
  preprocessing.py
  train.py
  evaluate.py
  pipeline.py
```

## Installation

Create and activate a Python environment, then install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib xgboost lightgbm shap
```

If you prefer a `requirements.txt`, you can install from one after adding it to the project.

## How To Run

Run the full pipeline from the project root:

```bash
python src/pipeline.py
```

You can also run individual stages if needed:

```bash
python src/eda.py
python src/preprocessing.py
python src/train.py
python src/evaluate.py
```

## Outputs

After running the pipeline, the repository will contain generated artifacts such as:

- `graphs/raw_data/` - EDA plots
- `graphs/preprocessing/` - preprocessing visualizations
- `graphs/models/` - model comparison plots, confusion matrices, ROC/PR curves, and feature importance charts
- `graphs/xai/` - SHAP summary plots
- `reports/` - markdown reports for preprocessing, clustering or training, and final evaluation
- `data/processed_data.csv` - preprocessed dataset
- `data/X_train.csv`, `data/X_val.csv`, `data/X_test.csv` - split features
- `data/y_train.csv`, `data/y_val.csv`, `data/y_test.csv` - split labels

## Notes

- The preprocessing step uses standardization and oversampling to help address class imbalance.
- If a dataset without a target-like column is provided, the training script falls back to an unsupervised workflow.
- Some plots and reports are written by the scripts directly, so make sure the output folders are writable.

## Contact

For questions or collaboration, contact:

saramahyan.smb@gmail.com