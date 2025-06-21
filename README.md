# Medical-data

This project evaluates several machine learning models on a medical dataset for disease classification.
Models are trained incrementally on data chunks to simulate streaming data scenarios.
Evaluation metrics and confusion matrices are provided to compare performance.


## Models Used

- **Logistic Regression (SGD)**
- **MLP Neural Network**
- **XGBoost**
- **Transformer (PyTorch)**

## Features

- Categorical and numeric preprocessing with `LabelEncoder` and `StandardScaler`
- Chunked training to handle large datasets
- Transformer model with embedding and numeric projection layers
- Confusion matrix visualization and metric comparison

## Files

- `Clalit Data Models.ipynb` – Main notebook with full training, evaluation, and visualization
- `train_temp.csv`, `test_temp.csv` – Automatically generated training and testing splits

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch
