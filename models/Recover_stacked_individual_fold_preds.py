import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

# Load predictions
df = pd.read_csv('stacked_ridge_predictions.csv')

# Calculate metrics per fold
fold_metrics = []
for fold in sorted(df['fold'].unique()):
    fold_data = df[df['fold'] == fold]

    metrics = {
        'fold': fold,
        'n_samples': len(fold_data),
        'pearson_r': pearsonr(fold_data['true_g'], fold_data['stacked_pred'])[0],
        'r2': r2_score(fold_data['true_g'], fold_data['stacked_pred']),
        'mae': mean_absolute_error(fold_data['true_g'], fold_data['stacked_pred'])
    }
    fold_metrics.append(metrics)

fold_df = pd.DataFrame(fold_metrics)
print(fold_df)
print("\nSummary across folds:")
print(fold_df[['pearson_r', 'r2', 'mae']].describe())