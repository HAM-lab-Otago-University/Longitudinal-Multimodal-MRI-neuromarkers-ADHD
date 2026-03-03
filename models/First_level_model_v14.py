#!/usr/bin/env python
"""
First level model script for XG Boost, Kernel ridge regression and PLS regression models.
Updated for nested cross-validation structure.

This script implements nested cross-validation where each subject appears across multiple folds.
Key changes from v9:
1. Updated job indexing for modality-specific array jobs (0-14 for 5 folds × 3 models)
2. Added validation for nested CV structure
3. Enhanced logging for nested CV

Usage:
  Use slurm scripts to call this script on the HPC.

Author: Jack Scott
Updated for Nested CV Pipeline
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import argparse
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("modality_prediction_{}.log".format(datetime.now().strftime("%Y%m%d_%H%M%S"))),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("modality_prediction")

###########################################
#          CONFIGURATION SECTION          #
###########################################

# Default parameters
SEED = 42
N_CV_FOLDS = 5
OUTPUT_BASE_DIR = "/projects/sciences/psychology/narunpat-lab/Jack/First_level_model_aligned/model_results/"

# Modalities list
STRUCTURAL_MODALITIES = [
    "cortical_area",
    "cortical_thickness",
    "subcortical_volume",
    "total_brain_volume"
]

FUNCTIONAL_MODALITIES = [
    "alff",
    "reho",
    "functional_connectivity"
]

ALL_MODALITIES = STRUCTURAL_MODALITIES + FUNCTIONAL_MODALITIES

# Models list
MODELS = ["xgboost", "krr", "pls", "rf"]


###########################################
#           END CONFIGURATION             #
###########################################

def parse_args():
    """Parse command line arguments with support for modality-specific job arrays."""
    parser = argparse.ArgumentParser(description='Build prediction models for neuroimaging modalities with nested CV')

    # For modality-specific array jobs (15 jobs: 5 folds × 3 models)
    parser.add_argument('--job_index', type=int, required=False, default=None,
                        help='Job index for modality-specific array jobs (0-14)')

    parser.add_argument('--modality', type=str, required=True,
                        choices=ALL_MODALITIES,
                        help='Neuroimaging modality to model')

    parser.add_argument('--model', type=str, required=False, default=None,
                        choices=MODELS,
                        help='Model type to use')

    parser.add_argument('--fold', type=int, required=False, default=None,
                        help='Specific fold to process (0-4)')

    parser.add_argument('--base_dir', type=str, required=False,
                        default="/projects/sciences/psychology/narunpat-lab/Jack/First_level_model_aligned",
                        help='Base directory for the project')

    parser.add_argument('--output_dir', type=str, required=False, default=None,
                        help='Custom output directory. If not specified, a default directory will be created.')

    return parser.parse_args()


def get_job_parameters_modality_specific(job_index):
    """
    Convert job index to fold and model indices for modality-specific array jobs.
    Updated for nested CV with 15 jobs per modality (5 folds × 3 models).

    Parameters:
    -----------
    job_index : int
        Job index from SLURM array (0-14)

    Returns:
    --------
    tuple : (fold, model)
    """
    num_folds = 5
    num_models = len(MODELS)  # 3

    # Calculate indices for modality-specific jobs
    fold = job_index % num_folds
    model_idx = job_index // num_folds

    # Validate indices
    if model_idx >= num_models:
        raise ValueError(f"Invalid job index {job_index}. Expected 0-{num_folds * num_models - 1}")

    model = MODELS[model_idx]

    logger.info(f"Job index {job_index} corresponds to fold {fold}, model {model}")
    return fold, model


def validate_nested_cv_structure(X_train, X_test, subject_ids_train, subject_ids_test, fold_idx, modality_name):
    """
    Validate that the data structure is consistent with nested CV expectations.
    """
    logger.info(f"Validating nested CV structure for {modality_name} fold {fold_idx}")

    # Extract subject IDs from timepoint IDs
    def extract_subject_ids(timepoint_ids):
        return [tid.split('_')[0] if '_' in str(tid) else str(tid) for tid in timepoint_ids]

    train_subjects = set(extract_subject_ids(subject_ids_train))
    test_subjects = set(extract_subject_ids(subject_ids_test))

    # Check for overlap between train and test in this fold (should be none)
    overlap = train_subjects.intersection(test_subjects)
    if overlap:
        logger.error(f"Data leakage detected in fold {fold_idx}: {len(overlap)} subjects in both train and test")
        logger.error(f"Example overlapping subjects: {list(overlap)[:5]}")
        return False

    logger.info(
        f"Fold {fold_idx} validation passed: {len(train_subjects)} train subjects, {len(test_subjects)} test subjects, no overlap")
    return True


def flatten_connectivity_matrix(matrix):
    """
    Flatten a connectivity matrix by extracting the lower triangular part (excluding diagonal).

    Parameters:
    -----------
    matrix : DataFrame or array
        Connectivity matrix

    Returns:
    --------
    array : Flattened connectivity matrix
    """
    # Convert to numpy array
    if isinstance(matrix, pd.DataFrame):
        matrix_values = matrix.values
    else:
        matrix_values = matrix

    # Get lower triangular indices (excluding diagonal)
    indices = np.tril_indices_from(matrix_values, k=-1)

    # Extract values
    flattened = matrix_values[indices]

    return flattened


def is_connectivity_data(feature_columns, data_shape):
    """
    Detect if the data might be a connectivity matrix.

    Parameters:
    -----------
    feature_columns : list
        List of feature column names
    data_shape : tuple
        Shape of the data array

    Returns:
    --------
    bool : True if data appears to be connectivity matrix
    """
    # Strategy 1: Check for square root relationship in feature count
    n_features = len(feature_columns)
    if n_features > 1000:  # Typical for connectivity
        sqrt_n = int(np.sqrt(n_features))
        if sqrt_n * (sqrt_n - 1) // 2 == n_features:
            return True

    # Strategy 2: Check for naming patterns in columns
    conn_patterns = ["roi", "region", "connectivity", "corr", "network", "atlas"]
    pattern_matches = sum(any(pattern in col.lower() for pattern in conn_patterns)
                          for col in feature_columns[:100])  # Check first 100 columns
    if pattern_matches > 20:  # If many column names match connectivity patterns
        return True

    # Strategy 3: Based on explicit modality name
    # This would be handled outside this function when modality name is known

    return False


def preprocess_connectivity_data(X_train, X_test, feature_columns, modality_name):
    """
    Apply specialized preprocessing for connectivity matrices.
    No feature reduction is performed.
    """
    # Check if this is connectivity data
    is_connectivity = modality_name == "functional_connectivity" or \
                      is_connectivity_data(feature_columns, X_train.shape)

    if not is_connectivity:
        # Return original data for non-connectivity modalities
        return X_train, X_test, feature_columns

    logger.info(f"Detected connectivity data, using all {X_train.shape[1]} features")

    # No feature reduction - return original data
    return X_train, X_test, feature_columns


def get_krr_param_grid(modality_name, n_features):
    """
    Get comprehensive parameter grid for Kernel Ridge Regression.
    Adapts based on modality and feature count.

    Parameters:
    -----------
    modality_name : str
        Name of the modality
    n_features : int
        Number of features

    Returns:
    --------
    dict : Dictionary of KRR hyperparameters for grid search
    """
    # For functional connectivity, try linear and RBF, but with RBF we need much stronger regularization.
    if modality_name == "functional_connectivity":
        return [
            {
                'kernel': ['linear'], # back because linear should work nicely with super high dimensionality
                'alpha': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 5000, 10000, 20000],
                'gamma': [None]
            },
            {
                'kernel': ['rbf'],
                'alpha': [1, 10, 100, 1000, 5000, 10000, 50000],
                'gamma': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
            }
        ]


    # For low-dimensional structural modalities that cause singular matrix issues
    elif modality_name in ["subcortical_volume", "total_brain_volume"]:
        return [
            {
                'kernel': ['linear'],
                'alpha': [100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 10000, 20000],  # Higher minimum alpha
                'gamma': [None]
            },
            {
                'kernel': ['rbf'],
                'alpha': [100, 500, 1000, 5000, 10000],  # Higher minimum alpha
                'gamma': [0.01, 0.1, 0.5, 1.0, 2.0]  # Larger gamma values for stability
            },
            {
                'kernel': ['poly'],
                'alpha': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 5000],  # Much higher minimum alpha
                'gamma': [0.01, 0.1, 0.5, 1.0],  # Larger gamma values to avoid singular matrices
                'degree': [2, 3]  # Keep both degrees
            }
        ]

    # Special case for ReHo - force stronger regularization
    elif modality_name == "reho":
        return [
            {
                'kernel': ['linear'],
                'alpha': [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],  # Start from 0.1
                'gamma': [None]
            },
            {
                'kernel': ['rbf'],
                'alpha': [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],  # Much stronger alpha
                'gamma': [1e-4, 1e-3, 1e-2, 0.1]  # Smaller gamma values to prevent overfitting
            },
            {
                'kernel': ['poly'],
                'alpha': [0.1, 0.5, 1, 5, 10, 50, 100],
                'gamma': [1e-3, 1e-2, 0.1],  # Conservative gamma
                'degree': [2, 3]
            }
        ]

    # Special case for cortical area - prone to overfitting even after bilateral averaging
    elif modality_name == "cortical_area":
        return [
            {
                'kernel': ['linear'],
                'alpha': [1000, 2000, 5000, 10000, 12000, 14000, 15000, 20000, 25000, 30000, 35000, 40000],  # Much stronger regularization
                'gamma': [None]
            },
            {
                'kernel': ['rbf'],
                'alpha': [10, 50, 100, 200, 300, 400, 500, 1000, 2000, 5000, 10000, 15000, 20000],  # Strong regularization
                'gamma': [0.1, 0.5, 1.0, 2.0, 5.0]  # Avoid tiny gamma values causing numerical issues
            }
            # No polynomial kernel - it was causing the worst overfitting
        ]

    # For extremely high-dimensional data
    elif n_features > 30000:
        return [
            {
                'kernel': ['linear'],
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 500, 1000, 5000, 10000],
                'gamma': [None]
            },
            {
                'kernel': ['rbf'],
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 500, 1000],
                'gamma': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]  # Only numeric values
            }
        ]

    # For high-dimensional but not extreme data
    elif n_features > 5000:
        return [
            {
                'kernel': ['linear'],
                'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
                'gamma': [None]
            },
            {
                'kernel': ['rbf'],
                'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500],
                'gamma': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]  # Only numeric values
            }
        ]

    # For moderate-dimensional data
    elif n_features > 1000:
        return [
            {
                'kernel': ['linear'],
                'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500],
                'gamma': [None]
            },
            {
                'kernel': ['rbf'],
                'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
                'gamma': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]  # Only numeric values
            },
            {
                'kernel': ['poly'],
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                'gamma': [1e-4, 1e-3, 1e-2, 0.1, 1.0],  # Only numeric values
                'degree': [2, 3]
            }
        ]

    # For lower-dimensional data
    else:
        return [
            {
                'kernel': ['linear'],
                'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 200, 300, 500],
                'gamma': [None]
            },
            {
                'kernel': ['rbf'],
                'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
                'gamma': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]  # Only numeric values
            },
            {
                'kernel': ['poly'],
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 200, 300, 500],
                'gamma': [1e-4, 1e-3, 1e-2, 0.1, 1.0],  # Only numeric values
                'degree': [2, 3]
            }
        ]


def get_xgb_param_grid(modality_name, n_features):
    """
    Get optimized XGBoost parameter grid based on modality and feature count.
    No feature selection is performed - parameters are adjusted to handle the full feature set.
    """
    # Base parameters common to all XGBoost models
    base_grid = {
        'booster': ['gbtree'],
        'objective': ['reg:squarederror'],
        'tree_method': ['hist'],  # More efficient algorithm
        'random_state': [SEED]
    }

    # Special case for cortical area - prone to overfitting, needs strong regularization
    if modality_name == "cortical_area":
        return {
            **base_grid,
            'eta': [0.01, 0.03, 0.05],  # Slower learning rates
            'max_depth': [2, 3, 4],  # Shallower trees
            'min_child_weight': [5, 10, 15],  # Require more samples per leaf
            'subsample': [0.6, 0.7, 0.8],  # More aggressive data subsampling
            'colsample_bytree': [0.5, 0.6, 0.7],  # More aggressive feature sampling
            'gamma': [0.1, 0.5, 1.0],  # Stronger minimum loss reduction
            'reg_alpha': [0, 0.5, 1, 5, 10, 20],  # Much stronger L1 regularization
            'reg_lambda': [1, 2, 3, 4, 5, 10, 20, 50]  # Much stronger L2 regularization
        }

    # Ultra-low dimension (3-20 features)
    if n_features <= 20:
        return {
            **base_grid,
            'eta': [0.03, 0.05, 0.1],
            'max_depth': [4, 5, 6],  # Deeper trees for fewer features
            'min_child_weight': [1, 3],  # Few samples per leaf acceptable
            'subsample': [0.9, 1.0],  # Minimal data subsampling
            'colsample_bytree': [1.0],  # Use all features
            'gamma': [0],  # No minimum loss reduction
            'reg_alpha': [0],  # No L1 regularization
            'reg_lambda': [1]  # Minimal L2
        }

    # Low dimension (21-100 features)
    elif n_features <= 100:
        return {
            **base_grid,
            'eta': [0.03, 0.05, 0.1],
            'max_depth': [4, 5],  # Moderate depth
            'min_child_weight': [3, 5],  # Require more samples per leaf
            'subsample': [0.8, 0.9],  # Moderate data subsampling
            'colsample_bytree': [0.7, 0.9],  # Partial feature sampling
            'gamma': [0, 0.1],  # Mild node splitting control
            'reg_alpha': [0, 0.5],  # Optional L1
            'reg_lambda': [1.5, 2]  # Moderate L2
        }

    # Medium dimension (101-5000 features)
    elif n_features <= 5000:
        return {
            **base_grid,
            'eta': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5],  # Moderate depth
            'min_child_weight': [3, 5],  # More samples per leaf
            'subsample': [0.7, 0.8, 0.9],  # Moderate data subsampling
            'colsample_bytree': [0.5, 0.7],  # Partial feature sampling
            'gamma': [0.1, 0.3],  # Mild node splitting control
            'reg_alpha': [0, 0.5],  # Optional L1
            'reg_lambda': [1, 2]  # Moderate L2
        }

    # High dimension (5K-30K features)
    elif n_features <= 30000:
        return {
            **base_grid,
            'eta': [0.01, 0.05],
            'max_depth': [3, 4],  # Shallow trees
            'min_child_weight': [5, 10],  # Strict leaf requirements
            'subsample': [0.6, 0.8],  # Aggressive data subsampling
            'colsample_bytree': [0.3, 0.5, 0.7],  # Heavy feature sampling
            'gamma': [0.3, 0.5],  # Strong node splitting control
            'reg_alpha': [0.5, 1],  # Strong L1
            'reg_lambda': [2, 3],  # Strong L2
            'max_bin': [256]  # Faster training
        }

    # Extreme dimension (30K+ features)
    else:
        return {
            **base_grid,
            'eta': [0.01],  # Very small learning rate
            'max_depth': [2],  # Very shallow trees
            'min_child_weight': [20],  # Large min samples per leaf
            'subsample': [0.5],  # Heavy data subsampling
            'colsample_bytree': [0.3, 0.5],  # Sample features per tree
            'colsample_bylevel': [0.2, 0.3],  # Further subsample per level
            'gamma': [0.5, 1],  # Stronger node splitting control
            'reg_alpha': [1, 2],  # Aggressive L1 (induces sparsity)
            'reg_lambda': [3, 5],  # Strong L2 regularization
            'max_bin': [128, 256],  # Fewer bins for speed
            'n_estimators': [1000, 1500],  # More trees due to smaller eta
        }

def get_rf_param_grid(modality_name, n_features):
    """
    Get comprehensive parameter grid for Random Forest Regression.
    Adapts based on modality and feature count for neuroimaging data.

    Parameters:
    -----------
    modality_name : str
        Name of the modality
    n_features : int
        Number of features

    Returns:
    --------
    dict : Dictionary of RF hyperparameters for grid search
    """

    # Base parameters for all RF models
    base_params = {
        'random_state': [SEED],
        'n_jobs': [1],  # Use all available cores
        'oob_score': [True]  # Out-of-bag scoring for free validation
    }

    # Special handling for functional connectivity (ultra-high dimensional)
    if modality_name == "functional_connectivity":
        return {
            **base_params,
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 15, 20, None],  # Deeper trees for complex patterns
            'min_samples_split': [10, 20, 50],  # Higher splits to prevent overfitting
            'min_samples_leaf': [5, 10, 20],  # More samples per leaf
            'max_features': [0.1, 0.2, 0.33, 'sqrt'],  # Heavy feature subsampling
            'bootstrap': [True],  # Always use bootstrap for high-dim
            'max_samples': [0.7, 0.8, 0.9]  # Subsample for efficiency
        }

    # Low-dimensional structural modalities prone to overfitting
    elif modality_name in ["subcortical_volume", "total_brain_volume"]:
        return {
            **base_params,
            'n_estimators': [200, 300, 500, 800],  # More trees for stability
            'max_depth': [5, 8, 12, 15],  # Moderate depth
            'min_samples_split': [5, 10, 15],  # Conservative splitting
            'min_samples_leaf': [3, 5, 8],  # Prevent tiny leaves
            'max_features': [0.5, 0.7, 'sqrt', 'log2'],  # Conservative feature sampling
            'bootstrap': [True],
            'max_samples': [0.8, 0.9, 1.0]
        }

    # Special case for cortical area - very prone to overfitting
    elif modality_name == "cortical_area":
        return {
            **base_params,
            'n_estimators': [300, 500, 800],  # Many trees for averaging
            'max_depth': [8, 12, 15],  # Limit depth
            'min_samples_split': [10, 20, 30],  # Conservative splits
            'min_samples_leaf': [5, 10, 15],  # Larger leaf nodes
            'max_features': [0.3, 0.5, 'sqrt'],  # Aggressive feature subsampling
            'bootstrap': [True],
            'max_samples': [0.7, 0.8, 0.9]  # Subsample to reduce overfitting
        }

    # ReHo and ALFF - moderate dimensional functional data
    elif modality_name in ["reho", "alff"]:
        return {
            **base_params,
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [8, 12, 15, 20],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 5, 8],
            'max_features': [0.5, 0.7, 'sqrt', 'log2'],
            'bootstrap': [True],
            'max_samples': [0.8, 0.9, 1.0]
        }

    # High-dimensional data (general case)
    elif n_features > 5000:
        return {
            **base_params,
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10],
            'max_features': [0.2, 0.33, 'sqrt'],
            'bootstrap': [True],
            'max_samples': [0.7, 0.8, 0.9]
        }

    # Moderate-dimensional data
    elif n_features > 1000:
        return {
            **base_params,
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [8, 12, 15, 20],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 5, 8],
            'max_features': [0.5, 0.7, 'sqrt', 'log2'],
            'bootstrap': [True]
        }

    # Lower-dimensional data
    else:
        return {
            **base_params,
            'n_estimators': [100, 200, 300, 500, 800],
            'max_depth': [5, 8, 12, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'max_features': [0.5, 0.7, 'sqrt', 'log2', None],
            'bootstrap': [True, False]  # Try both for low-dim
        }

def verify_data_types(df, modality_name):
    """
    Verify all data columns are numeric and log any issues found.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to check
    modality_name : str
        Name of the modality for logging purposes

    Returns:
    --------
    tuple : (clean_df, non_numeric_cols)
        Clean DataFrame with only numeric columns and list of removed columns
    """
    # Check for non-numeric columns (excluding eid and subject_id)
    id_cols = ['eid', 'subject_id', 'g']
    data_cols = [col for col in df.columns if col not in id_cols]

    # Find non-numeric columns
    non_numeric_cols = []
    for col in data_cols:
        try:
            # Try to convert to float
            pd.to_numeric(df[col])
        except (ValueError, TypeError):
            non_numeric_cols.append(col)

    # Log findings
    if non_numeric_cols:
        logger.warning(f"Found {len(non_numeric_cols)} non-numeric columns in {modality_name} data")
        logger.warning(f"First few non-numeric columns: {non_numeric_cols[:5]}")
        logger.warning(f"Example values: {df[non_numeric_cols[0]].iloc[:3].tolist() if non_numeric_cols else 'None'}")

        # Create a clean dataframe with only numeric columns and IDs
        clean_cols = [col for col in df.columns if col in id_cols or col not in non_numeric_cols]
        clean_df = df[clean_cols]

        logger.info(f"Removed {len(non_numeric_cols)} non-numeric columns, keeping {len(clean_cols)} columns")
        return clean_df, non_numeric_cols
    else:
        logger.info(f"All columns in {modality_name} data are numeric - good to proceed")
        return df, []

def perform_rf_tuning(X_train, y_train, feature_columns, modality_name, fold_idx, n_cv=5):
    """
    Perform Random Forest hyperparameter tuning without feature selection.
    Using the full feature set for all modalities.
    """
    logger.info(f"Starting Random Forest hyperparameter tuning for {modality_name}, fold {fold_idx}")

    num_features = len(feature_columns)
    logger.info(f"Dataset has {num_features} features - using all features without reduction")

    # Get adaptive parameter grid based on number of features and modality
    param_grid = get_rf_param_grid(modality_name, num_features)

    # Set up cross-validation for parameter tuning - MATCH KRR/PLS APPROACH
    from sklearn.ensemble import RandomForestRegressor
    rf_model = RandomForestRegressor(random_state=SEED)

    # Create CV splitter to match KRR and PLS approaches
    cv = KFold(n_splits=n_cv, shuffle=True, random_state=SEED)

    # Use GridSearchCV for parameter tuning
    search = GridSearchCV(
        rf_model,
        param_grid,
        cv=cv,  # Use explicit KFold like KRR/PLS
        scoring='neg_mean_squared_error',  # Match XGBoost/PLS scoring
        verbose=1,
        n_jobs=1
    )

    # Fit model on full feature set
    search.fit(X_train, y_train)

    # Get best parameters
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = -search.best_score_

    logger.info(f"Best score (RMSE): {np.sqrt(best_score):.4f} with parameters: {best_params}")

    return {
        'model': best_model,
        'n_features': num_features,
        'selected_indices': np.arange(num_features),  # All features
        'selected_features': feature_columns,  # All feature names
        'best_params': best_params,
        'rmse': np.sqrt(best_score),
        'search': search
    }


def perform_xgboost_tuning(X_train, y_train, feature_columns, modality_name, fold_idx, n_cv=5):
    """
    Perform XGBoost hyperparameter tuning without feature selection.
    Using the full feature set for all modalities.
    """
    logger.info(f"Starting XGBoost hyperparameter tuning for {modality_name}, fold {fold_idx}")

    num_features = len(feature_columns)
    logger.info(f"Dataset has {num_features} features - using all features without reduction")

    # Get adaptive parameter grid based on number of features and modality
    param_grid = get_xgb_param_grid(modality_name, num_features)

    # Set up cross-validation for parameter tuning
    xgb_model = xgb.XGBRegressor(random_state=SEED)

    # Use GridSearchCV for parameter tuning
    search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=n_cv,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )

    # Fit model on full feature set
    search.fit(X_train, y_train)

    # Get best parameters
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = -search.best_score_

    logger.info(f"Best score (RMSE): {np.sqrt(best_score):.4f} with parameters: {best_params}")

    return {
        'model': best_model,
        'n_features': num_features,
        'selected_indices': np.arange(num_features),  # All features
        'selected_features': feature_columns,  # All feature names
        'best_params': best_params,
        'rmse': np.sqrt(best_score),
        'search': search
    }


def impute_missing_values(X, feature_columns, modality_name, strategy='mean'):
    """
    Impute missing values in feature matrix.

    Parameters:
    -----------
    X : numpy array
        Feature matrix that may contain NaN values
    feature_columns : list
        List of feature column names
    modality_name : str
        Name of the modality for logging
    strategy : str, default='mean'
        Imputation strategy ('mean', 'median', or 'constant')

    Returns:
    --------
    numpy array : Imputed feature matrix
    """
    from sklearn.impute import SimpleImputer

    # Check if there are any NaN values
    n_missing = np.isnan(X).sum()
    if n_missing == 0:
        logger.info(f"No missing values found in {modality_name} data")
        return X

    # Log missing value info
    n_missing_cols = np.sum(np.isnan(X).any(axis=0))
    n_missing_rows = np.sum(np.isnan(X).any(axis=1))
    logger.warning(f"Found {n_missing} missing values in {modality_name} data")
    logger.warning(
        f"Missing values affect {n_missing_rows}/{X.shape[0]} samples and {n_missing_cols}/{X.shape[1]} features")

    # Create imputer
    imputer = SimpleImputer(strategy=strategy, missing_values=np.nan)

    # Fit and transform the data
    X_imputed = imputer.fit_transform(X)
    logger.info(f"Imputed missing values using {strategy} strategy")

    return X_imputed


def perform_krr_tuning(X_train, y_train, feature_columns, modality_name, fold_idx, n_cv=5):
    """
    Perform KRR hyperparameter tuning without feature selection.
    Using the full feature set for all modalities.
    """
    logger.info(f"Performing KRR hyperparameter tuning for {modality_name}, fold {fold_idx}")

    num_features = len(feature_columns)
    logger.info(f"Using all {num_features} features without reduction")

    # Get KRR parameter grid based on modality and feature count
    param_grid = get_krr_param_grid(modality_name, num_features)

    # Create CV splitter
    cv = KFold(n_splits=n_cv, shuffle=True, random_state=SEED)

    # Create GridSearchCV
    krr = KernelRidge()
    grid_search = GridSearchCV(
        krr,
        param_grid,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,  # Use all available cores
        verbose=1
    )

    # Fit the model
    logger.info(f"Starting KRR grid search with {n_cv}-fold CV")
    grid_search.fit(X_train, y_train)

    # Get best parameters and model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    logger.info(f"Best KRR parameters: {best_params}")
    logger.info(f"Best KRR score (neg MAE): {grid_search.best_score_:.4f}")

    # No feature selection - all features are used
    selected_indices = None
    selected_features = feature_columns

    # Return results
    return {
        'model': best_model,
        'best_params': best_params,
        'cv_results': grid_search.cv_results_,
        'selected_indices': selected_indices,
        'selected_features': selected_features,
        'training_metrics': {
            'rmse': np.sqrt(mean_squared_error(y_train, best_model.predict(X_train))),
            'r2': r2_score(y_train, best_model.predict(X_train)),
            'mae': mean_absolute_error(y_train, best_model.predict(X_train)),
            'pearson_r': pearsonr(y_train, best_model.predict(X_train))[0]
        }
    }


def perform_pls_tuning(X_train, y_train, feature_columns, modality_name, fold_idx, max_components=20, n_cv=5):
    """
    Perform hyperparameter tuning for Partial Least Squares regression.

    Parameters:
    -----------
    X_train : array
        Training features
    y_train : array
        Training target
    feature_columns : list
        List of feature column names
    modality_name : str
        Name of the modality
    fold_idx : int
        Index of the fold
    max_components : int
        Maximum number of components to try
    n_cv : int
        Number of cross-validation folds

    Returns:
    --------
    dict : Dictionary with best model, predictions, and evaluation metrics
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import cross_val_score, KFold

    logger.info(f"Performing PLS tuning for {modality_name}, fold {fold_idx}")

    # Number of samples and features
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]

    logger.info(f"Dataset has {n_features} features. PLS will create optimal components.")

    # constrain by number of features:
    max_components = min(max_components, n_features)

    # Adjust max_components based on data size
    max_components = min(max_components, n_samples - 1)

    # For high-dimensional data, try fewer components to avoid overfitting
    if n_features > 10000:
        max_components = min(max_components, 10)
    elif n_features > 1000:
        max_components = min(max_components, 15)

    logger.info(f"Searching for optimal number of components up to {max_components}")

    # Set up cross-validation
    cv = KFold(n_splits=n_cv, shuffle=True, random_state=SEED)

    # Track results
    cv_results = []

    # Try different numbers of components
    for n_components in range(1, max_components + 1):
        # Create PLS model
        pls = PLSRegression(n_components=n_components, scale=True, max_iter=1000)

        # Perform cross-validation
        scores = cross_val_score(
            pls, X_train, y_train,
            cv=cv,
            scoring='neg_mean_squared_error'
        )

        # Convert to positive RMSE
        rmse_scores = np.sqrt(-scores)

        # Store results
        cv_results.append({
            'n_components': n_components,
            'mean_rmse': np.mean(rmse_scores),
            'std_rmse': np.std(rmse_scores)
        })

        logger.info(f"  Components: {n_components}, RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")

    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(cv_results)

    # Find best number of components
    best_result = min(cv_results, key=lambda x: x['mean_rmse'])
    best_n_components = best_result['n_components']

    logger.info(f"Best PLS model: {best_n_components} components, RMSE: {best_result['mean_rmse']:.4f}")

    # Train final model with best parameters
    best_pls = PLSRegression(n_components=best_n_components, scale=True, max_iter=1000)
    best_pls.fit(X_train, y_train)

    return {
        'model': best_pls,
        'n_components': best_n_components,
        'cv_results': results_df,
        'rmse': best_result['mean_rmse'],
        'selected_indices': None,  # No explicit feature selection
        'selected_features': feature_columns  # Using all original features
    }


def evaluate_metrics(y_true, y_pred, prefix=""):
    """
    Evaluate model performance with multiple metrics.

    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    prefix : str
        Prefix for metric names in the returned dictionary

    Returns:
    --------
    dict : Dictionary of metrics
    """
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    pearson_r = pearsonr(y_true, y_pred)[0]

    # Log results
    logger.info(f"{prefix} Metrics:")
    logger.info(f"RMSE = {rmse:.4f}")
    logger.info(f"MAE = {mae:.4f}")
    logger.info(f"R² = {r2:.4f}")
    logger.info(f"Pearson's r = {pearson_r:.4f}")

    # Return as dictionary
    return {
        f'{prefix}RMSE': rmse,
        f'{prefix}MAE': mae,
        f'{prefix}R2': r2,
        f'{prefix}Pearson_r': pearson_r
    }


def load_standard_modality_data(modality_name, fold_idx, split, base_dir):
    """
    Load data for standard modalities (non-connectivity).

    Parameters:
    -----------
    modality_name : str
        Name of the modality
    fold_idx : int
        Fold index
    split : str
        'train' or 'test'
    base_dir : str
        Base directory

    Returns:
    --------
    tuple : (X, y, feature_columns, subject_ids)
    """
    # Determine modality type and directory
    if modality_name in STRUCTURAL_MODALITIES:
        modality_dir = os.path.join(base_dir, "structural", modality_name)
    elif modality_name in FUNCTIONAL_MODALITIES:
        modality_dir = os.path.join(base_dir, "functional", modality_name)
    else:
        raise ValueError(f"Unknown modality: {modality_name}")

    # Path to modality data file
    data_path = os.path.join(modality_dir, f"fold_{fold_idx}", f"{split}.csv")

    # Path to g-factor data file
    g_factor_path = os.path.join(base_dir, "g_factor", f"fold_{fold_idx}/g/g_{split}_regularized_{fold_idx}.csv")

    # Ensure both files exist
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Modality data file not found: {data_path}")

    if not os.path.exists(g_factor_path):
        raise FileNotFoundError(f"G-factor data file not found: {g_factor_path}")

    # Load modality data
    modality_df = pd.read_csv(data_path)

    # Load g-factor data
    g_factor_df = pd.read_csv(g_factor_path)

    # Merge on eid column
    merged_df = pd.merge(modality_df, g_factor_df[['eid', 'g']], on='eid', how='inner')

    if merged_df.empty:
        raise ValueError(f"No matching data between {modality_name} and g-factor for fold {fold_idx} {split}")

    # Store subject identifiers before any processing
    subject_ids = merged_df['eid'].values

    # Use the helper function to remove non-numeric columns
    merged_df, non_numeric_cols = verify_data_types(merged_df, modality_name)

    # Now get just the feature columns (all columns except ID and target columns)
    feature_cols = [col for col in merged_df.columns
                    if col not in ['eid', 'subject_id', 'g']]

    # Extract target (same for all modalities)
    y = merged_df['g'].values

    # Handle feature extraction based on modality
    if modality_name == "subcortical_volume":
        logger.info("Combining bilateral subcortical volumes")
        X, feature_cols = combine_bilateral_features(merged_df, feature_cols)
    elif modality_name == "cortical_area":
        logger.info("Combining bilateral cortical areas (lh_S_* + rh_S_*)")
        X, feature_cols = combine_bilateral_features(merged_df, feature_cols)
    else:
        X = merged_df[feature_cols].values

    # Impute missing values if any (after bilateral combination)
    X = impute_missing_values(X, feature_cols, modality_name)



    return X, y, feature_cols, subject_ids


def load_connectivity_data(fold_idx, split, base_dir):
    """
    Load functional connectivity data.
    Uses the original approach expecting connectivity files to be organized in fold/split directories.

    Parameters:
    -----------
    fold_idx : int
        Fold index
    split : str
        'train' or 'test'
    base_dir : str
        Base directory

    Returns:
    --------
    tuple : (X, y, feature_columns, subject_ids)
    """
    # Connectivity data directory
    conn_dir = os.path.join(base_dir, "functional/functional_connectivity")

    # Index file with timepoint IDs
    index_path = os.path.join(conn_dir, f"fold_{fold_idx}", f"{split}.csv")

    # G-factor data file
    g_factor_path = os.path.join(base_dir, "g_factor", f"fold_{fold_idx}/g/g_{split}_regularized_{fold_idx}.csv")

    # Ensure files exist
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Connectivity index file not found: {index_path}")

    if not os.path.exists(g_factor_path):
        raise FileNotFoundError(f"G-factor file not found: {g_factor_path}")

    # Load index and g-factor data
    index_df = pd.read_csv(index_path)
    g_factor_df = pd.read_csv(g_factor_path)

    # Process each connectivity matrix
    all_matrices = []
    common_subjects = []

    for _, row in index_df.iterrows():
        eid = row['eid']

        # Check if this eid is in g-factor data
        if eid in g_factor_df['eid'].values:
            # Path to connectivity matrix (original expected structure)
            conn_path = os.path.join(conn_dir, f"fold_{fold_idx}", split, f"{eid}_connectivity.csv")

            if os.path.exists(conn_path):
                # Load connectivity matrix
                conn_df = pd.read_csv(conn_path, index_col=0)

                # Verify data types using the helper function
                conn_df, non_numeric_cols = verify_data_types(conn_df, f"connectivity for subject {eid}")

                if non_numeric_cols:
                    logger.warning(
                        f"Removed {len(non_numeric_cols)} non-numeric columns from connectivity matrix for subject {eid}")

                # If we have any valid numeric data left
                if not conn_df.empty:
                    # Flatten the matrix
                    flat_matrix = flatten_connectivity_matrix(conn_df)

                    # Add to collection
                    all_matrices.append(flat_matrix)
                    common_subjects.append(eid)

    if not all_matrices:
        raise ValueError(f"No valid connectivity matrices found for fold {fold_idx} {split}")

    # Combine all matrices
    X = np.vstack(all_matrices)

    # Get corresponding g-factor values
    g_factor_values = []
    for eid in common_subjects:
        g_value = g_factor_df.loc[g_factor_df['eid'] == eid, 'g'].values[0]
        g_factor_values.append(g_value)

    y = np.array(g_factor_values)
    subject_ids = np.array(common_subjects)

    # Create feature column names based on the cleaned data shape
    n_regions = int((1 + np.sqrt(1 + 8 * X.shape[1])) / 2)
    feature_columns = [f"conn_{i}_{j}" for i in range(n_regions) for j in range(i)]

    # Impute missing values if any
    X = impute_missing_values(X, feature_columns, f"connectivity for {split}")

    return X, y, feature_columns, subject_ids


def train_evaluate_xgboost(X_train, y_train, X_test, y_test, feature_columns,
                           modality_name, fold_idx, output_dir, subject_ids_train, subject_ids_test):
    """Train and evaluate XGBoost model."""
    # Perform hyperparameter tuning
    tuning_results = perform_xgboost_tuning(
        X_train, y_train, feature_columns, modality_name, fold_idx
    )

    # Extract results
    model = tuning_results['model']
    selected_indices = tuning_results['selected_indices']
    selected_features = tuning_results['selected_features']

    # Apply feature selection to test data
    X_test_selected = X_test[:, selected_indices]
    X_train_selected = X_train[:, selected_indices]

    # Make predictions
    y_train_pred = model.predict(X_train_selected)
    y_test_pred = model.predict(X_test_selected)

    # Evaluate metrics
    train_metrics = evaluate_metrics(y_train, y_train_pred, prefix="Train_")
    test_metrics = evaluate_metrics(y_test, y_test_pred, prefix="Test_")

    # Save model
    model_file = os.path.join(output_dir, f"{modality_name}_xgboost_model.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    # Save subject ID mapping
    subject_mapping_df = pd.DataFrame({
        'train_subject_ids': pd.Series(subject_ids_train),
        'train_true_g': y_train,
        'train_predicted_g': y_train_pred
    })
    subject_mapping_file = os.path.join(output_dir, f"{modality_name}_xgboost_train_subjects.csv")
    subject_mapping_df.to_csv(subject_mapping_file, index=False)

    # Save feature importance if available
    importance_df = None
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        importance_file = os.path.join(output_dir, f"{modality_name}_xgboost_importance.csv")
        importance_df.to_csv(importance_file, index=False)

    # Save predictions with subject IDs
    predictions_df = pd.DataFrame({
        'subject_id': subject_ids_test,
        'true_g_factor': y_test,
        'xgb_predicted': y_test_pred
    })
    pred_file = os.path.join(output_dir, f"{modality_name}_xgboost_predictions.csv")
    predictions_df.to_csv(pred_file, index=False)

    # Create prediction plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    pearson_r = pearsonr(y_test, y_test_pred)[0]
    plt.title(f'XGBoost Predictions (r = {pearson_r:.4f})')
    plt.xlabel('Actual g-factor')
    plt.ylabel('Predicted g-factor')

    plot_file = os.path.join(output_dir, f"{modality_name}_xgboost_predictions.png")
    plt.savefig(plot_file)
    plt.close()

    return {
        'model': model,
        'metrics': {**train_metrics, **test_metrics},
        'importance': importance_df.to_dict('records') if importance_df is not None else None,
        'n_features': len(selected_features),
        'best_params': tuning_results['best_params'],
        'predictions': {
            'train': y_train_pred,
            'test': y_test_pred
        },
        'subject_ids': {
            'train': subject_ids_train,
            'test': subject_ids_test
        }
    }


def train_evaluate_krr(X_train, y_train, X_test, y_test, feature_columns,
                       modality_name, fold_idx, output_dir, subject_ids_train, subject_ids_test):
    """Train and evaluate KRR model."""
    # Perform hyperparameter tuning
    tuning_results = perform_krr_tuning(
        X_train, y_train, feature_columns, modality_name, fold_idx
    )

    # Extract results
    model = tuning_results['model']
    selected_indices = tuning_results['selected_indices']
    selected_features = tuning_results['selected_features']

    # Apply feature selection to test data if applicable
    if selected_indices is not None:
        X_test_selected = X_test[:, selected_indices]
        X_train_selected = X_train[:, selected_indices]
    else:
        X_test_selected = X_test
        X_train_selected = X_train

    # Make predictions
    y_train_pred = model.predict(X_train_selected)
    y_test_pred = model.predict(X_test_selected)

    # Evaluate metrics
    train_metrics = evaluate_metrics(y_train, y_train_pred, prefix="Train_")
    test_metrics = evaluate_metrics(y_test, y_test_pred, prefix="Test_")

    # Save model
    model_file = os.path.join(output_dir, f"{modality_name}_krr_model.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    # Save subject ID mapping for training set
    subject_mapping_df = pd.DataFrame({
        'train_subject_ids': pd.Series(subject_ids_train),
        'train_true_g': y_train,
        'train_predicted_g': y_train_pred
    })
    subject_mapping_file = os.path.join(output_dir, f"{modality_name}_krr_train_subjects.csv")
    subject_mapping_df.to_csv(subject_mapping_file, index=False)

    # Save selected features if applicable
    if selected_indices is not None:
        features_file = os.path.join(output_dir, f"{modality_name}_krr_selected_features.csv")
        pd.DataFrame({'Feature': selected_features}).to_csv(features_file, index=False)

    # Save predictions with subject IDs
    predictions_df = pd.DataFrame({
        'subject_id': subject_ids_test,
        'true_g_factor': y_test,
        'krr_predicted': y_test_pred
    })
    pred_file = os.path.join(output_dir, f"{modality_name}_krr_predictions.csv")
    predictions_df.to_csv(pred_file, index=False)

    # Create prediction plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    pearson_r = pearsonr(y_test, y_test_pred)[0]
    plt.title(f'KRR Predictions (r = {pearson_r:.4f})')
    plt.xlabel('Actual g-factor')
    plt.ylabel('Predicted g-factor')

    plot_file = os.path.join(output_dir, f"{modality_name}_krr_predictions.png")
    plt.savefig(plot_file)
    plt.close()

    return {
        'model': model,
        'metrics': {**train_metrics, **test_metrics},
        'selected_features': selected_features if selected_indices is not None else feature_columns,
        'n_features': len(selected_features) if selected_indices is not None else len(feature_columns),
        'best_params': tuning_results['best_params'],
        'predictions': {
            'train': y_train_pred,
            'test': y_test_pred
        },
        'subject_ids': {
            'train': subject_ids_train,
            'test': subject_ids_test
        }
    }

def train_evaluate_rf(X_train, y_train, X_test, y_test, feature_columns,
                      modality_name, fold_idx, output_dir, subject_ids_train, subject_ids_test):
    """Train and evaluate Random Forest model for cognitive prediction."""
    # Perform hyperparameter tuning
    tuning_results = perform_rf_tuning(
        X_train, y_train, feature_columns, modality_name, fold_idx
    )

    # Extract results
    model = tuning_results['model']
    selected_indices = tuning_results['selected_indices']
    selected_features = tuning_results['selected_features']

    # Apply feature selection to test data
    X_test_selected = X_test[:, selected_indices]
    X_train_selected = X_train[:, selected_indices]

    # Make predictions
    y_train_pred = model.predict(X_train_selected)
    y_test_pred = model.predict(X_test_selected)

    # Evaluate metrics
    train_metrics = evaluate_metrics(y_train, y_train_pred, prefix="Train_")
    test_metrics = evaluate_metrics(y_test, y_test_pred, prefix="Test_")

    # Save model
    model_file = os.path.join(output_dir, f"{modality_name}_rf_model.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    # Save subject ID mapping
    subject_mapping_df = pd.DataFrame({
        'train_subject_ids': pd.Series(subject_ids_train),
        'train_true_g': y_train,
        'train_predicted_g': y_train_pred
    })
    subject_mapping_file = os.path.join(output_dir, f"{modality_name}_rf_train_subjects.csv")
    subject_mapping_df.to_csv(subject_mapping_file, index=False)

    # Save feature importance
    importance_df = None
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        importance_file = os.path.join(output_dir, f"{modality_name}_rf_importance.csv")
        importance_df.to_csv(importance_file, index=False)

    # Save predictions with subject IDs
    predictions_df = pd.DataFrame({
        'subject_id': subject_ids_test,
        f'true_g': y_test,
        f'rf_predicted_g': y_test_pred
    })
    pred_file = os.path.join(output_dir, f"{modality_name}_rf_predictions.csv")
    predictions_df.to_csv(pred_file, index=False)

    # Create prediction plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    pearson_r = pearsonr(y_test, y_test_pred)[0]
    plt.title(f'Random Forest G-score Predictions (r = {pearson_r:.4f})')
    plt.xlabel(f'Actual G-score')
    plt.ylabel(f'Predicted G-score')

    plot_file = os.path.join(output_dir, f"{modality_name}_g_rf_predictions.png")
    plt.savefig(plot_file)
    plt.close()

    return {
        'model': model,
        'metrics': {**train_metrics, **test_metrics},
        'importance': importance_df.to_dict('records') if importance_df is not None else None,
        'n_features': len(selected_features),
        'best_params': tuning_results['best_params'],
        'predictions': {
            'train': y_train_pred,
            'test': y_test_pred
        },
        'subject_ids': {
            'train': subject_ids_train,
            'test': subject_ids_test
        }
    }

def train_evaluate_pls(X_train, y_train, X_test, y_test, feature_columns,
                       modality_name, fold_idx, output_dir, subject_ids_train, subject_ids_test):
    """Train and evaluate PLS model."""
    # Perform hyperparameter tuning
    tuning_results = perform_pls_tuning(
        X_train, y_train, feature_columns, modality_name, fold_idx
    )

    # Extract results
    model = tuning_results['model']
    n_components = tuning_results['n_components']
    selected_indices = tuning_results['selected_indices']
    selected_features = tuning_results['selected_features']

    # Apply feature selection to test data if applicable
    if selected_indices is not None:
        X_test_selected = X_test[:, selected_indices]
        X_train_selected = X_train[:, selected_indices]
    else:
        X_test_selected = X_test
        X_train_selected = X_train

    # Make predictions
    y_train_pred = model.predict(X_train_selected).flatten()
    y_test_pred = model.predict(X_test_selected).flatten()

    # Evaluate metrics
    train_metrics = evaluate_metrics(y_train, y_train_pred, prefix="Train_")
    test_metrics = evaluate_metrics(y_test, y_test_pred, prefix="Test_")

    # Save model
    model_file = os.path.join(output_dir, f"{modality_name}_pls_model.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    # Save subject ID mapping for training set
    subject_mapping_df = pd.DataFrame({
        'train_subject_ids': pd.Series(subject_ids_train),
        'train_true_g': y_train,
        'train_predicted_g': y_train_pred
    })
    subject_mapping_file = os.path.join(output_dir, f"{modality_name}_pls_train_subjects.csv")
    subject_mapping_df.to_csv(subject_mapping_file, index=False)

    # Save selected features and component details
    component_info = {
        'n_components': n_components,
        'n_features': len(selected_features)
    }
    with open(os.path.join(output_dir, f"{modality_name}_pls_components.json"), 'w') as f:
        import json
        json.dump(component_info, f)

    # Calculate VIP scores (feature importance for PLS)
    def get_pls_feature_importance(pls_model, feature_names):
        """Calculate VIP scores for PLS features."""
        if not hasattr(pls_model, 'x_weights_'):
            return None

        # Get matrix dimensions
        t = pls_model.x_scores_  # (n_samples, n_components)
        w = pls_model.x_weights_  # (n_features, n_components)
        q = pls_model.y_loadings_  # (n_targets, n_components)

        # Number of features and components
        n_features = w.shape[0]
        n_components = w.shape[1]  # Number of components

        # Initialize VIP scores
        vips = np.zeros((n_features,))

        # Sum of squared Y-loadings (adjusted for dimension)
        s = np.diag(t.T @ t @ q.T @ q).reshape(-1, 1)  # (n_components, 1)
        total_ss = np.sum(s)

        # Calculate VIP scores for each feature
        for i in range(n_features):
            # Get weights for feature i across all components
            feature_weights = w[i, :]  # (n_components,)

            # Calculate weighted sum for this feature
            weight_sum = 0
            for j in range(n_components):
                weight_sum += (feature_weights[j] ** 2) * s[j, 0]

            # Calculate VIP score
            vips[i] = np.sqrt(n_features * weight_sum / total_ss)

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': vips
        }).sort_values('Importance', ascending=False)

        return importance_df

    # Get PLS feature importance
    importance_df = get_pls_feature_importance(model, selected_features)

    if importance_df is not None:
        importance_file = os.path.join(output_dir, f"{modality_name}_pls_importance.csv")
        importance_df.to_csv(importance_file, index=False)

    # Save predictions with subject IDs
    predictions_df = pd.DataFrame({
        'subject_id': subject_ids_test,
        'true_g_factor': y_test,
        'pls_predicted': y_test_pred
    })
    pred_file = os.path.join(output_dir, f"{modality_name}_pls_predictions.csv")
    predictions_df.to_csv(pred_file, index=False)

    # Create prediction plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    pearson_r = pearsonr(y_test, y_test_pred)[0]
    plt.title(f'PLS Predictions (r = {pearson_r:.4f})')
    plt.xlabel('Actual g-factor')
    plt.ylabel('Predicted g-factor')

    plot_file = os.path.join(output_dir, f"{modality_name}_pls_predictions.png")
    plt.savefig(plot_file)
    plt.close()

    return {
        'model': model,
        'metrics': {**train_metrics, **test_metrics},
        'n_components': n_components,
        'importance': importance_df.to_dict('records') if importance_df is not None else None,
        'selected_features': selected_features,
        'n_features': len(selected_features),
        'predictions': {
            'train': y_train_pred,
            'test': y_test_pred
        },
        'subject_ids': {
            'train': subject_ids_train,
            'test': subject_ids_test
        }
    }



def process_modality_fold_model(modality_name, fold_idx, model_name, base_dir, output_dir=None):
    """
    Process a specific modality, fold, and model with appropriate preprocessing.
    Updated for nested cross-validation.
    """
    logger.info(f"Processing modality: {modality_name}, fold: {fold_idx}, model: {model_name}")
    logger.info("Using nested cross-validation structure")

    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_BASE_DIR, f"{modality_name}_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Create fold-specific output directory
    fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_output_dir, exist_ok=True)

    # Load data
    try:
        # Handle functional connectivity specially
        if modality_name == "functional_connectivity":
            X_train, y_train, feature_columns, subject_ids_train = load_connectivity_data(fold_idx, "train", base_dir)
            X_test, y_test, _, subject_ids_test = load_connectivity_data(fold_idx, "test", base_dir)
        else:
            # Handle all other modalities
            X_train, y_train, feature_columns, subject_ids_train = load_standard_modality_data(modality_name, fold_idx,
                                                                                               "train", base_dir)
            X_test, y_test, _, subject_ids_test = load_standard_modality_data(modality_name, fold_idx, "test", base_dir)

        logger.info(
            f"Loaded data for {modality_name}: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples with {X_train.shape[1]} features")

        # Validate nested CV structure
        validation_passed = validate_nested_cv_structure(X_train, X_test, subject_ids_train, subject_ids_test, fold_idx,
                                                         modality_name)
        if not validation_passed:
            logger.warning("Nested CV validation issues detected, but continuing with analysis")

        # Apply specialized preprocessing for connectivity data if needed
        if modality_name == "functional_connectivity":
            X_train, X_test, feature_columns = preprocess_connectivity_data(X_train, X_test, feature_columns,
                                                                            modality_name)

        # Train and evaluate the specified model
        if model_name == "xgboost":
            # Train XGBoost model
            results = train_evaluate_xgboost(
                X_train, y_train, X_test, y_test,
                feature_columns, modality_name, fold_idx,
                fold_output_dir, subject_ids_train, subject_ids_test
            )
        elif model_name == "krr":
            # Train KRR model
            results = train_evaluate_krr(
                X_train, y_train, X_test, y_test,
                feature_columns, modality_name, fold_idx,
                fold_output_dir, subject_ids_train, subject_ids_test
            )
        elif model_name == "pls":
            # Train PLS model
            results = train_evaluate_pls(
                X_train, y_train, X_test, y_test,
                feature_columns, modality_name, fold_idx,
                fold_output_dir, subject_ids_train, subject_ids_test
            )
        elif model_name == "rf":  # NEW: Add Random Forest case
            results = train_evaluate_rf(
                X_train, y_train, X_test, y_test,
                feature_columns, modality_name, fold_idx,
                fold_output_dir, subject_ids_train, subject_ids_test
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Save summary results
        summary = {
            'modality': modality_name,
            'fold': fold_idx,
            'model': model_name,
            'n_train_samples': X_train.shape[0],
            'n_test_samples': X_test.shape[0],
            'n_features': X_train.shape[1],
            'timestamp': timestamp,
            'nested_cv_validation_passed': validation_passed
        }

        # Add metrics to summary
        if 'metrics' in results:
            summary.update(results['metrics'])

        # Save summary
        summary_file = os.path.join(fold_output_dir, f"{modality_name}_{model_name}_summary.json")
        with open(summary_file, 'w') as f:
            import json
            json.dump(summary, f, indent=2)

        return results

    except Exception as e:
        logger.error(f"Error processing {modality_name}, fold {fold_idx}, model {model_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def check_correlation_matrix(X, feature_names, threshold=0.99):
    """
    Check for highly correlated features that might cause PLS issues.

    Parameters:
    -----------
    X : numpy array
        Feature matrix
    feature_names : list
        List of feature names
    threshold : float
        Correlation threshold for flagging pairs

    Returns:
    --------
    list : List of high correlation pairs
    """
    logger.info(f"Checking feature correlations with threshold {threshold}")

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X.T)

    # Find pairs with correlation > threshold
    high_corr_pairs = []
    n_features = len(feature_names)

    logger.info(f"Analyzing {n_features} features for high correlations")

    for i in range(n_features):
        for j in range(i + 1, n_features):
            if abs(corr_matrix[i, j]) > threshold:
                high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))

    logger.info(f"Found {len(high_corr_pairs)} feature pairs with |correlation| > {threshold}")

    if high_corr_pairs:
        logger.warning("High correlation pairs detected (showing first 10):")
        for i, pair in enumerate(high_corr_pairs[:10]):
            logger.warning(f"  {pair[0]} <-> {pair[1]}: {pair[2]:.4f}")

        if len(high_corr_pairs) > 10:
            logger.info(f"... and {len(high_corr_pairs) - 10} more pairs")

        # Log some statistics about correlations
        correlations = [abs(pair[2]) for pair in high_corr_pairs]
        logger.info(f"High correlation stats - Max: {max(correlations):.4f}, "
                    f"Mean: {np.mean(correlations):.4f}, "
                    f"Min: {min(correlations):.4f}")
    else:
        logger.info("No highly correlated feature pairs found")

    # Check matrix rank
    try:
        rank = np.linalg.matrix_rank(X)
        logger.info(f"Feature matrix rank: {rank} (out of {min(X.shape)} possible)")

        if rank < min(X.shape):
            logger.warning(f"Matrix is rank deficient! Rank={rank}, Shape={X.shape}")
            logger.warning("This will cause issues with PLS - consider reducing max_components")
        else:
            logger.info("Matrix has full rank - good for PLS")

    except Exception as e:
        logger.error(f"Could not compute matrix rank: {e}")

    return high_corr_pairs


def combine_bilateral_features(df, feature_columns):
    """
    Combine left/right hemisphere features by taking their mean.
    Handles both traditional 'left'/'right' naming and neuroimaging 'lh_'/'rh_' patterns.
    Common practice in neuroimaging when lateralization isn't the focus.
    """
    combined_features = []
    combined_names = []
    used_features = set()

    for col in feature_columns:
        if col in used_features:
            continue

        # Handle neuroimaging lh_* / rh_* pattern (any lh/rh pair)
        if col.startswith('lh_'):
            # Extract the region name after lh_
            region_name = col[3:]  # Remove 'lh_' prefix
            rh_col = f'rh_{region_name}'

            if rh_col in feature_columns and rh_col not in used_features:
                # Average left and right hemispheres
                combined = (df[col] + df[rh_col]) / 2
                combined_features.append(combined)
                combined_names.append(f'bilateral_{region_name}')
                used_features.add(col)
                used_features.add(rh_col)
                logger.info(f"Combined {col} + {rh_col} → bilateral_{region_name}")
                continue

        elif col.startswith('rh_'):
            # Extract the region name after rh_
            region_name = col[3:]  # Remove 'rh_' prefix
            lh_col = f'lh_{region_name}'

            if lh_col in feature_columns and lh_col not in used_features:
                # Average left and right hemispheres
                combined = (df[lh_col] + df[col]) / 2
                combined_features.append(combined)
                combined_names.append(f'bilateral_{region_name}')
                used_features.add(lh_col)
                used_features.add(col)
                logger.info(f"Combined {lh_col} + {col} → bilateral_{region_name}")
                continue

        # Handle traditional left/right pattern (existing code)
        elif 'left' in col.lower():
            right_col = col.lower().replace('left', 'right')
            right_col = next((c for c in feature_columns if c.lower() == right_col), None)

            if right_col:
                # Average left and right
                combined = (df[col] + df[right_col]) / 2
                combined_features.append(combined)
                combined_names.append(col.replace('left', 'bilateral').replace('Left', 'Bilateral'))
                used_features.add(col)
                used_features.add(right_col)
                logger.info(f"Combined {col} + {right_col} → {combined_names[-1]}")
                continue

        elif 'right' in col.lower():
            left_col = col.lower().replace('right', 'left')
            left_col = next((c for c in feature_columns if c.lower() == left_col), None)

            if left_col and left_col not in used_features:
                # Average left and right
                combined = (df[left_col] + df[col]) / 2
                combined_features.append(combined)
                combined_names.append(left_col.replace('left', 'bilateral').replace('Left', 'Bilateral'))
                used_features.add(left_col)
                used_features.add(col)
                logger.info(f"Combined {left_col} + {col} → {combined_names[-1]}")
                continue

        # If no pair found, keep original
        if col not in used_features:
            combined_features.append(df[col])
            combined_names.append(col)
            used_features.add(col)

    logger.info(f"Feature reduction: {len(feature_columns)} → {len(combined_names)} features")
    return np.column_stack(combined_features), combined_names


def main():
    """Main function to build modality-specific prediction models."""
    logger.info("Starting First Level Model Training with Nested Cross-Validation")

    # Parse command line arguments
    args = parse_args()

    # Check for job_index and get parameters if provided
    if args.job_index is not None:
        fold, model = get_job_parameters_modality_specific(args.job_index)
    else:
        # Otherwise use direct specification
        fold = args.fold
        model = args.model

        # Handle defaults if not specified
        if fold is None:
            fold = 0
            logger.info(f"No fold specified, using default: {fold}")

        if model is None:
            model = MODELS[0]
            logger.info(f"No model specified, using default: {model}")

    modality = args.modality

    logger.info(f"Configuration: modality={modality}, fold={fold}, model={model}")

    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_BASE_DIR, f"{modality}_{model}_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Process the specified fold, model, and modality
    process_modality_fold_model(modality, fold, model, args.base_dir, output_dir)

    logger.info(f"Completed processing fold {fold}, model {model}, modality {modality}")
    logger.info("Nested cross-validation analysis complete")


if __name__ == "__main__":
    main()