import os
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import argparse
import logging
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import psutil
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stacked_model_ridge_{}.log".format(datetime.now().strftime("%Y%m%d_%H%M%S"))),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("stacked_model_ridge")

###########################################
#          CONFIGURATION SECTION          #
###########################################

# Default parameters
SEED = 42
N_CV_FOLDS = 5

# Default modalities and models
DEFAULT_MODALITIES = [
    "cortical_area",
    "cortical_thickness",
    "subcortical_volume",
    "total_brain_volume",
    #"alff",
    #"reho",
    "functional_connectivity"
]

DEFAULT_MODELS = ["xgboost", "krr", "pls"]
DEFAULT_FOLDS = [0, 1, 2, 3, 4]

# CONFIG VARIABLES FOR IDE TESTING
CONFIG_BASE_DIR = "/media/hcs-sci-psy-narun/Jack/First_level_model_aligned"
CONFIG_OUTPUT_DIR = "/media/hcs-sci-psy-narun/Jack/First_level_model_aligned/Stacked_model_ridge_v4_EXTREME_conserve_SUBTRACT_REHO_ALFF"

# Ridge regression parameter grids
RIDGE_PARAM_GRID_OPTIMIZED = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    'fit_intercept': [True],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
    'random_state': [SEED]
}

RIDGE_PARAM_GRID_SIMPLE_OPTIMIZED = {
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'fit_intercept': [True],
    'solver': ['auto'],
    'random_state': [SEED]
}

# More conservative Ridge grids for high regularization
RIDGE_PARAM_GRID_CONSERVATIVE = {
    'alpha': [10.0, 100.0, 1000.0, 5000.0, 10000.0],
    'fit_intercept': [True],
    'solver': ['auto', 'svd', 'cholesky'],
    'random_state': [SEED]
}

RIDGE_PARAM_GRID_VERY_CONSERVATIVE = {
    'alpha': [100.0, 1000.0, 5000.0],
    'fit_intercept': [True],
    'solver': ['auto'],
    'random_state': [SEED]
}

# Ultra-conservative Ridge grid for extreme regularization
RIDGE_PARAM_GRID_ULTRA_CONSERVATIVE = {
    'alpha': [1000.0, 2000.0, 4000.0, 5000.0, 10000.0, 50000.0, 100000],
    'fit_intercept': [True],
    'solver': ['auto'],
    'random_state': [SEED]
}

###########################################
#           END CONFIGURATION             #
###########################################

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Build stacked model with Ridge regression from first-level predictions')

    parser.add_argument('--base_dir', type=str, required=False, default=CONFIG_BASE_DIR,
                        help='Base directory containing first-level model results')

    parser.add_argument('--output_dir', type=str, required=False, default=CONFIG_OUTPUT_DIR,
                        help='Output directory for stacked model results.')

    parser.add_argument('--modalities', nargs='+', default=DEFAULT_MODALITIES,
                        choices=DEFAULT_MODALITIES,
                        help='Neuroimaging modalities to include in stacking')

    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        choices=DEFAULT_MODELS + ['rf'],
                        help='First-level models to include in stacking')

    parser.add_argument('--folds', nargs='+', type=int, default=DEFAULT_FOLDS,
                        help='Fold indices to process')

    parser.add_argument('--tune_hyperparams', action='store_true',
                        help='Perform hyperparameter tuning for Ridge regression stacker')

    parser.add_argument('--simple_tuning', action='store_true',
                        help='Use simplified parameter grid for faster tuning')

    parser.add_argument('--conservative_tuning', action='store_true',
                        help='Use conservative parameter grid with high regularization')

    parser.add_argument('--very_conservative_tuning', action='store_true',
                        help='Use very conservative parameter grid with very high regularization')

    parser.add_argument('--ultra_conservative_tuning', action='store_true',
                        help='Use ultra conservative parameter grid with extreme regularization')

    parser.add_argument('--max_workers', type=int, default=32,
                        help='Number of parallel workers for file loading')

    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Number of cross-validation folds for hyperparameter tuning')

    parser.add_argument('--scaling', choices=['standard', 'none'], default='standard',
                        help='Feature scaling method (recommended: standard for Ridge)')

    parser.add_argument('--imputation', choices=['median', 'mean', 'none'], default='median',
                        help='Missing value imputation strategy')

    return parser.parse_args()


class OptimizedStackedModelBuilder:
    """Memory-optimized class for stacked model creation and evaluation using Ridge regression."""

    def __init__(self, args):
        """Initialize with args object."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.base_results_dir = Path(args.base_dir)
        self.output_dir = Path(args.output_dir or f"stacked_results_ridge_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store configuration parameters
        self.modalities = args.modalities
        self.models = args.models
        self.folds = args.folds
        self.tune_hyperparams = args.tune_hyperparams
        self.simple_tuning = args.simple_tuning
        self.conservative_tuning = args.conservative_tuning
        self.very_conservative_tuning = args.very_conservative_tuning
        self.ultra_conservative_tuning = args.ultra_conservative_tuning
        self.max_workers = args.max_workers
        self.cv_folds = args.cv_folds
        self.scaling = args.scaling
        self.imputation = args.imputation

        # Initialize memory monitoring
        self.process = psutil.Process(os.getpid())
        self.log_memory_usage("Initialization")

    def log_memory_usage(self, step_name):
        """Log current memory usage."""
        memory_gb = self.process.memory_info().rss / 1024 / 1024 / 1024
        logger.info(f"{step_name} - Memory usage: {memory_gb:.2f} GB")

    def _load_single_file(self, modality, model, fold_idx, split):
        """Load and standardize a single prediction file."""
        try:
            pred_file = self._find_prediction_file(modality, model, fold_idx, split)
            if pred_file is None:
                return None

            df = pd.read_csv(pred_file)
            standardized_df = self._standardize_columns(df, modality, model, split, fold_idx)

            return standardized_df

        except Exception as e:
            logger.error(f"Error loading {modality}-{model}-fold{fold_idx}-{split}: {e}")
            return None

    def load_fold_predictions_parallel(self, fold_idx, split='test'):
        """Load all modality-model files for a fold in parallel."""
        tasks = []
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all file loading tasks
            for modality in self.modalities:
                for model in self.models:
                    future = executor.submit(self._load_single_file, modality, model, fold_idx, split)
                    tasks.append((future, modality, model))

            # Collect results with progress bar
            desc = f"Loading {split} files for fold {fold_idx}"
            for future, modality, model in tqdm(tasks, desc=desc, leave=False):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        logger.debug(f"✓ Loaded {modality}-{model}-fold{fold_idx}-{split}")
                except Exception as e:
                    logger.warning(f"✗ Failed {modality}-{model}-fold{fold_idx}-{split}: {e}")

        if not results:
            raise ValueError(f"No {split} predictions found for fold {fold_idx}")

        # Merge results efficiently
        combined_df = self._merge_predictions_efficiently(results, fold_idx)

        # Cleanup
        del results
        gc.collect()

        return combined_df

    def _merge_predictions_efficiently(self, prediction_dfs, fold_idx):
        """Efficiently merge prediction dataframes."""
        if not prediction_dfs:
            return None

        # Start with base columns
        base_df = prediction_dfs[0][['subject_id', 'true_g_factor']].copy()

        # Collect all prediction columns
        pred_data = {}
        for df in prediction_dfs:
            pred_col = [col for col in df.columns if col not in ['subject_id', 'true_g_factor', 'fold']][0]
            pred_data[pred_col] = df.set_index('subject_id')[pred_col]

        # Create prediction dataframe and merge
        pred_df = pd.DataFrame(pred_data)
        combined_df = base_df.set_index('subject_id').join(pred_df, how='inner').reset_index()

        # Add fold information
        combined_df['fold'] = fold_idx

        return combined_df

    def prepare_fold_data_memory_efficient(self, fold_idx):
        """Prepare data for a single fold with memory optimization."""
        try:
            self.log_memory_usage(f"Before loading fold {fold_idx}")

            # Load train and test predictions in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                train_future = executor.submit(self.load_fold_predictions_parallel, fold_idx, 'train')
                test_future = executor.submit(self.load_fold_predictions_parallel, fold_idx, 'test')

                train_predictions = train_future.result()
                test_predictions = test_future.result()

            # Extract features and targets
            feature_cols = [col for col in train_predictions.columns
                            if col not in ['subject_id', 'true_g_factor', 'fold']]

            fold_data = {
                'train': {
                    'X': train_predictions[feature_cols].values,
                    'y': train_predictions['true_g_factor'].values,
                    'subject_ids': train_predictions['subject_id'].values,
                    'feature_names': feature_cols,
                    'predictions_df': train_predictions
                },
                'test': {
                    'X': test_predictions[feature_cols].values,
                    'y': test_predictions['true_g_factor'].values,
                    'subject_ids': test_predictions['subject_id'].values,
                    'feature_names': feature_cols,
                    'predictions_df': test_predictions
                }
            }

            self.log_memory_usage(f"After loading fold {fold_idx}")
            logger.info(
                f"Prepared fold {fold_idx}: {len(fold_data['train']['X'])} train, {len(fold_data['test']['X'])} test samples")

            return fold_data

        except Exception as e:
            logger.error(f"Error preparing fold {fold_idx}: {e}")
            return None

    def train_stacked_model_memory_efficient(self):
        """
        Train stacked model using Ridge regression: each fold's stacker trained on
        that fold's own train data and tested on that fold's test data.

        This eliminates data leakage and distribution mismatch.
        """
        logger.info("Training stacked Ridge regression model...")

        fold_results = []
        total_folds = len(self.folds)

        # Process each fold independently
        for i, fold_idx in enumerate(tqdm(self.folds, desc="Training folds")):
            logger.info(f"Processing fold {fold_idx} ({i + 1}/{total_folds})")

            # Load this fold's train and test data
            fold_data = self.prepare_fold_data_memory_efficient(fold_idx)
            if fold_data is None:
                logger.warning(f"Could not load data for fold {fold_idx}")
                continue

            # Get train and test data for this fold
            X_train = fold_data['train']['X']
            y_train = fold_data['train']['y']
            train_subject_ids = fold_data['train']['subject_ids']
            feature_names = fold_data['train']['feature_names']

            X_test = fold_data['test']['X']
            y_test = fold_data['test']['y']
            test_subject_ids = fold_data['test']['subject_ids']

            self.log_memory_usage(f"Before training fold {fold_idx}")

            # Train model for this fold using its own train/test split
            fold_result = self._train_single_fold(
                X_train, y_train, X_test, y_test,
                test_subject_ids, feature_names, fold_idx
            )

            if fold_result is not None:
                fold_results.append(fold_result)

            # Cleanup fold data
            del fold_data, X_train, y_train, X_test, y_test
            gc.collect()

            self.log_memory_usage(f"After training fold {fold_idx}")

        return fold_results

    def _train_single_fold(self, X_train, y_train, X_test, y_test,
                           test_subject_ids, feature_names, fold_idx):
        """Train Ridge regression model for a single fold using that fold's own train/test data."""
        try:
            # Handle missing values
            imputer = SimpleImputer(strategy=self.imputation if self.imputation != 'none' else 'median')
            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)

            # Scale features (highly recommended for Ridge regression)
            if self.scaling == 'standard':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_imputed)
                X_test_scaled = scaler.transform(X_test_imputed)
                logger.info(f"Features standardized for Ridge regression (fold {fold_idx})")
            else:
                X_train_scaled = X_train_imputed
                X_test_scaled = X_test_imputed
                scaler = None
                logger.warning(f"No scaling applied - Ridge performance may be suboptimal (fold {fold_idx})")

            # Choose parameter grid based on user preferences
            if self.tune_hyperparams:
                logger.info(f"Performing hyperparameter tuning for Ridge regression (fold {fold_idx})...")

                if self.ultra_conservative_tuning:
                    param_grid = RIDGE_PARAM_GRID_ULTRA_CONSERVATIVE
                    grid_name = "ultra conservative (extreme regularization)"
                elif self.very_conservative_tuning:
                    param_grid = RIDGE_PARAM_GRID_VERY_CONSERVATIVE
                    grid_name = "very conservative (high regularization)"
                elif self.conservative_tuning:
                    param_grid = RIDGE_PARAM_GRID_CONSERVATIVE
                    grid_name = "conservative"
                elif self.simple_tuning:
                    param_grid = RIDGE_PARAM_GRID_SIMPLE_OPTIMIZED
                    grid_name = "simple"
                else:
                    param_grid = RIDGE_PARAM_GRID_OPTIMIZED
                    grid_name = "full"

                logger.info(f"Using {grid_name} parameter grid for Ridge regression")

                ridge_base = Ridge(random_state=SEED)
                ridge_stacked = GridSearchCV(
                    estimator=ridge_base,
                    param_grid=param_grid,
                    cv=self.cv_folds,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=1
                )

                logger.info(
                    f"Training Ridge stacker for fold {fold_idx} with {len(X_train)} samples, {X_train.shape[1]} features")
                logger.info(f"Using corrected approach: training on fold {fold_idx}'s own train data")
                ridge_stacked.fit(X_train_scaled, y_train)

                logger.info(f"Best parameters for fold {fold_idx}: {ridge_stacked.best_params_}")
                logger.info(f"Best CV score: {ridge_stacked.best_score_:.4f}")

                # Get the best model for coefficient extraction
                best_ridge = ridge_stacked.best_estimator_

            else:
                # Use default parameters with moderate regularization
                ridge_stacked = Ridge(
                    alpha=1.0,  # Moderate regularization
                    fit_intercept=True,
                    random_state=SEED
                )

                logger.info(
                    f"Training Ridge stacker for fold {fold_idx} with {len(X_train)} samples, {X_train.shape[1]} features")
                logger.info(f"Using corrected approach: training on fold {fold_idx}'s own train data")
                logger.info(f"Using default Ridge parameters (alpha=1.0)")
                ridge_stacked.fit(X_train_scaled, y_train)

                best_ridge = ridge_stacked

            # Make predictions
            y_train_pred = ridge_stacked.predict(X_train_scaled)
            y_test_pred = ridge_stacked.predict(X_test_scaled)

            # Calculate Haufe-transformed feature importance (correlation-based)
            haufe_importance = self.calculate_haufe_importance(
                X_test_scaled, y_test_pred, feature_names
            )

            # Calculate Ridge coefficient-based importance
            ridge_importance = self.calculate_ridge_importance(
                best_ridge, feature_names
            )

            # Combine Haufe and Ridge importance
            feature_importance = haufe_importance.merge(ridge_importance, on='feature', how='outer')

            # Calculate metrics
            train_metrics = self.calculate_metrics(y_train, y_train_pred)
            test_metrics = self.calculate_metrics(y_test, y_test_pred)

            fold_result = {
                'test_fold': fold_idx,
                'model': ridge_stacked,
                'scaler': scaler,
                'imputer': imputer,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance,
                'predictions': {
                    'subject_ids': test_subject_ids,
                    'y_true': y_test,
                    'y_pred': y_test_pred
                },
                'alpha_used': best_ridge.alpha,
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test)
            }

            logger.info(f"Fold {fold_idx} - Test R²: {test_metrics['r2']:.4f}, "
                        f"Test Pearson r: {test_metrics['pearson_r']:.4f}")
            logger.info(f"  Alpha used: {best_ridge.alpha}")
            logger.info(f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}")

            return fold_result

        except Exception as e:
            logger.error(f"Error training fold {fold_idx}: {e}")
            return None

    def calculate_ridge_importance(self, ridge_model, feature_names):
        """
        Calculate feature importance based on Ridge regression coefficients.

        Parameters:
        -----------
        ridge_model : Ridge
            Trained Ridge regression model
        feature_names : list
            Names of the features

        Returns:
        --------
        pd.DataFrame : DataFrame with features and their Ridge coefficient importance
        """
        # Get coefficients from the Ridge model
        coefficients = ridge_model.coef_

        ridge_df = pd.DataFrame({
            'feature': feature_names,
            'ridge_coef': coefficients,
            'ridge_coef_abs': np.abs(coefficients)
        }).sort_values('ridge_coef_abs', ascending=False)

        logger.info("Ridge coefficient importance calculated")
        return ridge_df

    # Keep the existing utility methods
    def _find_prediction_file(self, modality, model, fold_idx, split):
        """Find prediction file using flexible directory patterns."""
        if split == 'train':
            filename = f"{modality}_{model}_train_subjects.csv"
        else:
            filename = f"{modality}_{model}_predictions.csv"

        if modality == 'functional_connectivity':
            patterns = [
                self.base_results_dir / 'model_results' / modality / model / f'fold_{fold_idx}' / f'fold_{fold_idx}' / filename,
                self.base_results_dir / 'model_results' / modality / model / f'fold_{fold_idx}' / filename,
            ]
        else:
            patterns = [
                self.base_results_dir / 'model_results' / modality / f'fold_{fold_idx}' / filename,
            ]

        for pattern in patterns:
            if pattern.exists():
                return str(pattern)

        for pattern_str in [
            f"**/model_results/{modality}/**/fold_{fold_idx}**/{filename}",
            f"**/model_results/{modality}/**/{filename}",
        ]:
            matches = list(self.base_results_dir.rglob(pattern_str))
            fold_matches = [m for m in matches if f'fold_{fold_idx}' in str(m)]
            if fold_matches:
                return str(fold_matches[0])

        return None

    def _standardize_columns(self, df, modality, model, split, fold_idx):
        """Standardize column names and extract relevant data."""
        try:
            if split == 'train':
                required_cols = ['train_subject_ids', 'train_true_g', 'train_predicted_g']
                if not all(col in df.columns for col in required_cols):
                    logger.warning(f"Missing expected train columns in {modality}_{model}_fold{fold_idx}")
                    logger.warning(f"Available columns: {list(df.columns)}")
                    return None

                standardized_df = pd.DataFrame({
                    'subject_id': df['train_subject_ids'],
                    'true_g_factor': df['train_true_g'],
                    f'{modality}_{model}': df['train_predicted_g'],
                })

            else:  # test
                subject_col = 'subject_id'
                true_g_col = 'true_g_factor'

                # Handle model name variations for prediction column
                possible_pred_cols = [
                    f'{model}_predicted',  # e.g., 'xgboost_predicted'
                    f'{model[:3]}_predicted',  # e.g., 'xgb_predicted' for xgboost
                    f'{model.replace("oost", "")}_predicted',  # 'xgb_predicted' for xgboost
                ]

                # Find the actual prediction column
                pred_col = None
                for col in possible_pred_cols:
                    if col in df.columns:
                        pred_col = col
                        break

                # If still not found, look for any column ending with '_predicted'
                if pred_col is None:
                    pred_candidates = [col for col in df.columns if col.endswith('_predicted')]
                    if pred_candidates:
                        pred_col = pred_candidates[0]
                        logger.info(f"Using prediction column: {pred_col} for {modality}_{model}")

                if pred_col is None:
                    logger.warning(f"No prediction column found for {modality}_{model}_fold{fold_idx}")
                    logger.warning(f"Available columns: {list(df.columns)}")
                    return None

                # Check required columns
                if not all(col in df.columns for col in [subject_col, true_g_col]):
                    logger.warning(f"Missing subject_id or true_g_factor in {modality}_{model}_fold{fold_idx}")
                    logger.warning(f"Available columns: {list(df.columns)}")
                    return None

                standardized_df = pd.DataFrame({
                    'subject_id': df[subject_col],
                    'true_g_factor': df[true_g_col],
                    f'{modality}_{model}': df[pred_col],
                })

            standardized_df['fold'] = fold_idx
            return standardized_df

        except Exception as e:
            logger.error(f"Error standardizing columns for {modality}_{model}_fold{fold_idx}: {e}")
            return None

    def calculate_haufe_importance(self, X, y_pred, feature_names):
        """
        Calculate Haufe-transformed feature importance using correlation between
        stacked predictions and first-level model predictions.

        Parameters:
        -----------
        X : array
            First-level model predictions (features)
        y_pred : array
            Stacked model predictions
        feature_names : list
            Names of the features (modality_model combinations)

        Returns:
        --------
        pd.DataFrame : DataFrame with features and their Haufe importance scores
        """
        haufe_scores = []

        for i, feature_name in enumerate(feature_names):
            # Calculate Pearson correlation between feature and stacked prediction
            feature_values = X[:, i]

            # Skip features with no variance
            if np.std(feature_values) == 0:
                correlation = 0.0
                logger.warning(f"Zero variance in feature {feature_name}, setting Haufe importance to 0")
            else:
                correlation = np.corrcoef(feature_values, y_pred)[0, 1]

                # Handle NaN correlations (shouldn't happen with proper data)
                if np.isnan(correlation):
                    correlation = 0.0
                    logger.warning(f"NaN correlation for feature {feature_name}, setting to 0")

            haufe_scores.append(correlation)

        # Create DataFrame with Haufe importance
        haufe_df = pd.DataFrame({
            'feature': feature_names,
            'haufe_importance': haufe_scores,
            'haufe_importance_abs': [abs(score) for score in haufe_scores]
        }).sort_values('haufe_importance_abs', ascending=False)

        logger.info("Haufe importance calculated based on correlation with stacked predictions")
        return haufe_df

    def calculate_metrics(self, y_true, y_pred):
        """Calculate performance metrics."""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'pearson_r': pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else 0
        }

    def save_results(self, fold_results):
        """Save stacked Ridge model results."""
        all_test_predictions = []
        all_metrics = []
        all_importance = []

        for result in fold_results:
            test_pred_df = pd.DataFrame({
                'fold': result['test_fold'],
                'subject_id': result['predictions']['subject_ids'],
                'true_g': result['predictions']['y_true'],
                'stacked_pred': result['predictions']['y_pred'],
                'split': 'test'
            })
            all_test_predictions.append(test_pred_df)

            metrics_df = pd.DataFrame([{
                'fold': result['test_fold'],
                'split': 'train',
                **result['train_metrics']
            }, {
                'fold': result['test_fold'],
                'split': 'test',
                **result['test_metrics']
            }])
            all_metrics.append(metrics_df)

            imp_df = result['feature_importance'].copy()
            imp_df['fold'] = result['test_fold']
            imp_df['alpha_used'] = result['alpha_used']
            all_importance.append(imp_df)

        # Save combined results
        all_predictions = pd.concat(all_test_predictions, ignore_index=True)
        all_predictions.to_csv(self.output_dir / 'stacked_ridge_predictions.csv', index=False)

        pd.concat(all_metrics, ignore_index=True).to_csv(
            self.output_dir / 'stacked_ridge_metrics.csv', index=False
        )

        pd.concat(all_importance, ignore_index=True).to_csv(
            self.output_dir / 'stacked_ridge_feature_importance.csv', index=False
        )

        # Calculate overall performance
        all_test_df = pd.concat(all_test_predictions, ignore_index=True)
        overall_test_metrics = self.calculate_metrics(
            all_test_df['true_g'], all_test_df['stacked_pred']
        )

        logger.info("\n" + "=" * 50)
        logger.info("STACKED RIDGE MODEL OVERALL PERFORMANCE")
        logger.info("=" * 50)
        logger.info("TEST SET:")
        for metric, value in overall_test_metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")

        # Save overall metrics and config
        with open(self.output_dir / 'stacked_ridge_overall_metrics.txt', 'w') as f:
            f.write("STACKED RIDGE MODEL OVERALL PERFORMANCE\n")
            f.write("=" * 50 + "\n")
            f.write("TEST SET:\n")
            for metric, value in overall_test_metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")

        config_info = {
            'modalities': self.modalities,
            'models': self.models,
            'folds': self.folds,
            'tune_hyperparams': self.tune_hyperparams,
            'conservative_tuning': self.conservative_tuning,
            'very_conservative_tuning': self.very_conservative_tuning,
            'ultra_conservative_tuning': self.ultra_conservative_tuning,
            'scaling': self.scaling,
            'imputation': self.imputation,
            'max_workers': self.max_workers,
            'cv_folds': self.cv_folds,
            'n_features_total': len(fold_results[0]['feature_importance']) if fold_results else 0,
            'n_test_samples_total': len(all_test_df) if all_test_predictions else 0
        }

        with open(self.output_dir / 'stacked_ridge_config.txt', 'w') as f:
            f.write("STACKED RIDGE MODEL CONFIGURATION\n")
            f.write("=" * 50 + "\n")
            for key, value in config_info.items():
                f.write(f"{key}: {value}\n")

        return {'test_metrics': overall_test_metrics}

    def create_visualizations(self, fold_results):
        """Create visualization plots for stacked Ridge model results."""
        all_test_true = []
        all_test_pred = []

        for result in fold_results:
            all_test_true.extend(result['predictions']['y_true'])
            all_test_pred.extend(result['predictions']['y_pred'])

        # Test prediction plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.scatter(all_test_true, all_test_pred, alpha=0.6, s=30, color='lightcoral')
        ax.plot([min(all_test_true), max(all_test_true)], [min(all_test_true), max(all_test_true)], 'r--', lw=2)
        ax.set_xlabel('True g-factor', fontsize=12)
        ax.set_ylabel('Predicted g-factor', fontsize=12)

        test_r = pearsonr(all_test_true, all_test_pred)[0]
        test_r2 = r2_score(all_test_true, all_test_pred)
        ax.set_title(f'Stacked Ridge Model Test Set Predictions\n(r = {test_r:.3f}, R² = {test_r2:.3f})', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'stacked_ridge_predictions_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Feature importance plot (using Haufe transformation)
        all_importance = []
        for result in fold_results:
            imp_df = result['feature_importance'].copy()
            imp_df['fold'] = result['test_fold']
            all_importance.append(imp_df)

        importance_combined = pd.concat(all_importance, ignore_index=True)

        # Use Haufe importance (absolute values for ranking)
        if 'haufe_importance_abs' in importance_combined.columns:
            avg_importance = importance_combined.groupby('feature')['haufe_importance_abs'].mean().sort_values(
                ascending=True)
            importance_type = "Haufe Importance (|correlation|)"
        else:
            # Fallback to Ridge coefficient importance if Haufe not available
            avg_importance = importance_combined.groupby('feature')['ridge_coef_abs'].mean().sort_values(ascending=True)
            importance_type = "Ridge Coefficient Importance (|coeff|)"

        plt.figure(figsize=(12, max(8, len(avg_importance) * 0.3)))

        colors = []
        for feature in avg_importance.index:
            if any(mod in feature for mod in ['cortical_area', 'cortical_thickness']):
                colors.append('lightcoral')
            elif any(mod in feature for mod in ['subcortical', 'total_brain']):
                colors.append('lightblue')
            elif any(mod in feature for mod in ['alff', 'reho', 'functional_connectivity']):
                colors.append('lightgreen')
            else:
                colors.append('lightgray')

        bars = plt.barh(range(len(avg_importance)), avg_importance.values, color=colors)
        plt.yticks(range(len(avg_importance)), avg_importance.index, fontsize=10)
        plt.xlabel(f'Average {importance_type}', fontsize=12)
        plt.title(f'Stacked Ridge Model Feature Importance\n({importance_type}, Averaged Across Folds)', fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightcoral', label='Structural (Cortical)'),
            Patch(facecolor='lightblue', label='Structural (Subcortical)'),
            Patch(facecolor='lightgreen', label='Functional')
        ]
        plt.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'stacked_ridge_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualizations saved to {self.output_dir}")
        logger.info(f"Feature importance calculated using {importance_type}")


def main():
    """Main execution function."""
    args = parse_args()

    if not args.base_dir or args.base_dir == "/path/to/first_level_results":
        logger.error("Please set CONFIG_BASE_DIR or provide --base_dir argument")
        return None

    logger.info(f"Starting optimized stacked Ridge model pipeline...")
    logger.info(f"Configuration:")
    logger.info(f"  Base directory: {args.base_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Modalities: {args.modalities}")
    logger.info(f"  Models: {args.models}")
    logger.info(f"  Folds: {args.folds}")
    logger.info(f"  Hyperparameter tuning: {args.tune_hyperparams}")
    logger.info(f"  Conservative tuning: {args.conservative_tuning}")
    logger.info(f"  Very conservative tuning: {args.very_conservative_tuning}")
    logger.info(f"  Ultra conservative tuning: {args.ultra_conservative_tuning}")
    logger.info(f"  Scaling: {args.scaling}")
    logger.info(f"  Imputation: {args.imputation}")
    logger.info(f"  Parallel workers: {args.max_workers}")

    stacker = OptimizedStackedModelBuilder(args)

    try:
        logger.info("Step 1: Training stacked Ridge model with memory optimization...")
        fold_results = stacker.train_stacked_model_memory_efficient()

        if not fold_results:
            logger.error("No models trained!")
            return None

        logger.info(f"Successfully trained Ridge models for {len(fold_results)} folds")

        logger.info("Step 2: Saving results...")
        all_metrics = stacker.save_results(fold_results)

        logger.info("Step 3: Creating visualizations...")
        stacker.create_visualizations(fold_results)

        logger.info(f"\nRIDGE STACKING COMPLETE! Results saved to {stacker.output_dir}")
        stacker.log_memory_usage("Final")

        return all_metrics

    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    main()