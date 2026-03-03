#!/usr/bin/env python3
"""
Bootstrap Analysis: Stacked Model vs Single Model-Modality Predictors

This script compares the predictive performance of stacked machine learning models
against individual model-modality combinations using bootstrap resampling and
Fisher's z-transformation of correlation coefficients.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("bootstrap_comparison")

###########################################
#          CONFIGURATION SECTION          #
###########################################

# Bootstrap parameters
N_BOOTSTRAP = 5000
CONFIDENCE_LEVEL = 0.95

# File paths
BASE_DIR = "/media/hcs-sci-psy-narun/Jack/First_level_model_aligned"
STACKED_PREDICTIONS_FILE = "/media/hcs-sci-psy-narun/Jack/First_level_model_aligned/Stacked_model_ridge_v4_EXTREME_conserve_a2000_keeper/stacked_ridge_predictions.csv"

# Default modalities and models
DEFAULT_MODALITIES = [
    "cortical_area",
    "cortical_thickness",
    "subcortical_volume",
    "total_brain_volume",
    "alff",
    "reho",
    "functional_connectivity"
]

DEFAULT_MODELS = ["krr", "xgboost", "pls"]
DEFAULT_FOLDS = [0, 1, 2, 3, 4]


###########################################
#           HELPER FUNCTIONS              #
###########################################

def fishers_z_transform(r):
    """Apply Fisher's z-transformation to correlation coefficient."""
    # Handle edge cases
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))


def inverse_fishers_z(z):
    """Inverse Fisher's z-transformation."""
    return (np.exp(2 * z) - 1) / (np.exp(2 + z) + 1)


def find_prediction_file(base_dir, modality, model, fold_idx):
    """Find prediction file using flexible directory patterns."""
    # Handle model name variations
    model_name = model
    if model == "xgboost":
        model_name = "xgboost"  # Keep as is

    filename = f"{modality}_{model_name}_predictions.csv"

    if modality == 'functional_connectivity':
        patterns = [
            Path(base_dir) / 'model_results' / modality / model / f'fold_{fold_idx}' / f'fold_{fold_idx}' / filename,
            Path(base_dir) / 'model_results' / modality / model / f'fold_{fold_idx}' / filename,
        ]
    else:
        patterns = [
            Path(base_dir) / 'model_results' / modality / f'fold_{fold_idx}' / filename,
        ]

    for pattern in patterns:
        if pattern.exists():
            return str(pattern)

    # Try recursive search
    for pattern_str in [
        f"**/model_results/{modality}/**/fold_{fold_idx}**/{filename}",
        f"**/model_results/{modality}/**/{filename}",
    ]:
        matches = list(Path(base_dir).rglob(pattern_str))
        fold_matches = [m for m in matches if f'fold_{fold_idx}' in str(m)]
        if fold_matches:
            return str(fold_matches[0])

    return None


def standardize_prediction_columns(df, modality, model):
    """Standardize column names for prediction files."""
    try:
        # Expected columns: subject_id, true_g_factor, [algorithm]_predicted
        subject_col = 'subject_id'
        true_g_col = 'true_g_factor'

        # Handle model name variations for prediction column
        possible_pred_cols = [
            f'{model}_predicted',  # e.g., 'krr_predicted'
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
            logger.warning(f"No prediction column found for {modality}_{model}")
            logger.warning(f"Available columns: {list(df.columns)}")
            return None

        # Check required columns
        if not all(col in df.columns for col in [subject_col, true_g_col]):
            logger.warning(f"Missing subject_id or true_g_factor in {modality}_{model}")
            logger.warning(f"Available columns: {list(df.columns)}")
            return None

        standardized_df = pd.DataFrame({
            'subject_id': df[subject_col],
            'true_g_factor': df[true_g_col],
            f'{modality}_{model}': df[pred_col],
        })

        return standardized_df

    except Exception as e:
        logger.error(f"Error standardizing columns for {modality}_{model}: {e}")
        return None


def load_single_model_predictions(base_dir, modalities, models, folds):
    """Load all single model-modality prediction files."""
    all_predictions = []

    for modality in modalities:
        for model in models:
            fold_predictions = []

            for fold_idx in folds:
                pred_file = find_prediction_file(base_dir, modality, model, fold_idx)
                if pred_file is None:
                    logger.warning(f"Could not find predictions for {modality}-{model}-fold{fold_idx}")
                    continue

                try:
                    df = pd.read_csv(pred_file)
                    standardized_df = standardize_prediction_columns(df, modality, model)

                    if standardized_df is not None:
                        standardized_df['fold'] = fold_idx
                        fold_predictions.append(standardized_df)
                        logger.debug(f"✓ Loaded {modality}-{model}-fold{fold_idx}")

                except Exception as e:
                    logger.error(f"Error loading {pred_file}: {e}")
                    continue

            if fold_predictions:
                combined_model = pd.concat(fold_predictions, ignore_index=True)
                all_predictions.append(combined_model)
                logger.info(f"✓ Combined {modality}-{model}: {len(combined_model)} predictions")

    return all_predictions


def align_predictions(stacked_df, single_predictions_list):
    """Align stacked predictions with single model predictions by subject_id."""
    # Start with stacked predictions (test set only)
    aligned_df = stacked_df[stacked_df['split'] == 'test'][['subject_id', 'true_g', 'stacked_pred']].copy()

    logger.info(f"Starting with {len(aligned_df)} stacked predictions")

    # Add each single model prediction
    for single_df in single_predictions_list:
        # Get the model-modality name from columns
        pred_col = [col for col in single_df.columns if col not in ['subject_id', 'true_g_factor', 'fold']][0]

        # Merge on subject_id
        merge_df = single_df[['subject_id', pred_col]].copy()
        aligned_df = aligned_df.merge(merge_df, on='subject_id', how='inner')

        logger.info(f"After merging {pred_col}: {len(aligned_df)} predictions")

    return aligned_df


def calculate_correlations(df, true_g_col='true_g'):
    """Calculate correlations between true_g and all prediction columns."""
    pred_cols = [col for col in df.columns if col not in ['subject_id', 'true_g', 'fold']]

    correlations = {}
    for col in pred_cols:
        r, p = pearsonr(df[true_g_col], df[col])
        correlations[col] = {'r': r, 'p_value': p}

    return correlations


def bootstrap_correlation_differences(df, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap correlation differences between stacked and single models."""
    true_g = df['true_g'].values
    stacked_pred = df['stacked_pred'].values

    # Get all single model prediction columns
    single_cols = [col for col in df.columns if col not in ['subject_id', 'true_g', 'stacked_pred', 'fold']]

    results = {}

    for single_col in tqdm(single_cols, desc="Bootstrapping model comparisons"):
        single_pred = df[single_col].values

        bootstrap_diffs = []

        for _ in range(n_bootstrap):
            # Bootstrap resample
            n_samples = len(df)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)

            # Calculate correlations for bootstrap sample
            boot_true_g = true_g[indices]
            boot_stacked = stacked_pred[indices]
            boot_single = single_pred[indices]

            # Calculate correlations
            r_stacked = np.corrcoef(boot_true_g, boot_stacked)[0, 1]
            r_single = np.corrcoef(boot_true_g, boot_single)[0, 1]

            # Fisher's z-transformation
            z_stacked = fishers_z_transform(r_stacked)
            z_single = fishers_z_transform(r_single)

            # Calculate difference
            diff = z_stacked - z_single
            bootstrap_diffs.append(diff)

        bootstrap_diffs = np.array(bootstrap_diffs)

        # Calculate confidence intervals
        alpha = 1 - CONFIDENCE_LEVEL
        ci_lower = np.percentile(bootstrap_diffs, (alpha / 2) * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)

        # Calculate original difference
        orig_r_stacked = np.corrcoef(true_g, stacked_pred)[0, 1]
        orig_r_single = np.corrcoef(true_g, single_pred)[0, 1]
        orig_z_diff = fishers_z_transform(orig_r_stacked) - fishers_z_transform(orig_r_single)

        results[single_col] = {
            'original_r_stacked': orig_r_stacked,
            'original_r_single': orig_r_single,
            'r_difference': orig_r_stacked - orig_r_single,  # Simple correlation difference
            'original_z_diff': orig_z_diff,  # This is our effect size
            'bootstrap_z_diffs': bootstrap_diffs,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': not (ci_lower <= 0 <= ci_upper),
            'mean_z_diff': np.mean(bootstrap_diffs),
            'std_z_diff': np.std(bootstrap_diffs)
        }

    return results


def create_seaborn_ridge_plot(results, output_dir):
    """Create a beautiful compressed ridge plot using seaborn similar to Farzane's approach."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data for ridge plot
    ridge_data = []

    # Define color gradients by modality type (darker = better performance, i.e., smaller Z difference)
    def get_color_by_performance(modality_type, z_diff, all_z_diffs_in_type):
        """Get color shade based on performance within modality type."""
        # Normalize z_diff within the modality type
        # Smaller Z difference = better performance = should get darker/more vibrant color
        min_z = min(all_z_diffs_in_type)
        max_z = max(all_z_diffs_in_type)
        if max_z == min_z:
            norm_performance = 0.5
        else:
            # Normalize so smaller z_diff gives lower value (better performance)
            norm_performance = (z_diff - min_z) / (max_z - min_z)

        # Improved color palettes with better contrast and vibrancy
        if modality_type == 'Functional':
            # Green palette - more vibrant (darkest first for best performance)
            colors = ['#1B5E20', '#2E7D32', '#388E3C', '#43A047', '#4CAF50', '#66BB6A', '#81C784']
        elif modality_type == 'Structural (Cortical)':
            # Coral/red palette - more vibrant (darkest first for best performance)
            colors = ['#B71C1C', '#C62828', '#D32F2F', '#E53935', '#F44336', '#EF5350', '#E57373']
        elif modality_type == 'Structural (Subcortical)':
            # Blue palette - more vibrant (darkest first for best performance)
            colors = ['#0D47A1', '#1565C0', '#1976D2', '#1E88E5', '#2196F3', '#42A5F5', '#64B5F6']
        else:
            colors = ['#424242', '#525252', '#616161', '#757575', '#9E9E9E', '#BDBDBD', '#E0E0E0']

        # Select color based on performance
        # norm_performance = 0 (best, smallest Z diff) gets darkest color (index 0)
        # norm_performance = 1 (worst, largest Z diff) gets lightest color (index -1)
        color_idx = int(norm_performance * (len(colors) - 1))
        return colors[color_idx]

    # First pass: collect all z_diffs by modality type to normalize colors
    z_diffs_by_type = {'Functional': [], 'Structural (Cortical)': [], 'Structural (Subcortical)': []}

    for model_name, result in results.items():
        parts = model_name.split('_')
        if len(parts) >= 2:
            modality = '_'.join(parts[:-1])
        else:
            modality = model_name

        if modality in ['alff', 'reho', 'functional_connectivity']:
            modality_type = 'Functional'
        elif modality in ['cortical_area', 'cortical_thickness']:
            modality_type = 'Structural (Cortical)'
        elif modality in ['subcortical_volume', 'total_brain_volume']:
            modality_type = 'Structural (Subcortical)'
        else:
            continue

        z_diffs_by_type[modality_type].append(result['original_z_diff'])

    # Second pass: create data with proper colors
    for model_name, result in results.items():
        parts = model_name.split('_')
        if len(parts) >= 2:
            algorithm = parts[-1]
            modality = '_'.join(parts[:-1])
        else:
            algorithm = 'unknown'
            modality = model_name

        # Determine modality type
        if modality in ['alff', 'reho', 'functional_connectivity']:
            modality_type = 'Functional'
        elif modality in ['cortical_area', 'cortical_thickness']:
            modality_type = 'Structural (Cortical)'
        elif modality in ['subcortical_volume', 'total_brain_volume']:
            modality_type = 'Structural (Subcortical)'
        else:
            modality_type = 'Other'

        # Get color based on performance within modality type
        color = get_color_by_performance(modality_type, result['original_z_diff'],
                                         z_diffs_by_type[modality_type])

        # Add each bootstrap sample to the data
        for z_diff in result['bootstrap_z_diffs']:
            ridge_data.append({
                'model_modality': model_name,
                'modality': modality,
                'algorithm': algorithm,
                'modality_type': modality_type,
                'z_difference': z_diff,
                'original_z_diff': result['original_z_diff'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'r_difference': result['r_difference'],
                'color': color
            })

    ridge_df = pd.DataFrame(ridge_data)

    # Sort by effect size (original z difference) for better visual ordering
    model_order = sorted(results.keys(), key=lambda x: results[x]['original_z_diff'], reverse=True)
    ridge_df['model_modality'] = pd.Categorical(ridge_df['model_modality'], categories=model_order, ordered=True)
    ridge_df = ridge_df.sort_values('model_modality')

    # Set consistent x-axis limits with extra white space for text positioning
    all_z_diffs = ridge_df['z_difference'].tolist()
    x_min = min(all_z_diffs) - 0.02
    x_max = max(all_z_diffs) + 0.02

    # Add substantial white space on both sides for text
    x_range = x_max - x_min
    x_min_expanded = x_min - (x_range * 0.4)  # 40% extra space on left
    x_max_expanded = x_max + (x_range * 0.3)  # 30% extra space on right
    xlim = (x_min_expanded, x_max_expanded)

    # Create the ridge plot with compressed spacing and wider figure
    plt.style.use('default')
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Create FacetGrid with very small height for compression and wider figure
    g = sns.FacetGrid(ridge_df, row="model_modality", aspect=25, height=0.4,
                      sharex=True, sharey=False)

    # Map KDE plots
    g.map_dataframe(sns.kdeplot, "z_difference", bw_adjust=0.5, clip_on=True,
                    fill=True, alpha=0.8, linewidth=1.2)
    g.map_dataframe(sns.kdeplot, "z_difference", clip_on=True, color="white",
                    lw=1.5, bw_adjust=0.5)

    # Color each distribution and add annotations
    for i, (ax, (model_name, group_data)) in enumerate(zip(g.axes.flat, ridge_df.groupby('model_modality'))):
        if len(ax.collections) > 0:
            # Get the color for this model
            fill_color = group_data['color'].iloc[0]

            # Set the fill color
            for collection in ax.collections:
                collection.set_facecolor(fill_color)
                collection.set_edgecolor('none')

            # Add vertical lines for original difference and confidence intervals
            original_diff = group_data['original_z_diff'].iloc[0]
            ci_lower = group_data['ci_lower'].iloc[0]
            ci_upper = group_data['ci_upper'].iloc[0]

            # Get the KDE line to determine height for clipping vertical lines
            if len(ax.lines) > 0:
                kdeline = ax.lines[0]
                xs = kdeline.get_xdata()
                ys = kdeline.get_ydata()
                max_height = max(ys) * 0.92  # Clip to 92% of max height

                # Calculate heights for vertical lines at specific x positions
                height_orig = min(np.interp(original_diff, xs, ys), max_height) if len(xs) > 0 else max_height
                height_lower = min(np.interp(ci_lower, xs, ys), max_height) if len(xs) > 0 else max_height
                height_upper = min(np.interp(ci_upper, xs, ys), max_height) if len(xs) > 0 else max_height

                # Add clipped vertical lines - main difference in light gray (thinner), CIs in dark gray
                ax.vlines(original_diff, 0, height_orig, color='#BDBDBD', ls='-', linewidth=1, alpha=0.8)
                ax.vlines(ci_lower, 0, height_lower, color='#424242', ls='--', linewidth=1.2, alpha=0.8)
                ax.vlines(ci_upper, 0, height_upper, color='#424242', ls='--', linewidth=1.2, alpha=0.8)

            # Add reference line at 0 - darker and wider for better visibility
            ax.axvline(x=0, color='#616161', linestyle='-', alpha=0.9, linewidth=2)
            ax.axhline(y=0, linewidth=1, linestyle="-", color='lightgray', clip_on=False)

            # Clean up model name for display
            display_name = model_name.replace('_', ' ').title()
            r_diff = group_data['r_difference'].iloc[0]

            # Position text in the white space areas using axis coordinates
            # With expanded x-limits, we can position text in the margin areas
            # Left side: Model name and Δr in the left white space
            ax.text(0.15, 0.7, display_name,
                    ha='left', va='center', transform=ax.transAxes,
                    fontsize=8, color='black', weight='bold')
            ax.text(0.15, 0.3, f"Δr = {r_diff:.3f}",
                    ha='left', va='center', transform=ax.transAxes,
                    fontsize=7, color='black')

            # Right side: Z value and CI in the right white space
            ax.text(0.92, 0.7, f"Z = {original_diff:.3f}",
                    ha='right', va='center', transform=ax.transAxes,
                    fontsize=7, color='black', weight='bold')
            ax.text(0.92, 0.3, f"[{ci_lower:.3f}, {ci_upper:.3f}]",
                    ha='right', va='center', transform=ax.transAxes,
                    fontsize=6, color='#424242')

            # Set x-axis limits with expanded white space
            ax.set_xlim(xlim)

            # Remove y-axis ticks but keep them in backend
            ax.set_yticks([])
            ax.set_ylabel("")

            # Customize x-axis ticks
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))

    # Remove y-axis labels globally
    g.set(ylabel="")
    g.set(xlabel="Fisher's Z Difference (Stacked - Single Model)")

    # Adjust spacing for compression and add margins for text
    g.figure.subplots_adjust(hspace=-0.25, left=0.02, right=0.98, top=0.95, bottom=0.08)
    g.figure.suptitle("Bootstrap Comparison: Stacked vs Single Model Performance",
                      y=0.98, fontsize=14, weight='bold')
    g.set_titles("")  # Remove individual plot titles
    g.despine(bottom=False, left=True)

    # Add improved legend for modality types with proper positioning and styling
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1B5E20', label='Functional', edgecolor='white', linewidth=0.5),
        Patch(facecolor='#B71C1C', label='Structural (Cortical)', edgecolor='white', linewidth=0.5),
        Patch(facecolor='#0D47A1', label='Structural (Subcortical)', edgecolor='white', linewidth=0.5)
    ]

    # Position legend outside the plot area
    legend = g.figure.legend(handles=legend_elements,
                             loc='upper center',
                             bbox_to_anchor=(0.85, 0.95),
                             frameon=True,
                             fancybox=True,
                             shadow=False,
                             ncol=1,
                             fontsize=10,
                             facecolor='white',
                             edgecolor='gray',
                             framealpha=0.9)

    # Save the plot
    plt.savefig(output_dir / 'bootstrap_ridge_plot.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'bootstrap_ridge_plot.svg', format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    logger.info(f"Seaborn ridge plot saved to {output_dir}")
    return ridge_df


def create_matplotlib_visualizations(results, output_dir):
    """Create matplotlib visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary plot of differences and confidence intervals
    model_names = list(results.keys())
    z_diffs = [results[model]['original_z_diff'] for model in model_names]
    ci_lowers = [results[model]['ci_lower'] for model in model_names]
    ci_uppers = [results[model]['ci_upper'] for model in model_names]

    # Sort by z_diff magnitude for better visualization
    sorted_indices = np.argsort(z_diffs)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    z_diffs = [z_diffs[i] for i in sorted_indices]
    ci_lowers = [ci_lowers[i] for i in sorted_indices]
    ci_uppers = [ci_uppers[i] for i in sorted_indices]

    # Create summary plot
    fig, ax = plt.subplots(figsize=(12, 10))

    y_pos = np.arange(len(model_names))

    # Plot error bars with black lines and red points
    ax.errorbar(z_diffs, y_pos,
                xerr=[np.array(z_diffs) - np.array(ci_lowers),
                      np.array(ci_uppers) - np.array(z_diffs)],
                fmt='o', color='black', capsize=5, capthick=2)

    # Add red points
    ax.scatter(z_diffs, y_pos, c='red', s=100, zorder=3)

    # Add vertical line at 0
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names, fontsize=10)
    ax.set_xlabel("Fisher's Z Difference (Stacked - Single Model)", fontsize=12)
    ax.set_title(
        f"Bootstrap Comparison: Stacked vs Single Models\n({N_BOOTSTRAP} iterations, {CONFIDENCE_LEVEL * 100}% CI)",
        fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'bootstrap_comparison_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Matplotlib visualizations saved to {output_dir}")


def save_results(results, aligned_df, output_dir):
    """Save bootstrap results to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary results
    summary_data = []
    for model, result in results.items():
        summary_data.append({
            'model_modality': model,
            'original_r_stacked': result['original_r_stacked'],
            'original_r_single': result['original_r_single'],
            'r_difference': result['original_r_stacked'] - result['original_r_single'],
            'original_z_difference': result['original_z_diff'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'mean_z_difference': result['mean_z_diff'],
            'std_z_difference': result['std_z_diff'],
            'significant': result['significant'],
            'ci_excludes_zero': not (result['ci_lower'] <= 0 <= result['ci_upper'])
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('original_z_difference', ascending=False)
    summary_df.to_csv(output_dir / 'bootstrap_comparison_summary.csv', index=False)

    # Save aligned predictions
    aligned_df.to_csv(output_dir / 'aligned_predictions.csv', index=False)

    logger.info(f"Results saved to {output_dir}")


def main():
    """Main execution function."""
    logger.info("Starting bootstrap comparison analysis...")
    logger.info(f"Bootstrap iterations: {N_BOOTSTRAP}")
    logger.info(f"Confidence level: {CONFIDENCE_LEVEL}")

    # Load stacked predictions
    logger.info("Loading stacked predictions...")
    stacked_df = pd.read_csv(STACKED_PREDICTIONS_FILE)
    logger.info(f"Loaded {len(stacked_df)} stacked predictions")

    # Load single model predictions
    logger.info("Loading single model predictions...")
    single_predictions = load_single_model_predictions(
        BASE_DIR, DEFAULT_MODALITIES, DEFAULT_MODELS, DEFAULT_FOLDS
    )

    if not single_predictions:
        logger.error("No single model predictions loaded!")
        return

    logger.info(f"Loaded {len(single_predictions)} model-modality combinations")

    # Align predictions
    logger.info("Aligning predictions by subject_id...")
    aligned_df = align_predictions(stacked_df, single_predictions)

    # Check for missing data
    logger.info(f"Final aligned dataset: {len(aligned_df)} predictions")
    logger.info(
        f"Number of prediction columns: {len([col for col in aligned_df.columns if col not in ['subject_id', 'true_g', 'fold']])}")

    # Check for any missing values
    missing_data = aligned_df.isnull().sum()
    if missing_data.any():
        logger.warning("Missing data detected:")
        for col, count in missing_data.items():
            if count > 0:
                logger.warning(f"  {col}: {count} missing values")

        # Use complete cases only
        aligned_df = aligned_df.dropna()
        logger.info(f"After removing incomplete cases: {len(aligned_df)} predictions")
    else:
        logger.info("No missing data detected")

    # Calculate original correlations
    logger.info("Calculating original correlations...")
    original_corrs = calculate_correlations(aligned_df)

    logger.info("Original correlations:")
    for model, corr_info in original_corrs.items():
        logger.info(f"  {model}: r = {corr_info['r']:.4f} (p = {corr_info['p_value']:.4f})")

    # Bootstrap analysis
    logger.info("Starting bootstrap analysis...")
    bootstrap_results = bootstrap_correlation_differences(aligned_df, N_BOOTSTRAP)

    # Create output directory
    output_dir = Path("stack_single_bootstrap_comparison_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    save_results(bootstrap_results, aligned_df, output_dir)

    # Create visualizations
    create_matplotlib_visualizations(bootstrap_results, output_dir)
    ridge_df = create_seaborn_ridge_plot(bootstrap_results, output_dir)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("BOOTSTRAP COMPARISON RESULTS")
    logger.info("=" * 80)

    significant_improvements = 0
    for model, result in bootstrap_results.items():
        significance = "***" if result['significant'] else ""
        logger.info(f"{model}:")
        logger.info(
            f"  Original correlations: Stacked={result['original_r_stacked']:.4f}, Single={result['original_r_single']:.4f}")
        logger.info(f"  Correlation improvement: {result['r_difference']:.4f}")
        logger.info(
            f"  Fisher's Z difference (effect size): {result['original_z_diff']:.4f} [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}] {significance}")
        logger.info(f"  Significant improvement: {result['significant']}")

        if result['significant']:
            significant_improvements += 1
        logger.info("")

    logger.info(
        f"Stacked model significantly outperformed {significant_improvements}/{len(bootstrap_results)} single models")
    logger.info(f"Results saved to {output_dir}")
    logger.info(
        f"Main visualizations: bootstrap_ridge_plot.png/.svg (seaborn), bootstrap_comparison_summary.png (matplotlib)")


if __name__ == "__main__":
    main()