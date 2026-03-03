#!/usr/bin/env python3
"""
First Level Model Visualization Script

This script creates comprehensive visualizations for first level model predictions across all modalities:
1. Scatterplot with regression line (Predicted vs Observed)
2. Density heatmap scatterplot with regression line
3. Spaghetti plots for longitudinal data (observed and predicted)

Modalities: alff, cortical_area, cortical_thickness, functional_connectivity, reho, subcortical_volume, total_brain_volume
Models: krr, pls, xgboost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, linregress
from scipy import stats
import statsmodels.api as sm
import os
import logging
import traceback
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Font size configuration variables
TITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 24
TICK_FONTSIZE = 24
LEGEND_FONTSIZE = 20
STATS_TEXT_FONTSIZE = 24
COLORBAR_LABEL_FONTSIZE = 12

# File paths
BASE_DATA_PATH = '/media/hcs-sci-psy-narun/Jack'
MODEL_RESULTS_PATH = '/media/hcs-sci-psy-narun/Jack/First_level_model_aligned/model_results'
NDAR_FILE = os.path.join(BASE_DATA_PATH, "ndar_subject01.csv")

# Output directory
OUTPUT_DIR = '/media/hcs-sci-psy-narun/Jack/First_level_model_aligned/first_level_visualizations/v4_just_density'

# Modality and model configurations
MODALITIES = ['alff', 'cortical_area', 'cortical_thickness', 'functional_connectivity',
              'reho', 'subcortical_volume', 'total_brain_volume']
MODELS = ['krr', 'pls', 'xgboost']
FOLDS = [f'fold_{i}' for i in range(5)]

# Model name mapping (CSV uses these names)
MODEL_NAME_MAPPING = {
    'krr': 'krr',
    'pls': 'pls',
    'xgboost': 'xgb'
}

# Phenotype definitions
PHENOTYPE_LABELS = {
    1: "Control",
    2: "Subthreshold",
    3: "ADHD",
    4: "Other",
    5: "Not Assessed"
}

PHENOTYPE_COLORS = {
    1: 'blue',  # Control
    2: 'green',  # Subthreshold
    3: 'red',  # ADHD
    4: 'purple',  # Other
    5: 'gray'  # Not Assessed
}


def extract_ndar_and_age(subject_id):
    """Extract NDAR identifier and age from subject_id."""
    try:
        # Split on '_Age' to separate NDAR ID and age
        parts = subject_id.split('_Age')
        if len(parts) != 2:
            logger.warning(f"Unexpected subject_id format: {subject_id}")
            return None, None

        ndar_id = parts[0]
        age_months = int(parts[1])
        age_years = age_months / 12.0

        return ndar_id, age_years

    except Exception as e:
        logger.warning(f"Error parsing subject_id {subject_id}: {e}")
        return None, None


def get_csv_path(modality, model, fold):
    """Get the path to the prediction CSV file for a given modality/model/fold combination."""
    if modality == 'functional_connectivity':
        # Special structure: functional_connectivity/model/fold/fold/
        return os.path.join(MODEL_RESULTS_PATH, modality, model, fold, fold,
                            f"{modality}_{model}_predictions.csv")
    else:
        # Regular structure: modality/fold/
        return os.path.join(MODEL_RESULTS_PATH, modality, fold,
                            f"{modality}_{model}_predictions.csv")


def load_predictions_for_modality_model(modality, model):
    """Load and combine all fold predictions for a specific modality/model combination."""
    logger.info(f"Loading predictions for {modality} - {model}")

    all_predictions = []
    csv_model_name = MODEL_NAME_MAPPING[model]

    for fold in FOLDS:
        csv_path = get_csv_path(modality, model, fold)

        if not os.path.exists(csv_path):
            logger.warning(f"CSV file not found: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)

            # Verify expected columns
            expected_cols = ['subject_id', 'true_g_factor', f'{csv_model_name}_predicted']
            if not all(col in df.columns for col in expected_cols):
                logger.warning(f"Unexpected columns in {csv_path}: {df.columns.tolist()}")
                continue

            # Rename prediction column to standardized name
            df = df.rename(columns={f'{csv_model_name}_predicted': 'predicted'})
            df['fold'] = fold
            df['modality'] = modality
            df['model'] = model

            all_predictions.append(df)
            logger.info(f"Loaded {len(df)} predictions from {fold}")

        except Exception as e:
            logger.error(f"Error loading {csv_path}: {e}")
            continue

    if all_predictions:
        combined_df = pd.concat(all_predictions, ignore_index=True)
        logger.info(f"Total {modality} {model} predictions: {len(combined_df)}")
        return combined_df
    else:
        logger.warning(f"No predictions loaded for {modality} - {model}")
        return None


def prepare_longitudinal_data(predictions_df):
    """Prepare data for longitudinal analysis by extracting NDAR IDs and ages."""
    logger.info("Preparing longitudinal data...")

    # Create copy
    long_df = predictions_df.copy()

    # Extract NDAR ID and age
    ndar_ages = [extract_ndar_and_age(sid) for sid in long_df['subject_id']]
    long_df['ndar_id'] = [x[0] for x in ndar_ages]
    long_df['age_years'] = [x[1] for x in ndar_ages]

    # Remove rows where extraction failed
    long_df = long_df.dropna(subset=['ndar_id', 'age_years'])

    logger.info(f"Successfully extracted NDAR IDs for {len(long_df)} records")
    logger.info(f"Found {long_df['ndar_id'].nunique()} unique subjects")

    return long_df


def load_ndar_phenotypes(ndar_file):
    """Load and process NDAR phenotype data."""
    logger.info(f"Loading phenotype data from {ndar_file}...")

    try:
        ndar_df = pd.read_csv(ndar_file)
        logger.info(f"Loaded NDAR file with {len(ndar_df)} subjects")

        # Create mapping from NDAR ID to phenotype
        phenotype_map = {}

        for _, row in ndar_df.iterrows():
            # Extract subject key
            if 'subjectkey' in row and not pd.isna(row['subjectkey']):
                subject_key = str(row['subjectkey'])
            elif 'ndar_subject01_id' in row and not pd.isna(row['ndar_subject01_id']):
                subject_key = str(row['ndar_subject01_id'])
            else:
                continue

            # Remove underscore and standardize
            subject_key = subject_key.replace("_", "")
            if not subject_key.upper().startswith("NDAR"):
                subject_key = "NDAR" + subject_key

            # Get phenotype
            phenotype = None
            if 'phenotype' in row and not pd.isna(row['phenotype']):
                try:
                    phenotype = int(float(row['phenotype']))
                except (ValueError, TypeError):
                    pass

            if phenotype is None and 'sibling_study' in row and not pd.isna(row['sibling_study']):
                try:
                    phenotype = int(float(row['sibling_study']))
                except (ValueError, TypeError):
                    pass

            if phenotype is not None:
                phenotype_map[subject_key] = phenotype

        logger.info(f"Found phenotype information for {len(phenotype_map)} subjects")
        return phenotype_map

    except Exception as e:
        logger.error(f"Error loading NDAR phenotypes: {e}")
        return {}


def assign_phenotypes_to_longitudinal_data(long_df, phenotype_map):
    """Assign phenotypes to longitudinal data with special handling for phenotype 5."""
    logger.info("Assigning phenotypes to longitudinal data...")

    long_df['phenotype'] = None

    # First pass: direct assignment
    for idx, row in long_df.iterrows():
        ndar_id = row['ndar_id']
        if ndar_id in phenotype_map:
            long_df.at[idx, 'phenotype'] = phenotype_map[ndar_id]

    # Second pass: handle phenotype 5 (not assessed)
    for ndar_id in long_df['ndar_id'].unique():
        subject_data = long_df[long_df['ndar_id'] == ndar_id]
        phenotypes = subject_data['phenotype'].dropna().unique()

        if len(phenotypes) > 1:
            # If subject has multiple phenotypes, use the non-5 one
            non_five_phenotypes = [p for p in phenotypes if p != 5]
            if non_five_phenotypes:
                assigned_phenotype = non_five_phenotypes[0]
                long_df.loc[long_df['ndar_id'] == ndar_id, 'phenotype'] = assigned_phenotype
                logger.info(f"Subject {ndar_id}: assigned phenotype {assigned_phenotype} (had {phenotypes})")
        elif len(phenotypes) == 1 and phenotypes[0] == 5:
            # If all timepoints are 5, assign to "other" (4)
            long_df.loc[long_df['ndar_id'] == ndar_id, 'phenotype'] = 4
            logger.info(f"Subject {ndar_id}: all timepoints were 5, assigned to other (4)")

    # Convert to int where possible
    long_df['phenotype'] = pd.to_numeric(long_df['phenotype'], errors='ignore')

    return long_df


def create_basic_scatterplot(predictions_df, modality, model, output_dir):
    """Create basic scatterplot with Predicted on x-axis, Observed on y-axis."""
    logger.info(f"Creating basic scatterplot for {modality} - {model}")

    plt.figure(figsize=(10, 8))

    # Extract data
    x = predictions_df['predicted']
    y = predictions_df['true_g_factor']

    # Create scatter plot
    plt.scatter(x, y, alpha=0.6, s=30, color='steelblue', edgecolors='black', linewidth=0.5)

    # Fit regression line
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line_x = np.array([x.min(), x.max()])
    line_y = slope * line_x + intercept

    plt.plot(line_x, line_y, 'red', linewidth=2,
             label=f'r={r_value:.3f}, p<.0001')

    # Set axis limits
    plt.xlim([-1.25, 1.25])
    plt.ylim([-3, 3])

    # Formatting
    plt.xlabel('Predicted G-Factor', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('Observed G-Factor', fontsize=AXIS_LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.title(f'{modality.title()} {model.upper()}: Predicted vs Observed G-Factor', fontsize=TITLE_FONTSIZE)
    plt.legend(loc='lower right', fontsize=LEGEND_FONTSIZE)
    plt.grid(True, alpha=0.3)

    # Add statistics text
    r2 = r_value ** 2
    plt.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {np.sqrt(np.mean((y - x) ** 2)):.3f}',
             transform=plt.gca().transAxes, fontsize=STATS_TEXT_FONTSIZE, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    filename = f'{modality}_{model}_scatterplot_predicted_vs_observed.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Basic scatterplot saved: {filename}")


def create_density_scatterplot(predictions_df, modality, model, output_dir):
    """Create density heatmap scatterplot."""
    logger.info(f"Creating density heatmap scatterplot for {modality} - {model}")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract data
    x = predictions_df['predicted'].values
    y = predictions_df['true_g_factor'].values

    # Remove any NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask]
    y = y[valid_mask]

    # Calculate density
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)

    # Sort points by density so that densest points are plotted last
    idx = density.argsort()
    x, y, density = x[idx], y[idx], density[idx]

    # Create scatter plot with density coloring
    scatter = ax.scatter(x, y, c=density, s=30, alpha=0.6, cmap='viridis',
                         edgecolors='black', linewidth=0.3)

    # Add colorbar
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.set_label('Density', fontsize=COLORBAR_LABEL_FONTSIZE)

    # Fit regression line
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line_x = np.array([x.min(), x.max()])
    line_y = slope * line_x + intercept

    ax.plot(line_x, line_y, 'red', linewidth=2,
            label=f'(r={r_value:.3f})')

    # Set axis limits
    ax.set_xlim([-2, 2])
    ax.set_ylim([-3, 3])

    # Formatting
    ax.set_xlabel('Predicted G-Factor', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Observed G-Factor', fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.set_title(f'{modality.title()} {model.upper()}: Density heatmap',
                 fontsize=16)
    ax.legend(loc='lower right', fontsize=40, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = f'{modality}_{model}_density_scatterplot_predicted_vs_observed.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Density scatterplot saved: {filename}")


def create_spaghetti_plot(long_df, y_column, modality, model, output_dir, phenotype_map=None):
    """Create spaghetti plot for longitudinal data."""
    plot_type = 'observed' if y_column == 'true_g_factor' else 'predicted'
    logger.info(f"Creating spaghetti plot for {modality} - {model} ({plot_type})")

    plt.figure(figsize=(12, 8))

    # Assign phenotypes if phenotype_map is provided
    if phenotype_map:
        long_df = assign_phenotypes_to_longitudinal_data(long_df, phenotype_map)

        # Plot by phenotype
        for phenotype in sorted(long_df['phenotype'].dropna().unique()):
            if phenotype not in PHENOTYPE_LABELS:
                continue

            group_data = long_df[long_df['phenotype'] == phenotype]
            color = PHENOTYPE_COLORS[phenotype]
            label = PHENOTYPE_LABELS[phenotype]

            # Plot individual subject trajectories
            plotted_subjects = set()
            for subject in group_data['ndar_id'].unique():
                subject_data = group_data[group_data['ndar_id'] == subject].sort_values('age_years')

                if len(subject_data) > 1:
                    plt.plot(subject_data['age_years'], subject_data[y_column],
                             color=color, alpha=0.3, linewidth=0.8)

                # Only add label once per phenotype
                label_text = label if subject not in plotted_subjects and len(plotted_subjects) == 0 else ""
                plt.scatter(subject_data['age_years'], subject_data[y_column],
                            color=color, alpha=0.6, s=25, label=label_text)
                plotted_subjects.add(subject)

        # Add trend lines for Control (1) and ADHD (3)
        for phenotype in [1, 3]:  # Control and ADHD only
            if phenotype not in long_df['phenotype'].values:
                continue

            group_data = long_df[long_df['phenotype'] == phenotype]
            if len(group_data) < 10:
                continue

            try:
                X = sm.add_constant(group_data['age_years'])
                model_fit = sm.OLS(group_data[y_column], X).fit()

                age_range = np.linspace(group_data['age_years'].min(),
                                        group_data['age_years'].max(), 100)
                X_pred = sm.add_constant(age_range)
                predictions = model_fit.predict(X_pred)

                plt.plot(age_range, predictions, color=PHENOTYPE_COLORS[phenotype],
                         linewidth=4, linestyle='--',
                         label=f'{PHENOTYPE_LABELS[phenotype]} Trend (R²={model_fit.rsquared:.3f})')

            except Exception as e:
                logger.warning(f"Could not fit trend line for {PHENOTYPE_LABELS[phenotype]}: {e}")

    else:
        # Plot without phenotype stratification
        unique_subjects = long_df['ndar_id'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_subjects)))

        # Plot each subject's trajectory
        for i, subject in enumerate(unique_subjects):
            subject_data = long_df[long_df['ndar_id'] == subject].sort_values('age_years')

            if len(subject_data) > 1:
                plt.plot(subject_data['age_years'], subject_data[y_column],
                         color=colors[i % len(colors)], alpha=0.3, linewidth=0.8)

            plt.scatter(subject_data['age_years'], subject_data[y_column],
                        color=colors[i % len(colors)], alpha=0.6, s=20)

        # Add overall trend line
        if len(long_df) > 10:
            try:
                X = sm.add_constant(long_df['age_years'])
                model_fit = sm.OLS(long_df[y_column], X).fit()

                age_range = np.linspace(long_df['age_years'].min(), long_df['age_years'].max(), 100)
                X_pred = sm.add_constant(age_range)
                predictions = model_fit.predict(X_pred)

                plt.plot(age_range, predictions, 'red', linewidth=2,
                         label=f'Overall Trend (R²={model_fit.rsquared:.3f})')
            except Exception as e:
                logger.warning(f"Could not fit trend line: {e}")

    # Set appropriate y-axis label
    if y_column == 'true_g_factor':
        y_label = 'Observed G-Factor (ground truth)'
    elif y_column == 'predicted':
        y_label = f'Predicted G-Factor ({model.upper()} model)'
    else:
        y_label = y_column.replace('_', ' ').title()

    # Formatting
    plt.xlabel('Age (years)', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)

    title_suffix = 'by Phenotype' if phenotype_map else ''
    plt.title(f'{modality.title()} {model.upper()}: Longitudinal {plot_type.title()} G-Factor by Age {title_suffix}',
              fontsize=TITLE_FONTSIZE)
    plt.grid(True, alpha=0.3)

    # Create legend if we have labels - only show trend lines
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        # Remove duplicate labels and keep only trend lines
        by_label = dict(zip(labels, handles))
        # Filter to only include labels that contain "Trend"
        trend_labels = {k: v for k, v in by_label.items() if "Trend" in k}
        if trend_labels:
            plt.legend(trend_labels.values(), trend_labels.keys(), fontsize=LEGEND_FONTSIZE, loc='lower right')

    plt.tight_layout()

    phenotype_suffix = '_by_phenotype' if phenotype_map else ''
    filename = f'{modality}_{model}_spaghetti_{plot_type}_g_factor{phenotype_suffix}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Spaghetti plot saved: {filename}")


def main():
    """Main function to create all visualizations."""
    try:
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"Output directory: {OUTPUT_DIR}")

        # Load phenotype data
        phenotype_map = load_ndar_phenotypes(NDAR_FILE)

        # Process each modality/model combination
        for modality in MODALITIES:
            for model in MODELS:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Processing {modality} - {model}")
                logger.info(f"{'=' * 60}")

                # Load predictions for this modality/model combination
                predictions_df = load_predictions_for_modality_model(modality, model)

                if predictions_df is None or len(predictions_df) == 0:
                    logger.warning(f"No data available for {modality} - {model}, skipping...")
                    continue

                # Create modality-specific output directory
                modality_output_dir = os.path.join(OUTPUT_DIR, modality)
                os.makedirs(modality_output_dir, exist_ok=True)

                # Create basic scatterplot
                # create_basic_scatterplot(predictions_df, modality, model, modality_output_dir)

                # Create density scatterplot
                create_density_scatterplot(predictions_df, modality, model, modality_output_dir)

                # Prepare longitudinal data
                long_df = prepare_longitudinal_data(predictions_df)

                #if len(long_df) > 0:
                    # Create spaghetti plots for observed g-factor
                #    create_spaghetti_plot(long_df, 'true_g_factor', modality, model,
                #                          modality_output_dir, phenotype_map if phenotype_map else None)

                    # Create spaghetti plots for predicted g-factor
                #    create_spaghetti_plot(long_df, 'predicted', modality, model,
                #                          modality_output_dir, phenotype_map if phenotype_map else None)

                # Print summary statistics
                print(f"\n{modality} - {model} Summary:")
                print(f"  Total predictions: {len(predictions_df)}")
                if len(long_df) > 0:
                    print(f"  Unique subjects: {long_df['ndar_id'].nunique()}")
                correlation = stats.pearsonr(predictions_df['true_g_factor'], predictions_df['predicted'])[0]
                print(f"  Correlation: {correlation:.3f}")
                print(f"  R²: {correlation ** 2:.3f}")

        logger.info(f"\n{'=' * 60}")
        logger.info("All visualizations completed successfully!")
        logger.info(f"Plots saved to: {OUTPUT_DIR}")
        logger.info(f"{'=' * 60}")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()