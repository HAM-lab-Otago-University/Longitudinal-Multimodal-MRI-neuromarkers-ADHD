#!/usr/bin/env python3
"""
Stacked Ridge Model Visualization Script

This script creates comprehensive visualizations for the stacked ridge model predictions:
1. Scatterplot with regression line (Predicted vs Observed)
2. Density heatmap scatterplot with regression line
3. Spaghetti plots for longitudinal data (observed and predicted)
4. Phenotype-stratified spaghetti plots with regression lines
5. Combined observed vs predicted spaghetti plot

Modified to match ADHD-RS style: only subjects with multiple timepoints, no scatter points,
trend lines only for Control and ADHD.
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

# File paths
BASE_DATA_PATH = '/media/hcs-sci-psy-narun/Jack'
STACKED_PREDICTIONS_FILE = '/media/hcs-sci-psy-narun/Jack/First_level_model_aligned/Stacked_model_ridge_v4_EXTREME_conserve_a2000_keeper/stacked_ridge_predictions.csv'
NDAR_FILE = os.path.join(BASE_DATA_PATH, "ndar_subject01.csv")

# Output directory
OUTPUT_DIR = '/media/hcs-sci-psy-narun/Jack/First_level_model_aligned/Stacked_model_ridge_v4_EXTREME_conserve_a2000_keeper/Viz_figures_v9_adhdrs_style'

# Phenotype definitions - Updated to include group 5
PHENOTYPE_LABELS = {
    1: "Control",
    2: "Subthreshold",
    3: "ADHD",
    4: "Other",
    5: "Not Assessed"  # Added phenotype group 5
}

PHENOTYPE_COLORS = {
    1: 'blue',  # Control
    2: 'purple',  # Subthreshold
    3: 'red',  # ADHD
    4: 'green',  # Other
    5: 'gray'  # Not Assessed
}

# Light colors for individual trajectories
PHENOTYPE_TRAJECTORY_COLORS = {
    1: 'blue',  # Control
    2: 'purple',  # Subthreshold
    3: 'red',  # ADHD
    4: 'green',  # Other
    5: 'gray'  # Not Assessed
}


def load_data():
    """Load the stacked predictions and NDAR data."""
    logger.info("Loading stacked predictions data...")

    try:
        # Load stacked predictions
        stacked_df = pd.read_csv(STACKED_PREDICTIONS_FILE)
        logger.info(f"Loaded {len(stacked_df)} prediction records")

        # Load NDAR phenotype data
        ndar_df = pd.read_csv(NDAR_FILE)
        logger.info(f"Loaded NDAR data with {len(ndar_df)} records")

        return stacked_df, ndar_df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


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


def prepare_longitudinal_data(stacked_df):
    """Prepare data for longitudinal analysis by extracting NDAR IDs and ages."""
    logger.info("Preparing longitudinal data...")

    # Create copies to avoid modifying original
    long_df = stacked_df.copy()

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
    """Load and process NDAR phenotype data (adapted from original script)."""
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
            # If all timepoints are 5, keep as 5 (don't reassign to 4)
            logger.info(f"Subject {ndar_id}: keeping as phenotype 5 (Not Assessed)")

    # Convert to int where possible
    long_df['phenotype'] = pd.to_numeric(long_df['phenotype'], errors='ignore')

    # Log phenotype distribution
    phenotype_counts = long_df['phenotype'].value_counts().sort_index()
    for phenotype, count in phenotype_counts.items():
        if phenotype in PHENOTYPE_LABELS:
            logger.info(f"  {PHENOTYPE_LABELS[phenotype]}: {count} records")

    return long_df


def create_basic_scatterplot(stacked_df, output_dir):
    """Create basic scatterplot with Predicted on x-axis, Observed on y-axis."""
    logger.info("Creating basic scatterplot...")

    plt.figure(figsize=(10, 8))

    # Extract data
    x = stacked_df['stacked_pred']
    y = stacked_df['true_g']

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
    plt.xlabel('Predicted G-Factor', fontsize=24)
    plt.ylabel('Observed G-Factor', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title('Stacked Ridge Model: Predicted vs Observed G-Factor', fontsize=18)
    plt.legend(loc='lower right', fontsize=24)
    plt.grid(True, alpha=0.3)

    # Add statistics text
    r2 = r_value ** 2
    plt.text(0.05, 0.95, f'RÂ² = {r2:.3f}\nRMSE = {np.sqrt(np.mean((y - x) ** 2)):.3f}',
             transform=plt.gca().transAxes, fontsize=24, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatterplot_predicted_vs_observed.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Basic scatterplot saved")


def create_density_scatterplot(stacked_df, output_dir):
    """Create density heatmap scatterplot."""
    logger.info("Creating density heatmap scatterplot...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract data
    x = stacked_df['stacked_pred'].values
    y = stacked_df['true_g'].values

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
    scatter = ax.scatter(x, y, c=density, s=30, alpha=0.6, cmap='viridis', edgecolors='black', linewidth=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Density', fontsize=12)

    # Fit regression line
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line_x = np.array([x.min(), x.max()])
    line_y = slope * line_x + intercept

    ax.plot(line_x, line_y, 'red', linewidth=2,
            label=f'Regression Line (r={r_value:.3f})')

    # Add identity line
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'gray', linestyle='--',
            alpha=0.8, linewidth=2, label='Identity Line')

    # Set axis limits
    ax.set_xlim([-2, 2])
    ax.set_ylim([-3, 3])

    # Formatting
    ax.set_xlabel('Predicted G-Factor', fontsize=14)
    ax.set_ylabel('Observed G-Factor', fontsize=14)
    ax.set_title('Density Heatmap: Predicted vs Observed G-Factor', fontsize=16)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'density_scatterplot_predicted_vs_observed.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Density scatterplot saved")


def create_spaghetti_plot(long_df, y_column, title, filename, output_dir):
    """Create spaghetti plot for longitudinal data."""
    logger.info(f"Creating spaghetti plot: {title}")

    plt.figure(figsize=(12, 8))

    # Generate colors for subjects
    unique_subjects = long_df['ndar_id'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_subjects)))

    # Plot each subject's trajectory
    for i, subject in enumerate(unique_subjects):
        subject_data = long_df[long_df['ndar_id'] == subject].sort_values('age_years')

        if len(subject_data) > 1:
            # Draw line connecting timepoints - increased thickness
            plt.plot(subject_data['age_years'], subject_data[y_column],
                     color=colors[i % len(colors)], alpha=0.3, linewidth=1.5)

        # Plot individual points
        plt.scatter(subject_data['age_years'], subject_data[y_column],
                    color=colors[i % len(colors)], alpha=0.6, s=20)

    # Add overall trend line
    trend_label = None
    if len(long_df) > 10:
        try:
            X = sm.add_constant(long_df['age_years'])
            model = sm.OLS(long_df[y_column], X).fit()

            age_range = np.linspace(long_df['age_years'].min(), long_df['age_years'].max(), 100)
            X_pred = sm.add_constant(age_range)
            predictions = model.predict(X_pred)

            plt.plot(age_range, predictions, 'red', linewidth=2,
                     label=f'Overall Trend (R²={model.rsquared:.3f})')
            trend_label = f'Overall Trend (R²={model.rsquared:.3f})'
        except Exception as e:
            logger.warning(f"Could not fit trend line: {e}")

    # Set appropriate y-axis label
    if y_column == 'true_g':
        y_label = 'Observed cognition (ground truth)'
    elif y_column == 'stacked_pred':
        y_label = 'Predicted cognition (stacked model)'
    else:
        y_label = y_column.replace('_', ' ').title()

    # Formatting
    plt.xlabel('Age (years)', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    # Add legend if trend line was fitted
    if trend_label:
        plt.legend(loc='lower right', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Spaghetti plot saved: {filename}")


def create_phenotype_spaghetti_plot(long_df, y_column, title, filename, output_dir):
    """
    Create spaghetti plot stratified by phenotype with trend lines for Control and ADHD only.
    Modified to show only subjects with multiple timepoints, no scatter points.
    """
    logger.info(f"Creating phenotype spaghetti plot: {title}")

    # Filter to only subjects with multiple timepoints
    subjects_with_multiple = long_df.groupby('ndar_id').size()
    subjects_multiple_timepoints = subjects_with_multiple[subjects_with_multiple > 1].index
    long_df_filtered = long_df[long_df['ndar_id'].isin(subjects_multiple_timepoints)].copy()

    logger.info(
        f"Filtered from {long_df['ndar_id'].nunique()} to {long_df_filtered['ndar_id'].nunique()} subjects with multiple timepoints")

    if long_df_filtered.empty:
        logger.error("No subjects with multiple timepoints found")
        return

    plt.figure(figsize=(16, 10))

    # Filter for subjects with phenotype data
    phenotype_df = long_df_filtered.dropna(subset=['phenotype'])

    # Plot individual trajectories for all phenotype groups (NO scatter points)
    for phenotype in sorted(phenotype_df['phenotype'].unique()):
        if phenotype not in PHENOTYPE_LABELS:
            continue

        group_data = phenotype_df[phenotype_df['phenotype'] == phenotype]
        trajectory_color = PHENOTYPE_TRAJECTORY_COLORS[phenotype]

        # Plot individual subject trajectories (lines only, no scatter)
        for subject in group_data['ndar_id'].unique():
            subject_data = group_data[group_data['ndar_id'] == subject].sort_values('age_years')

            if len(subject_data) >= 2:  # Only subjects with multiple timepoints
                plt.plot(subject_data['age_years'], subject_data[y_column],
                         color=trajectory_color, alpha=0.4, linewidth=1.2)

    # Add trend lines ONLY for Control (1) and ADHD (3) with SOLID lines
    trend_handles = []
    for phenotype in [1, 3]:  # Control and ADHD only
        if phenotype not in phenotype_df['phenotype'].values:
            continue

        group_data = phenotype_df[phenotype_df['phenotype'] == phenotype]
        if len(group_data) < 10:
            logger.warning(f"Not enough data for {PHENOTYPE_LABELS[phenotype]} trend line")
            continue

        try:
            X = sm.add_constant(group_data['age_years'])
            model = sm.OLS(group_data[y_column], X).fit()

            age_range = np.linspace(group_data['age_years'].min(), group_data['age_years'].max(), 100)
            X_pred = sm.add_constant(age_range)
            predictions = model.predict(X_pred)

            # SOLID trend lines (not dashed)
            line = plt.plot(age_range, predictions, color=PHENOTYPE_COLORS[phenotype],
                            linewidth=4, linestyle='-',  # Changed to solid line
                            label=f'{PHENOTYPE_LABELS[phenotype]} (R²: {model.rsquared:.3f})')

            trend_handles.extend(line)

        except Exception as e:
            logger.warning(f"Could not fit trend line for {PHENOTYPE_LABELS[phenotype]}: {e}")

    # Set appropriate y-axis label
    if y_column == 'true_g':
        y_label = 'Observed G-Factor'
    elif y_column == 'stacked_pred':
        y_label = 'Predicted G-Factor'
    else:
        y_label = y_column.replace('_', ' ').title()

    # Formatting to match ADHD-RS style
    plt.xlabel('Age (years)', fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, alpha=0.3)

    # Legend with large font (only trend lines)
    if trend_handles:
        plt.legend(handles=trend_handles, fontsize=30, loc='lower right')

    # Add summary statistics text
    #control_data = phenotype_df[phenotype_df['phenotype'] == 1] if 1 in phenotype_df[
    #    'phenotype'].values else pd.DataFrame()
    #adhd_data = phenotype_df[phenotype_df['phenotype'] == 3] if 3 in phenotype_df[
    #    'phenotype'].values else pd.DataFrame()

    #summary_text = f"""
    #Group Comparison:
    #Control: {control_data['ndar_id'].nunique()} subjects, {len(control_data)} datapoints
    #ADHD: {adhd_data['ndar_id'].nunique()} subjects, {len(adhd_data)} datapoints
    #"""
    #plt.figtext(0.02, 0.02, summary_text, fontsize=10, verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Phenotype spaghetti plot saved: {filename}")


def create_combined_observed_predicted_spaghetti_plot(long_df, title, filename, output_dir):
    """Create spaghetti plot with both observed and predicted g-factor trajectories."""
    logger.info(f"Creating combined observed vs predicted spaghetti plot: {title}")

    plt.figure(figsize=(14, 10))

    # Define colors for observed vs predicted
    observed_color = 'blue'
    predicted_color = 'red'

    # Get unique subjects
    unique_subjects = long_df['ndar_id'].unique()

    # Plot trajectories for each subject - both observed and predicted
    for subject in unique_subjects:
        subject_data = long_df[long_df['ndar_id'] == subject].sort_values('age_years')

        if len(subject_data) > 1:
            # Plot observed g trajectory
            plt.plot(subject_data['age_years'], subject_data['true_g'],
                     color=observed_color, alpha=0.3, linewidth=1.5)

            # Plot predicted g trajectory
            plt.plot(subject_data['age_years'], subject_data['stacked_pred'],
                     color=predicted_color, alpha=0.3, linewidth=1.5)

        # Plot individual points (observed)
        plt.scatter(subject_data['age_years'], subject_data['true_g'],
                    color=observed_color, alpha=0.6, s=25)

        # Plot individual points (predicted)
        plt.scatter(subject_data['age_years'], subject_data['stacked_pred'],
                    color=predicted_color, alpha=0.6, s=25)

    # Add trend lines for both observed and predicted
    for data_type, color, label in [('true_g', observed_color, 'Observed G'),
                                    ('stacked_pred', predicted_color, 'Predicted G')]:
        if len(long_df) > 10:
            try:
                X = sm.add_constant(long_df['age_years'])
                model = sm.OLS(long_df[data_type], X).fit()

                age_range = np.linspace(long_df['age_years'].min(), long_df['age_years'].max(), 100)
                X_pred = sm.add_constant(age_range)
                predictions = model.predict(X_pred)

                plt.plot(age_range, predictions, color=color,
                         linewidth=4, linestyle='--',
                         label=f'{label} Trend (RÂ²={model.rsquared:.3f})')

            except Exception as e:
                logger.warning(f"Could not fit trend line for {label}: {e}")

    # Formatting
    plt.xlabel('Age (years)', fontsize=20)
    plt.ylabel('G-Factor', fontsize=20)
    plt.title(title, fontsize=24)
    plt.grid(True, alpha=0.3)

    # Set tick font size
    plt.tick_params(labelsize=18)

    # Create legend
    plt.legend(fontsize=16, loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Combined observed vs predicted spaghetti plot saved: {filename}")


def main():
    """Main function to create all visualizations."""
    try:
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"Output directory: {OUTPUT_DIR}")

        # Load data
        stacked_df, ndar_df = load_data()

        # Create basic scatterplots
        create_basic_scatterplot(stacked_df, OUTPUT_DIR)
        create_density_scatterplot(stacked_df, OUTPUT_DIR)

        # Prepare longitudinal data
        long_df = prepare_longitudinal_data(stacked_df)

        # Create basic spaghetti plots
        create_spaghetti_plot(long_df, 'true_g',
                              'Longitudinal Observed G-Factor by Age',
                              'spaghetti_observed_g_factor.png', OUTPUT_DIR)

        create_spaghetti_plot(long_df, 'stacked_pred',
                              'Longitudinal Predicted G-Factor by Age',
                              'spaghetti_predicted_g_factor.png', OUTPUT_DIR)

        # Create combined observed vs predicted spaghetti plot
        create_combined_observed_predicted_spaghetti_plot(long_df,
                                                          'Longitudinal Observed vs Predicted G-Factor',
                                                          'spaghetti_observed_vs_predicted_g_factor.png',
                                                          OUTPUT_DIR)

        # Load phenotype data and create phenotype-stratified plots
        phenotype_map = load_ndar_phenotypes(NDAR_FILE)
        if phenotype_map:
            long_df_with_phenotype = assign_phenotypes_to_longitudinal_data(long_df, phenotype_map)

            create_phenotype_spaghetti_plot(long_df_with_phenotype, 'true_g',
                                            'Longitudinal Observed G-Factor - Control vs ADHD Comparison',
                                            'spaghetti_observed_g_factor_control_vs_adhd.png', OUTPUT_DIR)

            create_phenotype_spaghetti_plot(long_df_with_phenotype, 'stacked_pred',
                                            'Longitudinal Predicted G-Factor - Control vs ADHD Comparison',
                                            'spaghetti_predicted_g_factor_control_vs_adhd.png', OUTPUT_DIR)
        else:
            logger.warning("No phenotype data available - skipping phenotype-stratified plots")

        logger.info("All visualizations completed successfully!")
        logger.info(f"Plots saved to: {OUTPUT_DIR}")

        # Print summary statistics
        print("\n" + "=" * 50)
        print("SUMMARY STATISTICS")
        print("=" * 50)
        print(f"Total predictions: {len(stacked_df)}")
        print(f"Unique subjects: {long_df['ndar_id'].nunique()}")
        print(f"Overall correlation: {stats.pearsonr(stacked_df['true_g'], stacked_df['stacked_pred'])[0]:.3f}")
        print(f"Overall RÂ²: {stats.pearsonr(stacked_df['true_g'], stacked_df['stacked_pred'])[0] ** 2:.3f}")

        if 'phenotype' in long_df.columns:
            print(f"Subjects with phenotype data: {long_df.dropna(subset=['phenotype'])['ndar_id'].nunique()}")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()