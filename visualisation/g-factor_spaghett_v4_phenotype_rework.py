#!/usr/bin/env python3
"""
Longitudinal G-Factor Visualization Script with Stable Phenotype Integration

This script creates a longitudinal line plot for g-factor scores. It now
incorporates a robust, multi-step process to assign a single, stable phenotype
to each subject based on their entire longitudinal record, ensuring consistency
and accurately calculating descriptive statistics.

It now also saves the final merged DataFrame to a CSV file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import statsmodels.api as sm
from scipy.stats import sem
import os
import logging
import traceback
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File paths
BASE_DATA_PATH = '/media/hcs-sci-psy-narun/Jack'
G_FACTOR_FILE = os.path.join(BASE_DATA_PATH, 'g_factor_analysis/longitudinal_g_factor_regularized_with_metadata.csv')
NDAR_FILE = os.path.join(BASE_DATA_PATH, "ndar_subject01.csv")

# Define phenotype markers, labels, and colors
PHENOTYPE_MARKERS = {1: 'o', 2: 's', 3: '^', 4: 'X', 5: '*'}
PHENOTYPE_LABELS = {1: "Control", 2: "Subthreshold", 3: "ADHD", 4: "Not Clean Control", 5: "Not Assessed"}
PHENOTYPE_COLORS = {1: 'blue', 2: 'purple', 3: 'red', 4: 'green', 5: 'gray'}


################################################################################
#                   NEW ROBUST PHENOTYPE ASSIGNMENT LOGIC                      #
################################################################################

def pre_process_phenotype_data(ndar_filepath):
    """
    Loads and cleans the raw NDAR phenotype CSV file.
    """
    logger.info("Loading and pre-processing NDAR phenotype data...")
    try:
        df = pd.read_csv(ndar_filepath, low_memory=False)
    except FileNotFoundError:
        logger.error(f"ERROR: NDAR phenotype file not found at {ndar_filepath}")
        return None

    df['subjectkey_clean'] = df['subjectkey'].str.replace('NDAR_', 'NDAR').str.replace('_', '')

    pheno_map = {'ATA': 4, 'Nonspectrum': 5}
    df['phenotype_clean'] = df['phenotype'].replace(pheno_map)
    df['phenotype_clean'] = pd.to_numeric(df['phenotype_clean'], errors='coerce')

    logger.info(f"Finished pre-processing. Found {df['subjectkey_clean'].nunique()} unique subjects in phenotype file.")
    return df


def determine_stable_phenotype(pheno_df):
    """
    Applies hierarchical rules to determine a single, stable phenotype for each subject.
    """
    logger.info("Determining stable phenotype for each subject...")
    pheno_map = {}
    grouped = pheno_df.groupby('subjectkey_clean')

    for subject_id, group in grouped:
        phenotypes = group['phenotype_clean'].dropna().tolist()

        if 3.0 in phenotypes:
            pheno_map[subject_id] = 3
            continue

        filtered_phenotypes = [p for p in phenotypes if p != 5.0]

        if not filtered_phenotypes:
            pheno_map[subject_id] = 5
            continue

        counts = Counter(filtered_phenotypes)
        count_1 = counts.get(1.0, 0)
        count_4 = counts.get(4.0, 0)

        if count_1 > count_4:
            pheno_map[subject_id] = 1
        elif count_4 > count_1:
            pheno_map[subject_id] = 4
        elif count_1 == count_4 and count_1 > 0:
            pheno_map[subject_id] = 4
        else:
            pheno_map[subject_id] = counts.most_common(1)[0][0]

    logger.info(f"Assigned stable phenotypes to {len(pheno_map)} subjects.")
    return pheno_map


def add_phenotype_to_g_factor_data(g_df, stable_phenotype_map):
    """
    Adds the stable phenotype information to the g-factor data.
    """
    logger.info("Merging stable phenotypes into g-factor data...")
    g_df['phenotype'] = g_df['subject_id'].map(stable_phenotype_map)

    # --- Descriptive Statistics ---
    matched_subjects = g_df.dropna(subset=['phenotype'])
    n_matched = matched_subjects['subject_id'].nunique()
    n_total = g_df['subject_id'].nunique()

    logger.info(f"Successfully merged phenotypes for {n_matched} out of {n_total} subjects in the g-factor file.")

    logger.info("\n--- FINAL PHENOTYPE COUNTS FOR G-FACTOR SAMPLE ---")
    phenotype_counts = matched_subjects.groupby('phenotype')['subject_id'].nunique()
    for phenotype_code, count in phenotype_counts.items():
        label = PHENOTYPE_LABELS.get(int(phenotype_code), "Unknown")
        logger.info(f"  - {label} (Code {int(phenotype_code)}): {count} subjects")
    logger.info("--------------------------------------------------\n")

    return g_df


################################################################################
#                       EXISTING SCRIPT FUNCTIONS                            #
################################################################################

def load_g_factor_data(file_path):
    """
    Load the g-factor data from CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded g-factor data with {len(df)} rows and {len(df.columns)} columns")
        df['age_years'] = pd.to_numeric(df['age_years'], errors='coerce')
        df['g'] = pd.to_numeric(df['g'], errors='coerce')
        df.dropna(subset=['subject_id', 'age_years', 'g'], inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error loading g-factor data: {e}")
        return pd.DataFrame()


def analyze_timepoints(df):
    """
    Analyze the number of timepoints per subject.
    """
    if 'phenotype' not in df.columns or df['phenotype'].isna().all():
        logger.warning("Phenotype column not available for timepoint analysis.")
        return

    logger.info("Timepoint analysis by phenotype:")
    for phenotype, label in PHENOTYPE_LABELS.items():
        phenotype_df = df[df['phenotype'] == phenotype]
        if not phenotype_df.empty:
            phenotype_subjects = phenotype_df['subject_id'].nunique()
            multi_timepoint = sum(1 for subj in phenotype_df['subject_id'].unique()
                                  if len(phenotype_df[phenotype_df['subject_id'] == subj]) > 1)
            # Avoid division by zero
            percent_multi = (multi_timepoint / phenotype_subjects * 100) if phenotype_subjects > 0 else 0
            logger.info(f"  - {label}: {phenotype_subjects} subjects, " +
                        f"{multi_timepoint} with multiple timepoints " +
                        f"({percent_multi:.1f}%)")


def create_longitudinal_plot_with_phenotypes(df, output_path=None):
    """
    Create a longitudinal plot for g-factor scores with trend lines separated by phenotype.
    """
    plt.figure(figsize=(24, 18))

    num_participants = df['subject_id'].nunique()
    # Create a stable color map for subjects
    unique_subjects = sorted(df['subject_id'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, num_participants))
    color_map = {subject: colors[i] for i, subject in enumerate(unique_subjects)}

    for subject, subject_data in df.groupby('subject_id'):
        subject_data = subject_data.sort_values('age_years')
        if len(subject_data) > 1:
            plt.plot(subject_data['age_years'], subject_data['g'],
                     color=color_map[subject], linestyle='-', alpha=0.6)

        phenotype = subject_data['phenotype'].iloc[0] if pd.notna(subject_data['phenotype'].iloc[0]) else 5
        marker = PHENOTYPE_MARKERS.get(phenotype, '*')
        plt.scatter(subject_data['age_years'], subject_data['g'],
                    marker=marker, color=color_map[subject], s=40, alpha=0.8, edgecolors='w')

    # Create trend lines for each phenotype group
    def add_trend_line(data, color, label):
        if len(data) < 10: return
        X = sm.add_constant(data['age_years'])
        model = sm.OLS(data['g'], X).fit()
        x_pred = np.linspace(data['age_years'].min(), data['age_years'].max(), 100)
        y_pred = model.predict(sm.add_constant(x_pred))
        plt.plot(x_pred, y_pred, color=color, linewidth=4, label=f"{label} Trend (R²: {model.rsquared:.3f})")

    for phenotype_code, group_data in df.groupby('phenotype'):
        if pd.notna(phenotype_code):
            add_trend_line(group_data, PHENOTYPE_COLORS[int(phenotype_code)], PHENOTYPE_LABELS[int(phenotype_code)])

    plt.xlabel('Age (years)', fontsize=24)
    plt.ylabel('G-Factor Score', fontsize=24)
    plt.title('Longitudinal G-Factor Scores by Age and Phenotype', fontsize=28)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def main():
    """Main function to create the g-factor longitudinal plot with stable phenotypes."""
    try:
        # Load and process phenotype data to get stable assignments
        pheno_df = pre_process_phenotype_data(NDAR_FILE)
        if pheno_df is None: return

        stable_phenotype_map = determine_stable_phenotype(pheno_df)

        # Load the g-factor data
        g_factor_df = load_g_factor_data(G_FACTOR_FILE)
        if g_factor_df.empty: return

        # Add the stable phenotype information to g-factor data
        g_factor_with_phenotype = add_phenotype_to_g_factor_data(g_factor_df, stable_phenotype_map)

        # Analyze timepoints per subject
        analyze_timepoints(g_factor_with_phenotype)

        # --- SAVE THE MERGED DATAFRAME ---
        # Create output directory for the data
        data_output_dir = os.path.join(os.path.dirname(G_FACTOR_FILE), "data_stable_pheno")
        os.makedirs(data_output_dir, exist_ok=True)
        data_output_path = os.path.join(data_output_dir, "g_score_with_stable_phenotype.csv")

        # Save the dataframe to CSV
        g_factor_with_phenotype.to_csv(data_output_path, index=False)
        logger.info(f"Saved g-factor data with stable phenotypes to: {data_output_path}")

        # --- CREATE THE PLOT ---
        # Create output directory for the plot
        plot_output_dir = os.path.join(os.path.dirname(G_FACTOR_FILE), "plots_stable_pheno")
        os.makedirs(plot_output_dir, exist_ok=True)
        plot_output_path = os.path.join(plot_output_dir, "g_factor_longitudinal_plot_stable_phenotypes.png")

        # Create the final plot
        create_longitudinal_plot_with_phenotypes(g_factor_with_phenotype, plot_output_path)

        logger.info("G-Factor longitudinal plot creation with stable phenotypes completed successfully")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
