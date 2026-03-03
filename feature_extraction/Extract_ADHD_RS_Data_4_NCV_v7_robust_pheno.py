#!/usr/bin/env python3
"""
ADHD-RS Score Extraction Script with Robust Stable Phenotype Filtering

This script has been reworked to use the definitive list of subjects and their
stable phenotypes derived from the full longitudinal record. It filters the
ADHD-RS data to match ONLY the subjects present in the final neuroimaging
analysis sample. Includes detailed sanity checks to report N at each step.
"""

import pandas as pd
import numpy as np
import os
import logging
import re
from datetime import datetime
from collections import defaultdict
import traceback

# ===== CONFIGURATION PARAMETERS =====
CONFIG = {
    'AGE_TOLERANCE_MONTHS': 6,
    'DATE_TOLERANCE_DAYS': 180,
    'FMRI_SCAN_PATTERNS': ['fMRI', 'rfMRI', 'rsFMRI', 'task-fMRI', 'func', 'BOLD'],
}

# --- NEW: Define path to the final phenotype breakdown ---
FINAL_PHENO_PATH = '/media/hcs-sci-psy-narun/Jack/First_level_model_aligned/final_subjects_phenotype_breakdown.csv'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
BASE_DATA_PATH = '/media/hcs-sci-psy-narun/Jack'
FIRST_LEVEL_MODEL_PATH = '/media/hcs-sci-psy-narun/Jack/First_level_model_ADHDRS_stable_pheno'  # New output dir
IMAGE_FILE = os.path.join(BASE_DATA_PATH, "image03_copy.csv")
ADHDRS_FILE = os.path.join(BASE_DATA_PATH, "adhdrs_copy.csv")
FOLD_ASSIGNMENTS_PATH = os.path.join(BASE_DATA_PATH, "First_level_model_aligned/nested_fold_assignments.csv")

# ADHD-RS variables to extract
ADHDRS_VARIABLES = {
    'inattention': 'p_c_adhdrs_int_rs',
    'hyperactivity': 'p_c_adhdrs_hyp_rs',
}

PHENOTYPE_LABELS = {
    1: "Control", 2: "Subthreshold", 3: "ADHD", 4: "Not Clean Control"
}


def load_and_preprocess_data(file_path, has_subheader=False):
    """Load and preprocess data from CSV files."""
    try:
        df = pd.read_csv(file_path, skiprows=[1] if has_subheader else None, low_memory=False)
        if 'subjectkey' in df.columns:
            df['subjectkey'] = df['subjectkey'].astype(str).str.replace('_', '')
        logger.info(f"Loaded data from {file_path}: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()


def clean_missing_values(df, variable_name):
    """Clean missing values for ADHD-RS variables."""
    if df.empty or variable_name not in df.columns: return df
    result = df.copy()
    result[variable_name] = pd.to_numeric(result[variable_name], errors='coerce')
    missing_codes = [-999, 999, -888, 888, -777, 777]
    result[variable_name] = result[variable_name].replace(missing_codes, np.nan)
    return result


def enhanced_fmri_filtering(image_df):
    """Enhanced fMRI filtering with multiple scan type patterns."""
    logger.info("Applying enhanced fMRI filtering...")
    fmri_mask = image_df['scan_type'].str.contains('|'.join(CONFIG['FMRI_SCAN_PATTERNS']), na=False, case=False)
    fmri_df = image_df[fmri_mask]
    logger.info(f"Total fMRI entries after enhanced filtering: {len(fmri_df)}")
    return fmri_df


def match_data(fmri_row, subject_adhdrs_data):
    """Finds the best matching ADHD-RS record for a given fMRI scan."""
    try:
        fmri_age = pd.to_numeric(fmri_row.get('interview_age'), errors='coerce')
        fmri_date = pd.to_datetime(fmri_row.get('interview_date'), errors='coerce')

        # 1. Try date matching first
        if pd.notna(fmri_date):
            for _, adhdrs_row in subject_adhdrs_data.iterrows():
                adhdrs_date = pd.to_datetime(adhdrs_row.get('interview_date'), errors='coerce')
                if pd.notna(adhdrs_date) and abs((fmri_date - adhdrs_date).days) <= CONFIG['DATE_TOLERANCE_DAYS']:
                    return adhdrs_row

        # 2. Fallback to age matching
        if pd.notna(fmri_age):
            best_age_match = None
            best_age_diff = float('inf')
            for _, adhdrs_row in subject_adhdrs_data.iterrows():
                adhdrs_age = pd.to_numeric(adhdrs_row.get('interview_age'), errors='coerce')
                if pd.notna(adhdrs_age):
                    age_diff = abs(fmri_age - adhdrs_age)
                    if age_diff <= CONFIG['AGE_TOLERANCE_MONTHS'] and age_diff < best_age_diff:
                        best_age_match = adhdrs_row
                        best_age_diff = age_diff
            if best_age_match is not None:
                return best_age_match

    except Exception as e:
        logger.warning(f"Error during matching for subject {fmri_row['subjectkey']}: {e}")
    return None


def main():
    """Main function to extract ADHD-RS scores filtered by the final subject list."""
    logger.info("Starting ADHD-RS score extraction with STABLE PHENOTYPE FILTERING")
    os.makedirs(FIRST_LEVEL_MODEL_PATH, exist_ok=True)

    # --- 1. Load the definitive list of final subjects and their phenotypes ---
    try:
        final_subjects_df = pd.read_csv(FINAL_PHENO_PATH)
        final_subjects_set = set(final_subjects_df['subject_id'].unique())
        logger.info(f"SANITY CHECK: Loaded {len(final_subjects_set)} final subjects to be included in this analysis.")
    except FileNotFoundError:
        logger.error(f"CRITICAL ERROR: Final phenotype breakdown file not found at {FINAL_PHENO_PATH}")
        return

    # --- 2. Load all raw data ---
    image_df = load_and_preprocess_data(IMAGE_FILE, has_subheader=True)
    adhdrs_df = load_and_preprocess_data(ADHDRS_FILE, has_subheader=True)
    fold_df = pd.read_csv(FOLD_ASSIGNMENTS_PATH)

    if any(df.empty for df in [image_df, adhdrs_df, fold_df]):
        logger.error("One or more essential data files failed to load. Exiting.")
        return

    # Clean ADHD-RS data
    for var_name in ADHDRS_VARIABLES.values():
        adhdrs_df = clean_missing_values(adhdrs_df, var_name)

    # --- 3. Filter and Match Data ---
    fmri_df = enhanced_fmri_filtering(image_df)

    fmri_df_filtered = fmri_df[fmri_df['subjectkey'].isin(final_subjects_set)]
    logger.info(
        f"SANITY CHECK: Filtered fMRI data to {len(fmri_df_filtered)} scans from {fmri_df_filtered['subjectkey'].nunique()} final subjects.")

    all_records = []
    for subject_key, subject_fmri_group in fmri_df_filtered.groupby('subjectkey'):
        subject_adhdrs_data = adhdrs_df[adhdrs_df['subjectkey'] == subject_key]
        if subject_adhdrs_data.empty:
            continue

        for _, fmri_row in subject_fmri_group.iterrows():
            matched_adhdrs_row = match_data(fmri_row, subject_adhdrs_data)

            if matched_adhdrs_row is not None:
                age = fmri_row['interview_age'] if pd.notna(fmri_row['interview_age']) else matched_adhdrs_row[
                    'interview_age']
                record = {
                    'subject_id': subject_key,
                    'age_months': age,
                    'inattention': matched_adhdrs_row[ADHDRS_VARIABLES['inattention']],
                    'hyperactivity': matched_adhdrs_row[ADHDRS_VARIABLES['hyperactivity']]
                }
                all_records.append(record)

    if not all_records:
        logger.error("No matching records found after filtering and matching. Exiting.")
        return

    # --- 4. Final Processing and Saving ---
    matched_df_raw = pd.DataFrame(all_records)
    logger.info(f"SANITY CHECK: Found {len(matched_df_raw)} potential timepoint matches before cleaning.")

    matched_df = matched_df_raw.dropna(subset=['age_months', 'inattention', 'hyperactivity'])
    matched_df = matched_df.drop_duplicates(subset=['subject_id', 'age_months'])
    logger.info(
        f"SANITY CHECK: Created final matched dataset with {len(matched_df)} timepoints from {matched_df['subject_id'].nunique()} subjects after cleaning.")

    # Merge with the stable phenotype and fold assignments
    final_data = pd.merge(matched_df, final_subjects_df, on='subject_id', how='inner')
    fold_assignments = fold_df[['subject_id', 'outer_fold', 'inner_split']].drop_duplicates()
    final_data = pd.merge(final_data, fold_assignments, on='subject_id', how='left')
    logger.info(f"SANITY CHECK: Final dataset size after merging fold assignments: {len(final_data)} rows.")

    logger.info("Final dataset prepared. Saving to fold directories...")

    # Create and save files for each fold
    for fold in range(5):
        fold_dir = os.path.join(FIRST_LEVEL_MODEL_PATH, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        fold_data = final_data[final_data['outer_fold'] == fold]

        for split in ['train', 'test']:
            split_data = fold_data[fold_data['inner_split'] == split]

            for score_type in ['inattention', 'hyperactivity']:
                output_df = split_data[['subject_id', 'age_months', score_type, 'phenotype']]
                output_df = output_df.rename(columns={'phenotype': 'final_phenotype'})

                score_dir = os.path.join(fold_dir, score_type)
                os.makedirs(score_dir, exist_ok=True)
                output_path = os.path.join(score_dir, f"{split}.csv")
                output_df.to_csv(output_path, index=False)

                logger.info(f"  - Saved fold_{fold}/{score_type}/{split}.csv with N = {len(output_df)}")

    logger.info("All files saved successfully.")


if __name__ == "__main__":
    main()
