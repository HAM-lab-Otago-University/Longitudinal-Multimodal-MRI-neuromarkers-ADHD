#!/usr/bin/env python3
"""
Script to create perfectly aligned cognition and hyperactivity datasets
Filters both datasets to only include matching observations
"""

import pandas as pd
import os
from pathlib import Path

# Paths
cognition_path = "/media/hcs-sci-psy-narun/Jack/First_level_model_aligned/Stacked_model_ridge_v4_EXTREME_conserve_a2000_keeper/stacked_ridge_predictions.csv"
hyperactivity_base_path = "/media/hcs-sci-psy-narun/Jack/First_level_model_ADHDRS_6month/hyperactivity"
output_dir = "/media/hcs-sci-psy-narun/Jack/LME_cog_hyperactivity_phenotype_ADHD"


def load_all_data():
    """Load cognition and hyperactivity data"""
    print("Loading cognition data...")
    cog_data = pd.read_csv(cognition_path)
    print(f"Cognition data: {len(cog_data)} observations")

    print("Loading hyperactivity data...")
    all_hyperactivity = []

    for fold in range(5):  # fold_0 through fold_4
        fold_path = os.path.join(hyperactivity_base_path, f"fold_{fold}", "test.csv")

        if os.path.exists(fold_path):
            fold_data = pd.read_csv(fold_path)
            fold_data['fold'] = fold  # Add fold identifier
            all_hyperactivity.append(fold_data)
        else:
            print(f"Warning: {fold_path} not found")

    hyper_data = pd.concat(all_hyperactivity, ignore_index=True)
    print(f"Hyperactivity data: {len(hyper_data)} observations")

    return cog_data, hyper_data


def filter_and_align(cog_data, hyper_data):
    """Filter both datasets to only include matching observations"""
    print("\nFinding exact matches...")

    # Merge to find exact matches
    merged = pd.merge(cog_data, hyper_data,
                      left_on='subject_id', right_on='eid',
                      how='inner', suffixes=('', '_hyper'))

    print(f"Found {len(merged)} exact matches")

    # Extract the matching observations for cognition data
    # Keep only the original cognition columns
    cog_filtered = merged[['fold', 'subject_id', 'true_g', 'stacked_pred', 'split']].copy()

    # Extract the matching observations for hyperactivity data
    # Keep original hyperactivity columns plus fold info
    hyper_filtered = merged[['eid', 'subject_id', 'age_months', 'hyperactivity', 'fold']].copy()

    # Verify alignment
    print(f"\nVerification:")
    print(f"Cognition filtered: {len(cog_filtered)} observations")
    print(f"Hyperactivity filtered: {len(hyper_filtered)} observations")

    # Check if subject_ids are identical and in same order
    cog_subjects = cog_filtered['subject_id'].values
    hyper_subjects = hyper_filtered['subject_id'].values

    if (cog_subjects == hyper_subjects).all():
        print("Perfect alignment: subject_ids match exactly in same order")
    else:
        print("Warning: subject_ids don't match perfectly")

    # Sort both by subject_id to ensure alignment
    cog_filtered = cog_filtered.sort_values('subject_id').reset_index(drop=True)
    hyper_filtered = hyper_filtered.sort_values('subject_id').reset_index(drop=True)

    print(f"After sorting - alignment check:")
    cog_subjects_sorted = cog_filtered['subject_id'].values
    hyper_subjects_sorted = hyper_filtered['subject_id'].values

    if (cog_subjects_sorted == hyper_subjects_sorted).all():
        print("Perfect alignment after sorting")
    else:
        print("Still misaligned after sorting - investigate!")

    return cog_filtered, hyper_filtered


def save_filtered_data(cog_filtered, hyper_filtered):
    """Save the filtered datasets"""
    print(f"\nSaving filtered datasets...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save filtered cognition data
    cog_output_path = os.path.join(output_dir, "stacked_ridge_predictions_matched.csv")
    cog_filtered.to_csv(cog_output_path, index=False)
    print(f"Saved filtered cognition data: {cog_output_path}")

    # Save filtered hyperactivity data
    hyper_output_path = os.path.join(output_dir, "hyperactivity_scores_matched.csv")
    hyper_filtered.to_csv(hyper_output_path, index=False)
    print(f"Saved filtered hyperactivity data: {hyper_output_path}")

    return cog_output_path, hyper_output_path


def final_verification(cog_filtered, hyper_filtered):
    """Final checks on the filtered data"""
    print(f"\n{'FINAL VERIFICATION':=^60}")

    print(f"Cognition data shape: {cog_filtered.shape}")
    print(f"Hyperactivity data shape: {hyper_filtered.shape}")
    print(f"Unique subjects in cognition: {cog_filtered['subject_id'].nunique()}")
    print(f"Unique subjects in hyperactivity: {hyper_filtered['subject_id'].nunique()}")

    # Show distribution of timepoints per subject
    cog_counts = cog_filtered.groupby('subject_id').size()
    hyper_counts = hyper_filtered.groupby('subject_id').size()

    print(f"\nTimepoints per subject distribution (cognition):")
    print(cog_counts.value_counts().sort_index())

    print(f"\nTimepoints per subject distribution (hyperactivity):")
    print(hyper_counts.value_counts().sort_index())

    # Show some examples
    print(f"\nFirst 10 rows comparison:")
    print("Cognition:")
    print(cog_filtered[['subject_id', 'true_g', 'stacked_pred']].head(10))
    print("\nHyperactivity:")
    print(hyper_filtered[['subject_id', 'age_months', 'hyperactivity']].head(10))

    # Summary stats for hyperactivity scores
    print(f"\nHyperactivity score statistics:")
    print(hyper_filtered['hyperactivity'].describe())


def main():
    """Main function"""
    print("Creating aligned cognition and hyperactivity datasets...")
    print("=" * 60)

    # Load data
    cog_data, hyper_data = load_all_data()

    # Filter and align
    cog_filtered, hyper_filtered = filter_and_align(cog_data, hyper_data)

    # Save filtered datasets
    cog_path, hyper_path = save_filtered_data(cog_filtered, hyper_filtered)

    # Final verification
    final_verification(cog_filtered, hyper_filtered)

    print(f"\n{'SUCCESS':=^60}")
    print("Created perfectly aligned datasets:")
    print(f"  Cognition: {cog_path}")
    print(f"  Hyperactivity: {hyper_path}")
    print(f"  Both contain {len(cog_filtered)} observations")
    print("Ready for mediation analysis in R!")


if __name__ == "__main__":
    main()