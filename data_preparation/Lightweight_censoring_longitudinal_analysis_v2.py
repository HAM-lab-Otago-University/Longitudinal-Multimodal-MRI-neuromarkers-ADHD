#!/usr/bin/env python3
"""
K J Scott 1/4/25

Longitudinal Data Assessment Script for fMRI Censoring with Phenotype Analysis

This script analyzes the impact of volume censoring on longitudinal data,
determining if we lose critical time points when censoring at FD > 0.5mm,
with additional analysis by phenotype category, gives impact of censoring by phenotype according to OREGON NDAR csv.
"""

import os
import pandas as pd
import numpy as np
import re
import argparse
from collections import defaultdict

# Configuration variables
INPUT_FILE = "/media/hcs-sci-psy-narun/Jack/XCP-D_full_dataset_ExamineSubs_no_censor_May/volumes_censored_detailed.csv"
OUTPUT_DIR = "/media/hcs-sci-psy-narun/Jack/XCP-D_full_dataset_ExamineSubs_no_censor_May/"
MIN_RUNS_PER_TIMEPOINT = 2
PERC_REMAINING_THRESHOLD = 50
NDAR_FILE = "/media/hcs-sci-psy-narun/Jack/ndar_subject01.csv"  # Path to the NDAR subjects file with phenotype information


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Analyze impact of censoring on longitudinal data.')

    parser.add_argument('--input-file', type=str, default=INPUT_FILE,
                        help=f'Path to volumes_censored_detailed.csv (default: {INPUT_FILE})')

    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help=f'Directory to save output files (default: {OUTPUT_DIR})')

    parser.add_argument('--min-runs', type=int, default=MIN_RUNS_PER_TIMEPOINT,
                        help=f'Minimum runs required per time point (default: {MIN_RUNS_PER_TIMEPOINT})')

    parser.add_argument('--remaining-threshold', type=float, default=PERC_REMAINING_THRESHOLD,
                        help=f'Minimum percentage of volumes that must remain after censoring (default: {PERC_REMAINING_THRESHOLD}%)')

    parser.add_argument('--ndar-file', type=str, default=NDAR_FILE,
                        help='Path to NDAR subjects file with phenotype information')

    return parser.parse_args()


def load_ndar_phenotypes(ndar_file):
    """
    Load NDAR subjects file and extract phenotype information.
    Handles the case where phenotype info is displaced to the sibling_study column.

    Returns:
        Dictionary mapping (NDAR subject key, age) to phenotype categories (1-4)
    """
    print(f"Loading phenotype data from {ndar_file}...")
    try:
        ndar_df = pd.read_csv(ndar_file)
        print(f"Loaded NDAR file with {len(ndar_df)} subjects and {len(ndar_df.columns)} columns")

        # Create mapping from (NDAR ID, age) to phenotype
        phenotype_map = {}

        for _, row in ndar_df.iterrows():
            # Extract subject key (standardize format by removing underscore after NDAR if present)
            subject_key = row['subjectkey'] if 'subjectkey' in row else row.get('ndar_subject01_id', '')
            if isinstance(subject_key, str) and subject_key.startswith('NDAR_'):
                subject_key = subject_key.replace('NDAR_', 'NDAR', 1)

            # Extract age in months
            age_months = None
            if 'interview_age' in row and not pd.isna(row['interview_age']):
                try:
                    age_months = int(row['interview_age'])
                except (ValueError, TypeError):
                    age_months = None

            # Check phenotype column (K) first
            phenotype = None
            if 'phenotype' in row and not pd.isna(row['phenotype']):
                try:
                    # Try to convert to int - if successful, this is our phenotype
                    phenotype = int(float(row['phenotype']))
                except (ValueError, TypeError):
                    # Not a number, likely displaced data
                    phenotype = None

            # If phenotype not found, check sibling_study column (N)
            if phenotype is None and 'sibling_study' in row and not pd.isna(row['sibling_study']):
                try:
                    phenotype = int(float(row['sibling_study']))
                    if phenotype < 1 or phenotype > 4:  # Validate the value
                        phenotype = None
                except (ValueError, TypeError):
                    phenotype = None

            # Only add to map if we found a valid phenotype and have age information
            if phenotype is not None and isinstance(subject_key, str) and age_months is not None:
                # Store with age in months for matching
                phenotype_map[(subject_key, age_months)] = phenotype

                # Also store without age as fallback
                if subject_key not in phenotype_map:
                    phenotype_map[subject_key] = phenotype

        print(f"Found valid phenotype information for {len(phenotype_map)} subject-timepoints")
        # Print distribution of phenotypes for unique subjects
        subject_phenotypes = {}
        for key, phenotype in phenotype_map.items():
            if isinstance(key, str):  # This is a subject ID without age
                subject_phenotypes[key] = phenotype

        phenotype_counts = {
            1: sum(1 for p in subject_phenotypes.values() if p == 1),
            2: sum(1 for p in subject_phenotypes.values() if p == 2),
            3: sum(1 for p in subject_phenotypes.values() if p == 3),
            4: sum(1 for p in subject_phenotypes.values() if p == 4)
        }
        print(f"Phenotype distribution: Control: {phenotype_counts[1]}, Subthreshold: {phenotype_counts[2]}, "
              f"ADHD: {phenotype_counts[3]}, Not ADHD/Not Clean Control: {phenotype_counts[4]}")

        return phenotype_map

    except Exception as e:
        print(f"Error loading NDAR phenotypes: {e}")
        import traceback
        print(traceback.format_exc())
        return {}


def parse_subject_id(subject_id):
    """
    Parse the complex subject ID to extract true subject ID, age, and study year.
    """
    # Extract true subject ID (everything before "Age")
    true_subject_match = re.match(r'(sub-[A-Z0-9]+)Age', subject_id)

    if not true_subject_match:
        # Handle case where the pattern doesn't match
        return {
            'true_subject': subject_id,
            'age_years': None,
            'age_months': None,
            'study_year': None,
            'total_age_months': None
        }

    true_subject = true_subject_match.group(1)

    # Extract age information
    age_match = re.search(r'Age(\d+)Months', subject_id)

    age_years = None
    age_months = None
    total_age_months = None

    if age_match:
        age_str = age_match.group(1)
        if len(age_str) == 3:  # Format: Age091Months (9 years, 1 month)
            age_years = int(age_str[0:2])
            age_months = int(age_str[2])
            total_age_months = age_years * 12 + age_months
        elif len(age_str) == 4:  # Format: Age0910Months (9 years, 10 months)
            age_years = int(age_str[0:2])
            age_months = int(age_str[2:4])
            total_age_months = age_years * 12 + age_months
        else:  # Just in case there's a different format
            age_years = int(age_str)
            age_months = 0
            total_age_months = age_years * 12

    # Extract study year
    study_year_match = re.search(r'StudyYear(\d+)', subject_id)
    study_year = int(study_year_match.group(1)) if study_year_match else None

    return {
        'true_subject': true_subject,
        'age_years': age_years,
        'age_months': age_months,
        'study_year': study_year,
        'total_age_months': total_age_months
    }


def is_run_usable(row, remaining_threshold):
    """
    Determine if a run is usable after censoring based on remaining percentage.
    """
    # Skip combined runs (run_id = 'unknown')
    if row['run_id'] == 'unknown':
        return False

    # Check if enough volumes remain after censoring
    return row['perc_remaining_post_censoring'] >= remaining_threshold


def get_phenotype_category(subject_info, phenotype_map):
    """
    Get the phenotype category for a subject based on the phenotype map.

    Args:
        subject_info: Dictionary with 'true_subject' and 'total_age_months'
        phenotype_map: Dictionary mapping (NDAR ID, age) to phenotype categories

    Returns:
        String: 'control', 'subthreshold', 'adhd', 'other', or 'unknown'
    """
    if not phenotype_map:
        return 'unknown'

    true_subject = subject_info['true_subject']
    total_age_months = subject_info['total_age_months']

    # Extract the NDAR ID from the true subject
    ndar_match = re.search(r'sub-(NDAR[A-Z0-9]+)', true_subject)
    if not ndar_match:
        return 'unknown'

    ndar_id = ndar_match.group(1)

    # Try to match using both ID and age
    phenotype = None
    if total_age_months is not None:
        # Allow for some age flexibility (±1 month)
        for age_offset in [0, -1, 1, -2, 2]:
            match_age = total_age_months + age_offset
            if (ndar_id, match_age) in phenotype_map:
                phenotype = phenotype_map[(ndar_id, match_age)]
                break

    # Fallback to just using ID if not found with age
    if phenotype is None:
        phenotype = phenotype_map.get(ndar_id)

    # Map phenotype value to category string
    if phenotype == 1:
        return 'control'
    elif phenotype == 2:
        return 'subthreshold'
    elif phenotype == 3:
        return 'adhd'
    elif phenotype == 4:
        return 'other'
    else:
        return 'unknown'


def analyze_longitudinal_data(df, min_runs_per_timepoint, remaining_threshold, phenotype_map=None):
    """
    Analyze the impact of censoring on longitudinal data.

    Args:
        df: DataFrame with censoring data
        min_runs_per_timepoint: Minimum runs required per timepoint
        remaining_threshold: Minimum percentage of volumes that must remain after censoring
        phenotype_map: Dictionary mapping (NDAR ID, age) to phenotype categories

    Returns:
        Dictionary with analysis results
    """
    # Parse subject information
    df['parsed_info'] = df['subject_id'].apply(parse_subject_id)
    df['true_subject'] = df['parsed_info'].apply(lambda x: x['true_subject'])
    df['study_year'] = df['parsed_info'].apply(lambda x: x['study_year'])
    df['total_age_months'] = df['parsed_info'].apply(lambda x: x['total_age_months'])

    # Determine which runs are usable after censoring
    df['usable_after_censoring'] = df.apply(
        lambda row: is_run_usable(row, remaining_threshold), axis=1
    )

    # Initialize results for all true subjects
    true_subjects = df['true_subject'].unique()

    results = {
        'subjects_analyzed': len(true_subjects),
        'subjects_affected': 0,
        'subjects_lost_completely': 0,
        'timepoints_before': 0,
        'timepoints_after': 0,
        'runs_before': 0,
        'runs_after': 0,
        'subject_details': [],
        # Add phenotype-specific counters
        'phenotype_counts': {
            'control': {'total': 0, 'affected': 0, 'lost': 0},
            'subthreshold': {'total': 0, 'affected': 0, 'lost': 0},
            'adhd': {'total': 0, 'affected': 0, 'lost': 0},
            'other': {'total': 0, 'affected': 0, 'lost': 0},
            'unknown': {'total': 0, 'affected': 0, 'lost': 0}
        }
    }

    # Analyze each true subject
    for subject in true_subjects:
        subject_df = df[df['true_subject'] == subject]

        # Get the study years for this subject
        study_years = sorted(subject_df['study_year'].dropna().unique())

        # Get most frequent age in months for this subject (for phenotype lookup)
        subject_info = {
            'true_subject': subject,
            'total_age_months': subject_df['total_age_months'].mode().iloc[0] if not subject_df[
                'total_age_months'].isna().all() else None
        }

        # Get phenotype for this subject
        phenotype_category = get_phenotype_category(subject_info, phenotype_map)

        subject_result = {
            'true_subject': subject,
            'phenotype': phenotype_category,
            'study_years': study_years,
            'timepoints_before': len(study_years),
            'timepoints_after': 0,
            'total_runs_before': len(subject_df[subject_df['run_id'] != 'unknown']),
            'total_runs_after': 0,
            'timepoint_details': []
        }

        # Analyze each study year
        timepoints_lost = 0

        for year in study_years:
            year_df = subject_df[subject_df['study_year'] == year]
            # Filter out combined runs
            year_df = year_df[year_df['run_id'] != 'unknown']
            usable_year_df = year_df[year_df['usable_after_censoring']]

            runs_before = len(year_df)
            runs_after = len(usable_year_df)

            # Check if we still have enough runs for this time point
            has_enough_runs = runs_after >= min_runs_per_timepoint

            timepoint_detail = {
                'study_year': year,
                'runs_before': runs_before,
                'runs_after': runs_after,
                'has_enough_runs': has_enough_runs
            }

            subject_result['timepoint_details'].append(timepoint_detail)

            if has_enough_runs:
                subject_result['timepoints_after'] += 1
            else:
                timepoints_lost += 1

            subject_result['total_runs_after'] += runs_after

        # Determine if this subject was affected
        subject_affected = subject_result['timepoints_after'] < subject_result['timepoints_before']
        subject_result['affected'] = subject_affected

        # Determine if this subject was completely lost
        subject_lost = subject_result['timepoints_after'] == 0 and subject_result['timepoints_before'] > 0
        subject_result['lost_completely'] = subject_lost

        # Update overall results
        if subject_affected:
            results['subjects_affected'] += 1

        if subject_lost:
            results['subjects_lost_completely'] += 1

        results['timepoints_before'] += subject_result['timepoints_before']
        results['timepoints_after'] += subject_result['timepoints_after']
        results['runs_before'] += subject_result['total_runs_before']
        results['runs_after'] += subject_result['total_runs_after']

        # Update phenotype-specific counts
        results['phenotype_counts'][phenotype_category]['total'] += 1
        if subject_affected:
            results['phenotype_counts'][phenotype_category]['affected'] += 1
        if subject_lost:
            results['phenotype_counts'][phenotype_category]['lost'] += 1

        results['subject_details'].append(subject_result)

    return results


def select_best_runs(df, min_runs_per_timepoint, remaining_threshold, phenotype_map=None):
    """
    For each subject and time point, select the best N runs based on remaining data after censoring.
    Similar to the approach in XCP-D_output_subject_stratification.py.
    """
    # First filter out combined runs and ensure we have parsed subject info
    if 'true_subject' not in df.columns or 'study_year' not in df.columns:
        df['parsed_info'] = df['subject_id'].apply(parse_subject_id)
        df['true_subject'] = df['parsed_info'].apply(lambda x: x['true_subject'])
        df['study_year'] = df['parsed_info'].apply(lambda x: x['study_year'])
        df['total_age_months'] = df['parsed_info'].apply(lambda x: x['total_age_months'])

    filtered_df = df[df['run_id'] != 'unknown'].copy()

    # Initialize list to store selected runs
    selected_runs = []

    # Get unique true subjects
    true_subjects = filtered_df['true_subject'].unique()

    for subject in true_subjects:
        subject_df = filtered_df[filtered_df['true_subject'] == subject]

        # Get most frequent age in months for this subject (for phenotype lookup)
        subject_info = {
            'true_subject': subject,
            'total_age_months': subject_df['total_age_months'].mode().iloc[0] if not subject_df[
                'total_age_months'].isna().all() else None
        }

        # Get phenotype for this subject
        phenotype_category = get_phenotype_category(subject_info, phenotype_map)

        # Get study years for this subject
        study_years = subject_df['study_year'].dropna().unique()

        for year in study_years:
            # Get runs for this subject and year
            year_df = subject_df[subject_df['study_year'] == year].copy()

            # Sort by percentage remaining after censoring (descending)
            year_df = year_df.sort_values('perc_remaining_post_censoring', ascending=False)

            # Select the best N runs, but only if they meet the remaining threshold
            usable_runs = year_df[year_df['perc_remaining_post_censoring'] >= remaining_threshold]

            # Get the top N runs (or all if less than N)
            top_runs = usable_runs.head(min_runs_per_timepoint)

            if len(top_runs) >= min_runs_per_timepoint:
                selected_runs.append({
                    'true_subject': subject,
                    'phenotype': phenotype_category,
                    'study_year': year,
                    'usable_runs': len(top_runs),
                    'avg_remaining': top_runs['perc_remaining_post_censoring'].mean(),
                    'runs': top_runs
                })

    return selected_runs


def create_summary_report(results, output_file):
    """
    Create a text summary report of the longitudinal analysis.
    """
    with open(output_file, 'w') as f:
        f.write("================================================\n")
        f.write("Longitudinal fMRI Data Censoring Impact Analysis\n")
        f.write("================================================\n\n")

        f.write(f"Total true subjects analyzed: {results['subjects_analyzed']}\n")
        f.write(f"Subjects affected by censoring: {results['subjects_affected']} "
                f"({results['subjects_affected'] / results['subjects_analyzed'] * 100:.1f}%)\n")
        f.write(f"Subjects lost completely: {results['subjects_lost_completely']} "
                f"({results['subjects_lost_completely'] / results['subjects_analyzed'] * 100:.1f}%)\n\n")

        f.write(f"Total timepoints before censoring: {results['timepoints_before']}\n")
        f.write(f"Total timepoints after censoring: {results['timepoints_after']}\n")
        f.write(f"Timepoints lost: {results['timepoints_before'] - results['timepoints_after']} "
                f"({(results['timepoints_before'] - results['timepoints_after']) / results['timepoints_before'] * 100:.1f}%)\n\n")

        f.write(f"Total runs before censoring: {results['runs_before']}\n")
        f.write(f"Total runs after censoring: {results['runs_after']}\n")
        f.write(f"Runs lost: {results['runs_before'] - results['runs_after']} "
                f"({(results['runs_before'] - results['runs_after']) / results['runs_before'] * 100:.1f}%)\n\n")

        # Add phenotype-specific analysis section
        f.write("Impact by Phenotype Category:\n")
        f.write("----------------------------\n\n")

        for category in ['control', 'subthreshold', 'adhd', 'other', 'unknown']:
            count = results['phenotype_counts'][category]
            if count['total'] > 0:
                f.write(f"{category.upper()} Subjects ({count['total']} total):\n")
                f.write(
                    f"  Affected by censoring: {count['affected']} ({count['affected'] / count['total'] * 100:.1f}%)\n")
                f.write(f"  Lost completely: {count['lost']} ({count['lost'] / count['total'] * 100:.1f}%)\n\n")

        f.write("Subject-level details for affected subjects:\n")
        f.write("-------------------------------------------\n\n")

        # Sort subjects by most affected (most timepoints lost)
        sorted_subjects = sorted(
            results['subject_details'],
            key=lambda x: (x['timepoints_before'] - x['timepoints_after']) if x['timepoints_before'] > 0 else 0,
            reverse=True
        )

        for subject in sorted_subjects:
            if subject['affected']:
                f.write(f"Subject: {subject['true_subject']} (Phenotype: {subject['phenotype']})\n")
                f.write(f"  Timepoints before: {subject['timepoints_before']}\n")
                f.write(f"  Timepoints after: {subject['timepoints_after']}\n")
                f.write(f"  Timepoints lost: {subject['timepoints_before'] - subject['timepoints_after']}\n")
                f.write(f"  Total runs before: {subject['total_runs_before']}\n")
                f.write(f"  Total runs after: {subject['total_runs_after']}\n")
                f.write("  Timepoint details:\n")

                for tp in subject['timepoint_details']:
                    status = "KEPT" if tp['has_enough_runs'] else "LOST"
                    f.write(
                        f"    Year {tp['study_year']}: {tp['runs_before']} runs → {tp['runs_after']} runs ({status})\n")

                f.write("\n")

        f.write("\n================================================\n")
        f.write("End of Report\n")
        f.write("================================================\n")


def main():
    """Main function for longitudinal censoring impact analysis."""
    # Parse arguments
    args = parse_arguments()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from {args.input_file}...")

    try:
        # Load the censoring data
        df = pd.read_csv(args.input_file)
        print(f"Loaded {len(df)} rows.")

        # Load phenotype data if available
        phenotype_map = None
        if args.ndar_file and os.path.exists(args.ndar_file):
            phenotype_map = load_ndar_phenotypes(args.ndar_file)

        # Verify input data has run-specific values
        if len(df) > 10:
            run_variation = df.groupby(['subject_id']).agg({
                'perc_gt_0.5': ['std']
            })
            mean_std = run_variation['perc_gt_0.5', 'std'].mean()
            if mean_std < 0.01:
                print("WARNING: Input data shows very little variation between runs. "
                      "Please verify your volumes_censored_detailed.csv has run-specific values.")

        # Analyze longitudinal impact
        print(f"Analyzing longitudinal impact with minimum {args.min_runs} runs and "
              f"{args.remaining_threshold}% remaining threshold...")
        results = analyze_longitudinal_data(df, args.min_runs, args.remaining_threshold, phenotype_map)

        # Create summary report
        report_file = os.path.join(args.output_dir, "longitudinal_censoring_impact.txt")
        create_summary_report(results, report_file)
        print(f"Created summary report: {report_file}")

        # Select best runs for each subject/timepoint
        print("Selecting best runs for each subject and timepoint...")
        best_runs = select_best_runs(df, args.min_runs, args.remaining_threshold, phenotype_map)

        # Create dataframe of usable subject/timepoints
        usable_timepoints = pd.DataFrame([
            {
                'true_subject': r['true_subject'],
                'phenotype': r['phenotype'],
                'study_year': r['study_year'],
                'usable_runs': r['usable_runs'],
                'avg_remaining_percentage': r['avg_remaining']
            } for r in best_runs
        ])

        # Save to CSV
        usable_file = os.path.join(args.output_dir, "usable_longitudinal_timepoints_v2.csv")
        usable_timepoints.to_csv(usable_file, index=False)
        print(f"Saved usable timepoints to: {usable_file}")

        # Print key findings
        print("\nKey Findings:")
        print(f"- {results['subjects_analyzed']} true subjects analyzed")
        print(
            f"- {results['subjects_affected']} subjects ({results['subjects_affected'] / results['subjects_analyzed'] * 100:.1f}%) affected by censoring")
        print(
            f"- {results['subjects_lost_completely']} subjects ({results['subjects_lost_completely'] / results['subjects_analyzed'] * 100:.1f}%) lost completely")
        print(f"- {results['timepoints_before'] - results['timepoints_after']} timepoints lost "
              f"({(results['timepoints_before'] - results['timepoints_after']) / results['timepoints_before'] * 100:.1f}%)")
        print(f"- {len(best_runs)} usable subject-timepoints with at least {args.min_runs} good runs")

        # Print phenotype-specific findings
        if phenotype_map:
            print("\nImpact by Phenotype:")
            for category in ['control', 'subthreshold', 'adhd', 'other', 'unknown']:
                count = results['phenotype_counts'][category]
                if count['total'] > 0:
                    print(f"- {category.upper()}: {count['affected']}/{count['total']} affected "
                          f"({count['affected'] / count['total'] * 100:.1f}%), {count['lost']} lost completely")

        # Return key output files
        return report_file, usable_file

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None


if __name__ == "__main__":
    main()