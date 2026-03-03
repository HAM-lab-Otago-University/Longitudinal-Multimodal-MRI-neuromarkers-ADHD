#!/usr/bin/env python3
"""
Demographics and Descriptive Statistics Script for OREGON1000 Final Dataset

This script generates comprehensive demographics and descriptive statistics for the final
participants retained in the stacked ridge model analysis, including:
- Phenotype distribution (subjects and timepoints)
- Sex distribution
- Race/ethnicity distribution
- Timepoints per subject statistics
- Breakdown by phenotype groups
"""

import pandas as pd
import numpy as np
import os
import logging
from collections import Counter
import traceback
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File paths
BASE_DATA_PATH = '/media/hcs-sci-psy-narun/Jack'
STACKED_PREDICTIONS_FILE = '/media/hcs-sci-psy-narun/Jack/First_level_model_aligned/Stacked_model_ridge_v4_EXTREME_conserve_a2000_keeper/stacked_ridge_predictions.csv'
NDAR_FILE = os.path.join(BASE_DATA_PATH, "ndar_subject01.csv")

# Output directory
OUTPUT_DIR = '/media/hcs-sci-psy-narun/Jack'

# Phenotype definitions
PHENOTYPE_LABELS = {
    1: "Control",
    2: "Subthreshold",
    3: "ADHD",
    4: "Other",
    5: "Not Assessed"
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

        return ndar_id, age_months

    except Exception as e:
        logger.warning(f"Error parsing subject_id {subject_id}: {e}")
        return None, None


def load_ndar_phenotypes_and_demographics(ndar_file):
    """
    Load NDAR subjects file and extract phenotype, sex, race, and ethnicity information.
    """
    logger.info(f"Loading phenotype and demographics data from {ndar_file}...")

    try:
        # Skip the subheader row (row 1, 0-indexed)
        ndar_df = pd.read_csv(ndar_file, skiprows=[1])
        logger.info(f"Loaded NDAR file with {len(ndar_df)} records and {len(ndar_df.columns)} columns")

        # Create comprehensive mapping from NDAR ID to all relevant data
        subject_data_map = {}

        for _, row in ndar_df.iterrows():
            # Extract subject key (standardize format by removing underscore after NDAR if present)
            subject_key = row['subjectkey'] if 'subjectkey' in row else row.get('ndar_subject01_id', '')
            if isinstance(subject_key, str) and subject_key.startswith('NDAR_'):
                subject_key = subject_key.replace('NDAR_', 'NDAR', 1)

            if not isinstance(subject_key, str) or not subject_key:
                continue

            # Strip underscores for matching (as done in previous scripts)
            standardized_key = subject_key.replace('_', '')

            # Get interview age
            interview_age = None
            if 'interview_age' in row and not pd.isna(row['interview_age']):
                try:
                    interview_age = int(float(row['interview_age']))
                except (ValueError, TypeError):
                    interview_age = None

            # Check phenotype column first
            phenotype = None
            if 'phenotype' in row and not pd.isna(row['phenotype']):
                try:
                    phenotype = int(float(row['phenotype']))
                except (ValueError, TypeError):
                    phenotype = None

            # If phenotype not found, check sibling_study column
            if phenotype is None and 'sibling_study' in row and not pd.isna(row['sibling_study']):
                try:
                    phenotype = int(float(row['sibling_study']))
                    if phenotype < 1 or phenotype > 5:  # Validate the value
                        phenotype = None
                except (ValueError, TypeError):
                    phenotype = None

            # Get sex
            sex = None
            if 'sex' in row and not pd.isna(row['sex']):
                sex = str(row['sex']).strip()
                if sex not in ['M', 'F']:
                    sex = 'Unknown'
            else:
                sex = 'Unknown'

            # Get race
            race = None
            if 'race' in row and not pd.isna(row['race']):
                race = str(row['race']).strip()
            else:
                race = 'Unknown or not reported'

            # Get ethnic group
            ethnic_group = None
            if 'ethnic_group' in row and not pd.isna(row['ethnic_group']):
                ethnic_group = str(row['ethnic_group']).strip()
            else:
                ethnic_group = 'Unknown or not reported'

            # Create combined ethnicity category
            combined_ethnicity = create_combined_ethnicity(race, ethnic_group)

            # Store all data for this subject-timepoint
            key = (standardized_key, interview_age)
            subject_data_map[key] = {
                'subject_key': standardized_key,
                'interview_age': interview_age,
                'phenotype': phenotype,
                'sex': sex,
                'race': race,
                'ethnic_group': ethnic_group,
                'combined_ethnicity': combined_ethnicity
            }

        logger.info(f"Found demographic information for {len(subject_data_map)} subject-timepoint combinations")
        return subject_data_map

    except Exception as e:
        logger.error(f"Error loading NDAR data: {e}")
        logger.error(traceback.format_exc())
        return {}


def create_combined_ethnicity(race, ethnic_group):
    """Create combined ethnicity categories as specified."""
    if pd.isna(race) or pd.isna(
            ethnic_group) or race == 'Unknown or not reported' or ethnic_group == 'Unknown or not reported':
        return 'Not specified'

    race = str(race).strip()
    ethnic_group = str(ethnic_group).strip()

    if 'White' in race:
        # Check for Non-Hispanic first to avoid the substring issue
        if 'Non-Hispanic' in ethnic_group:
            return 'White Non-Hispanic'
        elif 'Hispanic' in ethnic_group or 'Latino' in ethnic_group:
            return 'White Hispanic/Latino'
        else:
            return 'White (ethnicity not specified)'
    elif 'Black' in race or 'African American' in race:
        # Check for Non-Hispanic first to avoid the substring issue
        if 'Non-Hispanic' in ethnic_group:
            return 'Black or African American Non-Hispanic'
        elif 'Hispanic' in ethnic_group or 'Latino' in ethnic_group:
            return 'Black or African American Hispanic/Latino'
        else:
            return 'Black or African American (ethnicity not specified)'
    elif 'Asian' in race:
        if 'Non-Hispanic' in ethnic_group:
            return 'Asian Non-Hispanic'
        elif 'Hispanic' in ethnic_group or 'Latino' in ethnic_group:
            return 'Asian Hispanic/Latino'
        else:
            return 'Asian'
    elif 'Hawaiian' in race or 'Pacific Islander' in race:
        if 'Non-Hispanic' in ethnic_group:
            return 'Hawaiian/Pacific Islander Non-Hispanic'
        elif 'Hispanic' in ethnic_group or 'Latino' in ethnic_group:
            return 'Hawaiian/Pacific Islander Hispanic/Latino'
        else:
            return 'Hawaiian/Pacific Islander'
    elif 'American Indian' in race or 'Alaska Native' in race:
        if 'Non-Hispanic' in ethnic_group:
            return 'American Indian/Alaska Native Non-Hispanic'
        elif 'Hispanic' in ethnic_group or 'Latino' in ethnic_group:
            return 'American Indian/Alaska Native Hispanic/Latino'
        else:
            return 'American Indian/Alaska Native'
    elif 'More than one race' in race or 'More than One Race' in race:
        # Check for Non-Hispanic first to avoid the substring issue
        if 'Non-Hispanic' in ethnic_group:
            return 'More than One Race, Not Hispanic/Latino'
        elif 'Hispanic/Latino' in ethnic_group or 'Latino' in ethnic_group:
            return 'More than One Race, Hispanic/Latino'
        else:
            return 'More than One Race (ethnicity not specified)'
    else:
        return 'Other/Not specified'


def assign_subject_phenotype(subject_phenotypes):
    """
    Assign phenotype to a subject based on their timepoints.
    Apply the same logic as other scripts: prefer non-5 phenotypes, assign "other" if all are 5.
    """
    if not subject_phenotypes:
        return None

    # Remove None values
    valid_phenotypes = [p for p in subject_phenotypes if p is not None]
    if not valid_phenotypes:
        return None

    # If we have multiple phenotypes, prefer non-5 ones
    non_five_phenotypes = [p for p in valid_phenotypes if p != 5]
    if non_five_phenotypes:
        # Return the most common non-5 phenotype
        return Counter(non_five_phenotypes).most_common(1)[0][0]
    else:
        # If all phenotypes are 5, assign to "other" (4)
        return 4


def load_and_match_data():
    """Load stacked predictions and match with NDAR demographics."""
    logger.info("Loading stacked ridge predictions...")

    try:
        # Load stacked predictions
        stacked_df = pd.read_csv(STACKED_PREDICTIONS_FILE)
        logger.info(f"Loaded {len(stacked_df)} prediction records")

        # Load NDAR demographics
        ndar_data_map = load_ndar_phenotypes_and_demographics(NDAR_FILE)
        logger.info(f"NDAR data map contains {len(ndar_data_map)} subject-timepoint combinations")

        # Create a lookup for available ages per NDAR ID for debugging
        ndar_ages_lookup = {}
        for (ndar_id, age), data in ndar_data_map.items():
            if ndar_id not in ndar_ages_lookup:
                ndar_ages_lookup[ndar_id] = []
            ndar_ages_lookup[ndar_id].append(age)

        # Diagnostic counters
        parsing_failures = 0
        exact_matches = 0
        approximate_matches = 0
        no_matches = 0

        # Track detailed unmatched records
        unmatched_details = []

        # Extract NDAR IDs and ages from subject_id
        matched_records = []

        for idx, row in stacked_df.iterrows():
            subject_id = row['subject_id']
            ndar_id, age_months = extract_ndar_and_age(subject_id)

            # Check parsing
            if ndar_id is None or age_months is None:
                parsing_failures += 1
                continue

            # Try exact match first
            key = (ndar_id, age_months)
            if key in ndar_data_map:
                record = ndar_data_map[key].copy()
                record['subject_id'] = subject_id
                record['true_g'] = row['true_g'] if 'true_g' in row else None
                record['stacked_pred'] = row['stacked_pred'] if 'stacked_pred' in row else None
                matched_records.append(record)
                exact_matches += 1
            else:
                # Try approximate matching (±6 months)
                found_match = False
                for (stored_ndar, stored_age), data in ndar_data_map.items():
                    if stored_ndar == ndar_id and stored_age is not None:
                        # Allow for age differences (±6 months) since demographics are static
                        if abs(stored_age - age_months) <= 6:
                            record = data.copy()
                            record['subject_id'] = subject_id
                            record['true_g'] = row['true_g'] if 'true_g' in row else None
                            record['stacked_pred'] = row['stacked_pred'] if 'stacked_pred' in row else None
                            matched_records.append(record)
                            approximate_matches += 1
                            found_match = True
                            break

                if not found_match:
                    # Special handling for known edge cases - try to find closest age for this subject
                    if ndar_id in ndar_ages_lookup and len(ndar_ages_lookup[ndar_id]) > 0:
                        available_ages = ndar_ages_lookup[ndar_id]
                        # Find the closest age
                        closest_age = min(available_ages, key=lambda x: abs(x - age_months))
                        age_diff = abs(closest_age - age_months)

                        # If within reasonable range (±12 months), use it since demographics are static
                        if age_diff <= 12:
                            closest_key = (ndar_id, closest_age)
                            if closest_key in ndar_data_map:
                                record = ndar_data_map[closest_key].copy()
                                record['subject_id'] = subject_id
                                record['true_g'] = row['true_g'] if 'true_g' in row else None
                                record['stacked_pred'] = row['stacked_pred'] if 'stacked_pred' in row else None
                                matched_records.append(record)
                                approximate_matches += 1
                                found_match = True
                                logger.info(
                                    f"Special match: {subject_id} (age {age_months}) matched to closest available age {closest_age} (diff: {age_diff} months)")

                if not found_match:
                    no_matches += 1
                    # Check if the NDAR ID exists at all, and if so, what ages are available
                    if ndar_id in ndar_ages_lookup:
                        available_ages = sorted(ndar_ages_lookup[ndar_id])
                        unmatched_details.append({
                            'subject_id': subject_id,
                            'ndar_id': ndar_id,
                            'requested_age': age_months,
                            'available_ages': available_ages,
                            'reason': f"Age mismatch - requested {age_months}, available: {available_ages}"
                        })
                    else:
                        unmatched_details.append({
                            'subject_id': subject_id,
                            'ndar_id': ndar_id,
                            'requested_age': age_months,
                            'available_ages': [],
                            'reason': f"NDAR ID not found in demographics file"
                        })

        # Log detailed diagnostics
        total_processed = len(stacked_df)
        total_matched = exact_matches + approximate_matches
        total_unmatched = parsing_failures + no_matches

        logger.info(f"MATCHING DIAGNOSTICS:")
        logger.info(f"  Total records in stacked predictions: {total_processed}")
        logger.info(f"  Parsing failures: {parsing_failures}")
        logger.info(f"  Exact matches: {exact_matches}")
        logger.info(f"  Approximate matches (±6 months): {approximate_matches}")
        logger.info(f"  No matches found: {no_matches}")
        logger.info(f"  Total successfully matched: {total_matched}")
        logger.info(f"  Total unmatched: {total_unmatched}")
        logger.info(f"  Match rate: {(total_matched / total_processed) * 100:.1f}%")

        # Show detailed breakdown of unmatched records
        if unmatched_details:
            logger.info(f"\nDETAILED BREAKDOWN OF {len(unmatched_details)} UNMATCHED RECORDS:")

            # Group by reason
            id_not_found = [x for x in unmatched_details if not x['available_ages']]
            age_mismatch = [x for x in unmatched_details if x['available_ages']]

            logger.info(f"  NDAR IDs not found in demographics file: {len(id_not_found)}")
            if len(id_not_found) <= 20:  # Show all if reasonable number
                for item in id_not_found:
                    logger.info(f"    {item['subject_id']} -> {item['ndar_id']}")
            else:
                # Show first 20
                for item in id_not_found[:20]:
                    logger.info(f"    {item['subject_id']} -> {item['ndar_id']}")
                logger.info(f"    ... and {len(id_not_found) - 20} more")

            logger.info(f"  Age mismatches (NDAR ID exists but different ages): {len(age_mismatch)}")
            if len(age_mismatch) <= 20:  # Show all if reasonable number
                for item in age_mismatch:
                    logger.info(
                        f"    {item['subject_id']} -> requested age {item['requested_age']}, available: {item['available_ages']}")
            else:
                # Show first 20
                for item in age_mismatch[:20]:
                    logger.info(
                        f"    {item['subject_id']} -> requested age {item['requested_age']}, available: {item['available_ages']}")
                logger.info(f"    ... and {len(age_mismatch) - 20} more")

        # Check overall ID overlap
        prediction_ndar_ids = set()
        for _, row in stacked_df.iterrows():
            ndar_id, _ = extract_ndar_and_age(row['subject_id'])
            if ndar_id:
                prediction_ndar_ids.add(ndar_id)

        ndar_ids_in_map = set(key[0] for key in ndar_data_map.keys())
        ndar_only = ndar_ids_in_map - prediction_ndar_ids
        predictions_only = prediction_ndar_ids - ndar_ids_in_map

        logger.info(f"\nOVERALL ID COMPARISON:")
        logger.info(f"  Unique NDAR IDs in predictions: {len(prediction_ndar_ids)}")
        logger.info(f"  Unique NDAR IDs in demographics: {len(ndar_ids_in_map)}")
        logger.info(f"  NDAR IDs only in demographics file: {len(ndar_only)}")
        logger.info(f"  NDAR IDs only in predictions file: {len(predictions_only)}")

        if matched_records:
            matched_df = pd.DataFrame(matched_records)
            logger.info(f"\nFinal matched dataframe: {len(matched_df)} records")
            return matched_df
        else:
            logger.error("No records were successfully matched")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error in load_and_match_data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def calculate_subject_level_data(matched_df):
    """Calculate subject-level data with assigned phenotypes."""
    logger.info("Calculating subject-level data...")

    subject_data = {}

    for _, row in matched_df.iterrows():
        subject_key = row['subject_key']

        if subject_key not in subject_data:
            subject_data[subject_key] = {
                'timepoints': [],
                'phenotypes': [],
                'sex': row['sex'],  # Static per subject
                'combined_ethnicity': row['combined_ethnicity']  # Static per subject
            }

        subject_data[subject_key]['timepoints'].append(row['subject_id'])
        if row['phenotype'] is not None:
            subject_data[subject_key]['phenotypes'].append(row['phenotype'])

    # Assign final phenotypes to subjects
    subject_records = []
    for subject_key, data in subject_data.items():
        assigned_phenotype = assign_subject_phenotype(data['phenotypes'])

        subject_records.append({
            'subject_key': subject_key,
            'n_timepoints': len(data['timepoints']),
            'assigned_phenotype': assigned_phenotype,
            'sex': data['sex'],
            'combined_ethnicity': data['combined_ethnicity']
        })

    subject_df = pd.DataFrame(subject_records)
    logger.info(f"Created subject-level data for {len(subject_df)} unique subjects")

    return subject_df


def calculate_timepoint_statistics(matched_df):
    """Calculate statistics at the timepoint level."""
    stats = {}

    # Overall timepoint stats
    stats['total_timepoints'] = len(matched_df)
    stats['total_unique_subjects'] = matched_df['subject_key'].nunique()

    # Timepoints per subject
    timepoints_per_subject = matched_df.groupby('subject_key').size()
    stats['mean_timepoints_per_subject'] = timepoints_per_subject.mean()
    stats['median_timepoints_per_subject'] = timepoints_per_subject.median()
    stats['std_timepoints_per_subject'] = timepoints_per_subject.std()
    stats['sem_timepoints_per_subject'] = timepoints_per_subject.sem()
    stats['mode_timepoints_per_subject'] = timepoints_per_subject.mode().iloc[0] if len(
        timepoints_per_subject.mode()) > 0 else None
    stats['min_timepoints_per_subject'] = timepoints_per_subject.min()
    stats['max_timepoints_per_subject'] = timepoints_per_subject.max()

    # Timepoint distribution
    timepoint_counts = timepoints_per_subject.value_counts().sort_index()
    stats['timepoint_distribution'] = timepoint_counts.to_dict()

    # Phenotype distribution at timepoint level
    phenotype_timepoint_counts = matched_df['phenotype'].value_counts().sort_index()
    stats['phenotype_timepoint_distribution'] = phenotype_timepoint_counts.to_dict()

    return stats


def calculate_subject_statistics(subject_df):
    """Calculate statistics at the subject level."""
    stats = {}

    # Overall subject stats
    stats['total_subjects'] = len(subject_df)

    # Subject-level phenotype distribution
    phenotype_subject_counts = subject_df['assigned_phenotype'].value_counts().sort_index()
    stats['phenotype_subject_distribution'] = phenotype_subject_counts.to_dict()

    # Sex distribution (overall)
    sex_counts = subject_df['sex'].value_counts()
    stats['sex_distribution'] = sex_counts.to_dict()

    # Ethnicity distribution (overall)
    ethnicity_counts = subject_df['combined_ethnicity'].value_counts()
    stats['ethnicity_distribution'] = ethnicity_counts.to_dict()

    return stats


def calculate_demographics_by_phenotype(subject_df, matched_df):
    """Calculate demographics broken down by phenotype."""
    phenotype_stats = {}

    for phenotype in sorted(subject_df['assigned_phenotype'].dropna().unique()):
        if phenotype not in PHENOTYPE_LABELS:
            continue

        phenotype_label = PHENOTYPE_LABELS[phenotype]
        phenotype_subjects = subject_df[subject_df['assigned_phenotype'] == phenotype]
        phenotype_timepoints = matched_df[matched_df['phenotype'] == phenotype]

        stats = {
            'n_subjects': len(phenotype_subjects),
            'n_timepoints': len(phenotype_timepoints),
            'sex_distribution': phenotype_subjects['sex'].value_counts().to_dict(),
            'ethnicity_distribution': phenotype_subjects['combined_ethnicity'].value_counts().to_dict(),
        }

        # Timepoints per subject for this phenotype
        if len(phenotype_subjects) > 0:
            timepoints_per_subject = phenotype_subjects['n_timepoints']
            stats['mean_timepoints'] = timepoints_per_subject.mean()
            stats['median_timepoints'] = timepoints_per_subject.median()
            stats['std_timepoints'] = timepoints_per_subject.std()
            stats['sem_timepoints'] = timepoints_per_subject.sem()
            stats['mode_timepoints'] = timepoints_per_subject.mode().iloc[0] if len(
                timepoints_per_subject.mode()) > 0 else None

            # Debug: show distribution for this phenotype
            timepoint_dist = timepoints_per_subject.value_counts().sort_index().to_dict()
            stats['timepoint_distribution'] = timepoint_dist

        phenotype_stats[phenotype_label] = stats

    return phenotype_stats


def print_and_save_results(timepoint_stats, subject_stats, phenotype_stats, subject_df, output_dir):
    """Print results to console and save to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"demographics_descriptive_stats_{timestamp}.txt")

    # Prepare output text
    output_lines = []

    def add_line(line=""):
        print(line)
        output_lines.append(line)

    add_line("=" * 80)
    add_line("DEMOGRAPHICS AND DESCRIPTIVE STATISTICS")
    add_line("OREGON1000 Stacked Ridge Model Final Dataset")
    add_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add_line("=" * 80)

    # Overall statistics
    add_line("\n1. OVERALL DATASET STATISTICS")
    add_line("-" * 40)
    add_line(f"Total unique subjects: {subject_stats['total_subjects']}")
    add_line(f"Total timepoints: {timepoint_stats['total_timepoints']}")
    add_line()
    add_line("Timepoints per subject:")
    add_line(f"  Mean: {timepoint_stats['mean_timepoints_per_subject']:.2f}")
    add_line(f"  Median: {timepoint_stats['median_timepoints_per_subject']:.1f}")
    add_line(f"  Mode: {timepoint_stats['mode_timepoints_per_subject']}")
    add_line(f"  Standard Deviation: {timepoint_stats['std_timepoints_per_subject']:.2f}")
    add_line(f"  Standard Error: {timepoint_stats['sem_timepoints_per_subject']:.3f}")
    add_line(
        f"  Range: {timepoint_stats['min_timepoints_per_subject']} - {timepoint_stats['max_timepoints_per_subject']}")
    add_line()
    add_line("Distribution of timepoints per subject:")
    for n_timepoints, n_subjects in sorted(timepoint_stats['timepoint_distribution'].items()):
        percentage = (n_subjects / subject_stats['total_subjects']) * 100
        add_line(f"  {n_timepoints} timepoint(s): {n_subjects} subjects ({percentage:.1f}%)")

    # Phenotype distribution
    add_line("\n2. PHENOTYPE DISTRIBUTION")
    add_line("-" * 40)
    add_line("By unique subjects:")
    total_subjects_with_phenotype = sum(subject_stats['phenotype_subject_distribution'].values())
    for phenotype, count in sorted(subject_stats['phenotype_subject_distribution'].items()):
        if phenotype in PHENOTYPE_LABELS:
            percentage = (count / total_subjects_with_phenotype) * 100
            add_line(f"  {PHENOTYPE_LABELS[phenotype]}: {count} subjects ({percentage:.1f}%)")

    add_line()
    add_line("By timepoints:")
    total_timepoints_with_phenotype = sum(timepoint_stats['phenotype_timepoint_distribution'].values())
    for phenotype, count in sorted(timepoint_stats['phenotype_timepoint_distribution'].items()):
        if phenotype in PHENOTYPE_LABELS:
            percentage = (count / total_timepoints_with_phenotype) * 100
            add_line(f"  {PHENOTYPE_LABELS[phenotype]}: {count} timepoints ({percentage:.1f}%)")

    # Sex distribution
    add_line("\n3. SEX DISTRIBUTION (Overall)")
    add_line("-" * 40)
    total_subjects_with_sex = sum(subject_stats['sex_distribution'].values())
    for sex, count in subject_stats['sex_distribution'].items():
        percentage = (count / total_subjects_with_sex) * 100
        add_line(f"  {sex}: {count} subjects ({percentage:.1f}%)")

    # Ethnicity distribution
    add_line("\n4. ETHNICITY DISTRIBUTION (Overall)")
    add_line("-" * 40)
    total_subjects_with_ethnicity = sum(subject_stats['ethnicity_distribution'].values())
    for ethnicity, count in sorted(subject_stats['ethnicity_distribution'].items()):
        percentage = (count / total_subjects_with_ethnicity) * 100
        add_line(f"  {ethnicity}: {count} subjects ({percentage:.1f}%)")

    # Phenotype-specific demographics
    add_line("\n5. DEMOGRAPHICS BY PHENOTYPE")
    add_line("-" * 40)

    # Debug: Check for subjects without assigned phenotypes
    subjects_without_phenotype = subject_df[subject_df['assigned_phenotype'].isna()]
    if len(subjects_without_phenotype) > 0:
        add_line(f"\nDEBUG: Found {len(subjects_without_phenotype)} subjects without assigned phenotypes:")
        timepoints_no_phenotype = subjects_without_phenotype['n_timepoints']
        add_line(f"  These subjects contribute {timepoints_no_phenotype.sum()} timepoints")
        add_line(
            f"  Timepoint distribution for subjects without phenotype: {timepoints_no_phenotype.value_counts().sort_index().to_dict()}")
        add_line()

    for phenotype_label, stats in phenotype_stats.items():
        add_line(f"\n{phenotype_label.upper()}:")
        add_line(f"  Subjects: {stats['n_subjects']}")
        add_line(f"  Timepoints: {stats['n_timepoints']}")
        add_line(f"  Timepoints per subject:")
        add_line(
            f"    Mean: {stats['mean_timepoints']:.2f}, Median: {stats['median_timepoints']:.1f}, Mode: {stats['mode_timepoints']}")
        add_line(f"    Std Dev: {stats['std_timepoints']:.2f}, SEM: {stats['sem_timepoints']:.3f}")

        # Show distribution within this phenotype to debug median issue
        add_line("  Timepoint distribution within group:")
        for n_timepoints, n_subjects in sorted(stats['timepoint_distribution'].items()):
            percentage = (n_subjects / stats['n_subjects']) * 100
            add_line(f"    {n_timepoints} timepoint(s): {n_subjects} subjects ({percentage:.1f}%)")

        add_line("  Sex distribution:")
        total_sex = sum(stats['sex_distribution'].values())
        for sex, count in stats['sex_distribution'].items():
            percentage = (count / total_sex) * 100 if total_sex > 0 else 0
            add_line(f"    {sex}: {count} ({percentage:.1f}%)")

        add_line("  Ethnicity distribution:")
        total_ethnicity = sum(stats['ethnicity_distribution'].values())
        for ethnicity, count in sorted(stats['ethnicity_distribution'].items()):
            percentage = (count / total_ethnicity) * 100 if total_ethnicity > 0 else 0
            add_line(f"    {ethnicity}: {count} ({percentage:.1f}%)")

    # Save to file
    try:
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        add_line(f"\nResults saved to: {output_file}")
    except Exception as e:
        add_line(f"\nError saving results to file: {e}")


def main():
    """Main function to generate demographics and descriptive statistics."""
    try:
        logger.info("Starting demographics and descriptive statistics analysis...")

        # Load and match data
        matched_df = load_and_match_data()
        if matched_df.empty:
            logger.error("No matched data available. Exiting.")
            return

        # Calculate subject-level data
        subject_df = calculate_subject_level_data(matched_df)

        # Calculate statistics
        timepoint_stats = calculate_timepoint_statistics(matched_df)
        subject_stats = calculate_subject_statistics(subject_df)
        phenotype_stats = calculate_demographics_by_phenotype(subject_df, matched_df)

        # Print and save results
        print_and_save_results(timepoint_stats, subject_stats, phenotype_stats, subject_df, OUTPUT_DIR)

        logger.info("Analysis completed successfully!")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()