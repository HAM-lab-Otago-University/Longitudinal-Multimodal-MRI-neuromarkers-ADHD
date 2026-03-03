#!/usr/bin/env python3
"""
First-Level Model Data Preprocessing Script

This script organizes neuroimaging modality data for the OREGON1000 dataset into
appropriate fold structures for model training. It processes both structural and
functional modalities according to predefined fold assignments.

Outputs:
- Properly organized data files for each modality in the First_level_model directory
- Train/test splits for each modality matching the g-factor fold assignments

Author: Jack Scott
"""

import os
import sys
import pandas as pd
import numpy as np
import glob
import re
import logging
from pathlib import Path
from datetime import datetime
import shutil
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess_data_{}.log".format(datetime.now().strftime("%Y%m%d_%H%M%S"))),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("preprocessing")

# Define paths
BASE_DIR = "/media/hcs-sci-psy-narun/Jack/First_level_model"
STRUCT_DATA_DIR = "/media/hcs-sci-psy-narun/Jack/Oregon_Structural"
ALFF_REHO_DIR = "/media/hcs-sci-psy-narun/Jack/ALFF_ReHo_Analysis_v3_final_w_CSV"
FUNC_CONN_DIR = "/media/hcs-sci-psy-narun/Jack/Functional_connectivity_XCPD_Censor_May"
FOLD_ASSIGNMENTS_PATH = os.path.join(BASE_DIR, "nested_fold_assignments.csv")
# Define modalities
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

# List of subjects to exclude entirely (regardless of timepoint)
SUBJECT_EXCLUDE = [
    "sub-NDARINVKPGRKEZW", "sub-NDARINVCGJ4G85F", "sub-NDARINV0H2MNXAU", "sub-NDARINVE5F245ZC",
    "sub-NDARINVD0UHEXV5", "sub-NDARINVKKE5M36H", "sub-NDARINVDPJEMFNZ", "sub-NDARINV9U08EKVL",
    "sub-NDARINVEPTLP4ED", "sub-NDARINVT4FZHCTW", "sub-NDARINV4E1GLBYM", "sub-NDARINV8LZY8RBZ",
    "sub-NDARINVUYKLT0YE", "sub-NDARINVXF3RN4HD", "sub-NDARINV09MPGC1A", "sub-NDARINVPF66BBJZ",
    "sub-NDARINVZZVK3FXR", "sub-NDARINVT3V0N9JR", "sub-NDARINV9LMBWJ8Y", "sub-NDARINVKF5TZ56H",
    "sub-NDARINVRD9739F6", "sub-NDARINVXGH2TU13", "sub-NDARINVDGLFV9JE", "sub-NDARINV31PK79NV",
    "sub-NDARINV4AWRRTML", "sub-NDARINVKRWMZUZ0", "sub-NDARINVC597YE96", "sub-NDARINVT5HEHVWA",
    "sub-NDARINVM5GDJH10"
]

# List of specific subject-timepoint combinations to exclude
SUBJECT_TIMEPOINT_EXCLUDE = [
    "sub-NDARAB554KWNAge156MonthsStudyYear05",
    "sub-NDARBL381WXTAge186MonthsStudyYear05",
    "sub-NDARBL853AM1Age189MonthsStudyYear05",
    "sub-NDAREA400JB9Age136MonthsStudyYear05",
    "sub-NDARFG935JBWAge147MonthsStudyYear05",
    "sub-NDARHC389FNBAge164MonthsStudyYear05",
    "sub-NDARHM950VRNAge172MonthsStudyYear05",
    "sub-NDARINV0TMRP8W9Age110MonthsStudyYear02",
    "sub-NDARINV7P60663LAge89MonthsStudyYear01",
    "sub-NDARINVEMKF9W2FAge143MonthsStudyYear05",
    "sub-NDARINVFNCBBB3ZAge177MonthsStudyYear05",
    "sub-NDARINVVEXNDKF5Age87MonthsStudyYear01",
    "sub-NDARAN054EUFAge99MonthsStudyYear02",
    "sub-NDARBE279VUGAge109MonthsStudyYear02",
    "sub-NDARBF078JZHAge111MonthsStudyYear01",
    "sub-NDARBF078JZHAge123MonthsStudyYear02",
    "sub-NDAREH600YYDAge137MonthsStudyYear01",
    "sub-NDAREH600YYDAge161MonthsStudyYear03",
    "sub-NDARGW477DZUAge107MonthsStudyYear01",
    "sub-NDARHC389FNBAge141MonthsStudyYear03",
    "sub-NDARINV09MPGC1AAge116MonthsStudyYear01",
    "sub-NDARINV0H2MNXAUAge104MonthsStudyYear01",
    "sub-NDARINV0X7T7YPRAge126MonthsStudyYear01",
    "sub-NDARINV4FCNLAA6Age124MonthsStudyYear02",
    "sub-NDARINV5VML6URXAge125MonthsStudyYear02",
    "sub-NDARINV8R5NVBTHAge098MonthsStudyYear01",
    "sub-NDARINVELE7W2WVAge101MonthsStudyYear01",
    "sub-NDARINVHD6646BYAge132MonthsStudyYear01",
    "sub-NDARINVL9PBUMPVAge151MonthsStudyYear01",
    "sub-NDARINVM78JYC7GAge104MonthsStudyYear01",
    "sub-NDARINVXVR7YLZZAge119MonthsStudyYear01",
    "sub-NDARINV31PK79NVAge100MonthsStudyYear01",
    "sub-NDARINV3TRYNB4FAge124MonthsStudyYear03",
    "sub-NDARINV3ZN3A79XAge103MonthsStudyYear01",
    "sub-NDARINV5W4MYDL2Age104MonthsStudyYear01",
    "sub-NDARINV6KGN3D1CAge092MonthsStudyYear01",
    "sub-NDARINV7XANUXTMAge097MonthsStudyYear01",
    "sub-NDARINV8BP9WA01Age134MonthsStudyYear03",
    "sub-NDARINV8LZY8RBZAge098MonthsStudyYear01",
    "sub-NDARINV8UJ769XBAge132MonthsStudyYear03",
    "sub-NDARINV9D6YLNDCAge088MonthsStudyYear01",
    "sub-NDARINV9LMBWJ8YAge97MonthsStudyYear01",
    "sub-NDARINV9P3TB42ZAge093MonthsStudyYear01",
    "sub-NDARINV9U08EKVLAge108MonthsStudyYear01",
    "sub-NDARINV9U08EKVLAge120MonthsStudyYear02",
    "sub-NDARINV9U08EKVLAge135MonthsStudyYear03",
    "sub-NDARINV9VDT98JWAge120MonthsStudyYear01",
    "sub-NDARINVA1BEG2CLAge105MonthsStudyYear01",
    "sub-NDARINVA7LXRXJWAge124MonthsStudyYear02",
    "sub-NDARINVA7LXRXJWAge131MonthsStudyYear03",
    "sub-NDARINVBDZMWL9JAge118MonthsStudyYear01",
    "sub-NDARINVBDZMWL9JAge127MonthsStudyYear02",
    "sub-NDARINVBDZMWL9JAge139MonthsStudyYear03",
    "sub-NDARINVBT2NFBE5Age092MonthsStudyYear01",
    "sub-NDARINVCGJ4G85FAge101MonthsStudyYear01",
    "sub-NDARINVCGJ4G85FAge126MonthsStudyYear03",
    "sub-NDARINVD140CNXVAge131MonthsStudyYear02",
    "sub-NDARINVE5F245ZCAge103MonthsStudyYear01",
    "sub-NDARINVEPTLP4EDAge109MonthsStudyYear01",
    "sub-NDARINVFUJJ03CHAge108MonthsStudyYear02",
    "sub-NDARINVG19CZ4NUAge123MonthsStudyYear01",
    "sub-NDARINVHZCJVCZ9Age129MonthsStudyYear02",
    "sub-NDARINVHZCJVCZ9Age141MonthsStudyYear03",
    "sub-NDARINVJ2GPW7RHAge095MonthsStudyYear01",
    "sub-NDARINVJ8TJZ9BCAge088MonthsStudyYear01",
    "sub-NDARINVJ8TJZ9BCAge100MonthsStudyYear02",
    "sub-NDARINVKF5TZ56HAge101MonthsStudyYear01",
    "sub-NDARINVKM864ZFCAge110MonthsStudyYear02",
    "sub-NDARINVL50ABG4TAge145MonthsStudyYear02",
    "sub-NDARINVM41MAHX2Age135MonthsStudyYear01",
    "sub-NDARINVM5GDJH10Age110MonthsStudyYear01",
    "sub-NDARINVM5PGAF1WAge107MonthsStudyYear02",
    "sub-NDARINVNPHR7KZGAge151MonthsStudyYear06",
    "sub-NDARINVNZ8Y8Y9MAge102MonthsStudyYear01",
    "sub-NDARINVNZ8Y8Y9MAge126MonthsStudyYear03",
    "sub-NDARINVPF66BBJZAge112MonthsStudyYear01",
    "sub-NDARINVRD9739F6Age106MonthsStudyYear01",
    "sub-NDARINVT38Y2MU1Age096MonthsStudyYear01",
    "sub-NDARINVT3V0N9JRAge100MonthsStudyYear01",
    "sub-NDARINVT5HEHVWAAge108MonthsStudyYear01",
    "sub-NDARINVTXYCDHAWAge112MonthsStudyYear01",
    "sub-NDARINVUYKLT0YEAge097MonthsStudyYear01",
    "sub-NDARINVV4TFAYPZAge137MonthsStudyYear01",
    "sub-NDARINVVWT2EXNRAge125MonthsStudyYear01",
    "sub-NDARINVWU4B1BEJAge105MonthsStudyYear01",
    "sub-NDARINVXF3RN4HDAge118MonthsStudyYear01",
    "sub-NDARINVXGH2TU13Age114MonthsStudyYear01",
    "sub-NDARINVYHRR74C3Age095MonthsStudyYear01",
    "sub-NDARINVZBNJXU7MAge124MonthsStudyYear02",
    "sub-NDARLW133FWGAge097MonthsStudyYear01",
    "sub-NDARMA471CL4Age112MonthsStudyYear01",
    "sub-NDARPC423JTCAge147MonthsStudyYear01",
    "sub-NDARPJ373TU7Age102MonthsStudyYear02",
    "sub-NDARPJ373TU7Age114MonthsStudyYear03",
    "sub-NDARTD417EC4Age199MonthsStudyYear05",
    "sub-NDARTE304MDHAge100MonthsStudyYear01",
    "sub-NDARUE152ELBAge104MonthsStudyYear01",
    "sub-NDARUN588MBCAge105MonthsStudyYear01",
    "sub-NDARUN588MBCAge128MonthsStudyYear03",
    "sub-NDARWM302NLPAge211MonthsStudyYear06",
    "sub-NDARXK430GTQAge116MonthsStudyYear03",
    "sub-NDARYD918PBGAge120MonthsStudyYear02",
    "sub-NDARZN519WNHAge098MonthsStudyYear01",
    "sub-NDARZL911CDTAge158MonthsStudyYear05"
]

# track matched/missing timepoints
tracked_matches = {
    'structural': {modality: {'matched': set(), 'missing': set()} for modality in STRUCTURAL_MODALITIES},
    'functional': {modality: {'matched': set(), 'missing': set()} for modality in FUNCTIONAL_MODALITIES}
}

def setup_directory_structure():
    """Create the required directory structure for all modalities and folds."""
    # Create base directories for structural and functional modalities
    os.makedirs(os.path.join(BASE_DIR, "structural"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "functional"), exist_ok=True)

    # Create modality-specific directories
    for modality in STRUCTURAL_MODALITIES:
        modality_dir = os.path.join(BASE_DIR, "structural", modality)
        os.makedirs(modality_dir, exist_ok=True)

        # Create fold directories
        for fold in range(5):
            fold_dir = os.path.join(modality_dir, f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)

    for modality in FUNCTIONAL_MODALITIES:
        modality_dir = os.path.join(BASE_DIR, "functional", modality)
        os.makedirs(modality_dir, exist_ok=True)

        # Create fold directories
        for fold in range(5):
            fold_dir = os.path.join(modality_dir, f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)

    logger.info("Directory structure setup complete")


def normalize_subject_id(subject_id):
    """Extract just the NDAR identifier from any subject ID format."""
    if not subject_id:
        return None

    # Convert to string
    subject_id = str(subject_id)

    # Remove 'sub-' prefix if present
    if subject_id.startswith('sub-'):
        subject_id = subject_id[4:]

    # Try direct string manipulation first
    if "_Age" in subject_id:
        # Format: NDARXXXXX_Age123
        return subject_id.split("_Age")[0]

    if "Age" in subject_id:
        # Format: NDARXXXXXAge123MonthsStudyYear01
        return subject_id.split("Age")[0]

    # If we can't split by Age, use regex to extract the NDAR ID
    match = re.search(r'(NDAR[A-Za-z0-9]+)', subject_id)
    if match:
        return match.group(1)

    return None  # Return None if no pattern matches


def should_exclude_subject(subject_id):
    """
    Check if a subject should be excluded based on the exclusion list.

    Parameters:
    -----------
    subject_id : str
        Subject ID to check

    Returns:
    --------
    bool : True if subject should be excluded, False otherwise
    """
    normalized_id = normalize_subject_id(subject_id)

    # Normalize the exclude list to match
    normalized_exclude = [normalize_subject_id(s) for s in SUBJECT_EXCLUDE]

    return normalized_id in normalized_exclude


def should_exclude_timepoint(subject_timepoint_id):
    """
    Check if a specific subject-timepoint should be excluded.

    Parameters:
    -----------
    subject_timepoint_id : str
        Subject-timepoint ID to check (e.g., "sub-NDARXXXXAge123MonthsStudyYear01")

    Returns:
    --------
    bool : True if timepoint should be excluded, False otherwise
    """
    # First check if the subject is in the complete exclusion list
    if should_exclude_subject(subject_timepoint_id):
        return True

    # Normalize the timepoint format for comparison
    normalized_timepoint = subject_timepoint_id
    if not normalized_timepoint.startswith('sub-'):
        normalized_timepoint = f"sub-{normalized_timepoint}"

    # Direct match first
    if normalized_timepoint in SUBJECT_TIMEPOINT_EXCLUDE:
        return True

    # If no direct match, try matching based on core components
    subject_id = normalize_subject_id(subject_timepoint_id)

    # Extract age and study year if available
    age_match = re.search(r'Age(\d+)Months', subject_timepoint_id)
    study_match = re.search(r'StudyYear(\d+)', subject_timepoint_id)

    if not (age_match and study_match):
        return False

    age_months = age_match.group(1)
    study_year = study_match.group(1)

    # Check each excluded timepoint for a match on core components
    for exclude_entry in SUBJECT_TIMEPOINT_EXCLUDE:
        exclude_subject_id = normalize_subject_id(exclude_entry)

        if exclude_subject_id != subject_id:
            continue

        # Extract age and study year from exclude entry
        exclude_age_match = re.search(r'Age(\d+)Months', exclude_entry)
        exclude_study_match = re.search(r'StudyYear(\d+)', exclude_entry)

        if not (exclude_age_match and exclude_study_match):
            continue

        exclude_age = exclude_age_match.group(1)
        exclude_study = exclude_study_match.group(1)

        # Match if both age and study year match
        if age_months == exclude_age and study_year == exclude_study:
            return True

    return False


def extract_timepoint_info(timepoint_id):
    """
    CORRECTED: Extract age information from a timepoint ID.

    Age111 = 111 total months (not years+months)!

    Parameters:
    -----------
    timepoint_id : str
        Timepoint ID (e.g., "NDARXXXXAge111" or "NDARXXXX_Age111")

    Returns:
    --------
    tuple : (subject_id, age_str) where age_str is the extracted age
    """
    # First normalize to get subject ID
    subject_id = normalize_subject_id(timepoint_id)

    # Extract age - this is now MUCH simpler!
    age_match = re.search(r'Age(\d+)', timepoint_id)
    if age_match:
        age_str = age_match.group(1)  # Just the number, no conversion needed!
        return subject_id, age_str

    # If no age pattern matched
    return subject_id, None


def extract_subject_and_age(timepoint_id):
    """
    CORRECTED: Extract subject ID and age from timepoint ID.

    Key fix: Age111 means 111 total months!
    """
    subject_id = normalize_subject_id(timepoint_id)

    # Extract age - this is now MUCH simpler
    age_match = re.search(r'Age(\d+)', timepoint_id)
    age_str = age_match.group(1) if age_match else None

    return subject_id, age_str


def build_subject_timepoint_map(g_factor_timepoints):
    """
    Build a mapping from subject ID to available timepoints in g-factor data.

    Parameters:
    -----------
    g_factor_timepoints : dict
        Dictionary of g-factor timepoints by fold and split

    Returns:
    --------
    dict : {subject_id: {fold: {split: [timepoint_ids]}}}
    """
    subject_map = {}

    for fold in range(5):
        for split in ['train', 'test']:
            for timepoint_id in g_factor_timepoints[fold][split]:
                subject_id, _ = extract_subject_and_age(timepoint_id)

                if subject_id:
                    if subject_id not in subject_map:
                        subject_map[subject_id] = {}

                    if fold not in subject_map[subject_id]:
                        subject_map[subject_id][fold] = {}

                    if split not in subject_map[subject_id][fold]:
                        subject_map[subject_id][fold][split] = []

                    subject_map[subject_id][fold][split].append(timepoint_id)

    return subject_map


def find_best_timepoint_match(subject_id, age_str, subject_g_timepoints, logger):
    """
    CORRECTED: Find best match with proper age interpretation.
    """
    if not subject_g_timepoints:
        return None

    if len(subject_g_timepoints) == 1:
        return subject_g_timepoints[0]

    if not age_str or not age_str.isdigit():
        return subject_g_timepoints[0]

    # Simple numeric comparison (both are now in months)
    modality_age = int(age_str)

    best_match = None
    best_diff = float('inf')

    for gf_timepoint_id in subject_g_timepoints:
        # Extract age from g-factor timepoint (NDARXXX_Age111 format)
        gf_age_match = re.search(r'Age(\d+)', gf_timepoint_id)
        if gf_age_match:
            gf_age = int(gf_age_match.group(1))
            age_diff = abs(modality_age - gf_age)

            if age_diff < best_diff:
                best_diff = age_diff
                best_match = gf_timepoint_id

                if age_diff == 0:  # Exact match
                    break

    return best_match if best_match else subject_g_timepoints[0]


def create_subject_baseline():
    """
    Process functional connectivity data first to determine the baseline set of subjects.
    Returns a set of valid subject IDs and a dictionary mapping subject IDs to their fold assignments.
    """
    logger.info("Determining baseline subjects from functional connectivity data...")

    # Find all subject directories in the functional connectivity data
    subject_dirs = glob.glob(os.path.join(FUNC_CONN_DIR, "sub-*"))
    logger.info(f"Found {len(subject_dirs)} functional connectivity subject directories")

    # Create a set of valid subject IDs
    valid_subjects = set()

    # Process each subject directory
    for subject_dir in subject_dirs:
        basename = os.path.basename(subject_dir)
        subject_timepoint_id = basename[4:] if basename.startswith("sub-") else basename

        # Skip excluded subjects
        if should_exclude_subject(subject_timepoint_id) or should_exclude_timepoint(subject_timepoint_id):
            continue

        # Get the normalized subject ID
        subject_id = normalize_subject_id(subject_timepoint_id)
        if subject_id:
            valid_subjects.add(subject_id)

    logger.info(f"Found {len(valid_subjects)} unique subjects with functional connectivity data")

    # Filter fold assignments to only include valid subjects
    valid_fold_assignments = {}
    original_assignments_df = pd.read_csv(FOLD_ASSIGNMENTS_PATH)

    for _, row in original_assignments_df.iterrows():
        subject_id = normalize_subject_id(row['subject_id'])
        fold = int(row['outer_fold'])
        split = row['inner_split']

        if subject_id in valid_subjects:
            valid_fold_assignments[subject_id] = (fold, split)

    logger.info(f"Filtered fold assignments to {len(valid_fold_assignments)} subjects")

    # Write updated fold assignments to a new file
    updated_fold_df = pd.DataFrame([
        {'subject_id': subj, 'outer_fold': fold, 'inner_split': split, 'fold_assignment': f"fold_{fold}_{split}"}
        for subj, (fold, split) in valid_fold_assignments.items()
    ])

    updated_fold_path = os.path.join(BASE_DIR, "fold_assignments_filtered.csv")
    updated_fold_df.to_csv(updated_fold_path, index=False)
    logger.info(f"Saved filtered fold assignments to {updated_fold_path}")

    return valid_subjects, valid_fold_assignments, updated_fold_path


def convert_connectivity_to_gscore_format(connectivity_filename):
    """
    Convert a connectivity filename to g-score ID format.

    From: NDARXXXXAge111MonthsStudyYear01_averaged_combined_connectivity.csv
    To:   NDARXXXX_Age111

    Parameters:
    -----------
    connectivity_filename : str
        Original connectivity filename

    Returns:
    --------
    str : Equivalent ID in g-score format
    """
    # Extract the subject ID part (before "Age")
    match = re.search(r'(NDAR[A-Za-z0-9]+)Age', connectivity_filename)
    if not match:
        return None

    subject_id = match.group(1)

    # Extract the age part
    age_match = re.search(r'Age(\d+)Months', connectivity_filename)
    if not age_match:
        return None

    age_str = age_match.group(1)

    # Combine in g-score format
    return f"{subject_id}_Age{age_str}"

def create_master_timepoint_list(subject_to_fold):
    """
    Create a definitive master list of timepoints from functional connectivity data.
    Enhanced to handle 1-to-1 mapping between connectivity and g-factor timepoints.
    """
    logger.info("Creating master timepoint list from functional connectivity data...")

    # Process functional connectivity first
    functional_timepoints = {fold: {'train': [], 'test': []} for fold in range(5)}  # Use lists instead of sets

    # Find all subject directories
    subject_dirs = glob.glob(os.path.join(FUNC_CONN_DIR, "sub-*"))
    logger.info(f"Found {len(subject_dirs)} subject directories")

    # Track processed subjects and their timepoints
    subject_timepoints = defaultdict(list)  # Maps subject_id to list of timepoint_ids

    # Tracking variables
    exclusion_reasons = defaultdict(int)
    conn_timepoints_by_g_timepoint = defaultdict(list)  # To track duplicates

    # First pass: gather all timepoints for each subject
    for subject_dir in subject_dirs:
        basename = os.path.basename(subject_dir)
        subject_timepoint_id = basename[4:] if basename.startswith("sub-") else basename

        # Skip excluded subjects/timepoints
        if should_exclude_subject(subject_timepoint_id):
            exclusion_reasons["excluded_subject"] += 1
            continue

        if should_exclude_timepoint(subject_timepoint_id):
            exclusion_reasons["excluded_timepoint"] += 1
            continue

        # Extract subject ID and age
        subject_id, age_months = extract_subject_and_age(subject_timepoint_id)

        if not subject_id:
            exclusion_reasons["invalid_subject_id"] += 1
            continue

        # Find the connectivity CSV file
        csv_files = glob.glob(os.path.join(subject_dir, "*_averaged_combined_connectivity.csv"))
        if not csv_files:
            exclusion_reasons["no_connectivity_file"] += 1
            continue

        conn_file = csv_files[0]

        # Extract additional info from timepoint ID for better matching
        age_info = None
        study_year = None

        # Try to extract more detailed age/study year info
        age_match = re.search(r'Age(\d+)Months', subject_timepoint_id)
        study_match = re.search(r'StudyYear(\d+)', subject_timepoint_id)

        if age_match:
            age_info = age_match.group(1)

        if study_match:
            study_year = study_match.group(1)

        # Add to the list of timepoints for this subject
        subject_timepoints[subject_id].append({
            'subject_timepoint_id': subject_timepoint_id,
            'age': age_months,
            'age_info': age_info,
            'study_year': study_year,
            'conn_file': conn_file,
            'full_path': subject_dir
        })

    # Track how many subjects have multiple timepoints
    subjects_with_multiple = sum(1 for subj, tps in subject_timepoints.items() if len(tps) > 1)
    logger.info(f"Found {len(subject_timepoints)} subjects with connectivity data")
    logger.info(f"{subjects_with_multiple} subjects have multiple timepoints")

    # Second pass: match timepoints to g-factor data and assign to folds
    total_timepoints_processed = 0
    timepoints_matched = 0
    g_timepoints_used = set()

    for subject_id, timepoints in subject_timepoints.items():
        # Skip subjects without fold assignments
        if subject_id not in subject_to_fold:
            timepoints_without_fold = len(timepoints)
            exclusion_reasons["no_fold_assignment"] += timepoints_without_fold
            continue

        fold, split = subject_to_fold[subject_id]

        # Load g-factor data for this subject
        g_factor_path = os.path.join(BASE_DIR, f"g_factor/fold_{fold}/g/g_{split}_regularized_{fold}.csv")
        if not os.path.exists(g_factor_path):
            exclusion_reasons["missing_gfactor_file"] += len(timepoints)
            continue

        g_factor_df = pd.read_csv(g_factor_path)

        # Extract timepoint IDs and create detailed info for matching
        g_factor_timepoints = []
        for _, row in g_factor_df.iterrows():
            if 'eid' not in row:
                continue

            timepoint_id = row['eid']
            g_subject_id = normalize_subject_id(timepoint_id)

            if g_subject_id != subject_id:
                continue

            # Extract age info from g-factor timepoint
            g_age_match = re.search(r'_Age(\d+)', timepoint_id)
            g_age = g_age_match.group(1) if g_age_match else None

            g_factor_timepoints.append({
                'timepoint_id': timepoint_id,
                'subject_id': g_subject_id,
                'age': g_age
            })

        if not g_factor_timepoints:
            exclusion_reasons["subject_not_in_gfactor"] += len(timepoints)
            continue

        # Match each connectivity timepoint to g-factor timepoints
        for timepoint_info in timepoints:
            total_timepoints_processed += 1
            conn_timepoint_id = timepoint_info['subject_timepoint_id']
            conn_age = timepoint_info['age']
            conn_file = timepoint_info['conn_file']

            # Additional matching criteria
            age_info = timepoint_info['age_info']
            study_year = timepoint_info['study_year']

            best_match = None
            best_score = -1

            # Loop through g-factor timepoints to find best match
            for g_timepoint in g_factor_timepoints:
                g_id = g_timepoint['timepoint_id']
                g_age = g_timepoint['age']

                # Skip if this g-factor timepoint is already assigned
                # Comment this line out if you want to allow multiple matches
                # if g_id in g_timepoints_used:
                #     continue

                # Calculate match score (higher is better)
                score = 0

                # Exact subject ID match gives base score
                score += 100

                # Age match bonus
                if conn_age and g_age and conn_age == g_age:
                    score += 50  # Perfect age match
                elif conn_age and g_age:
                    # Partial age match - check first digits
                    common_prefix_len = 0
                    for i in range(min(len(conn_age), len(g_age))):
                        if conn_age[i] == g_age[i]:
                            common_prefix_len += 1
                        else:
                            break

                    score += common_prefix_len * 10  # 10 points per matching digit

                # Update best match if better score found
                if score > best_score:
                    best_score = score
                    best_match = g_id

            # If found a match, add to timepoints
            if best_match:
                timepoints_matched += 1
                g_timepoints_used.add(best_match)

                # Store the connectivity timepoint that matched to this g-factor timepoint
                conn_timepoints_by_g_timepoint[best_match].append({
                    'conn_timepoint_id': conn_timepoint_id,
                    'match_score': best_score,
                    'conn_file': conn_file,
                    'full_path': timepoint_info['full_path']
                })

                # Add to the appropriate fold list (not a set anymore)
                # We're keeping them as lists for now to handle potential duplicates
                functional_timepoints[fold][split].append({
                    'g_timepoint_id': best_match,
                    'conn_timepoint_id': conn_timepoint_id,
                    'conn_file': conn_file,
                    'match_score': best_score
                })
            else:
                exclusion_reasons["no_gfactor_match"] += 1

    # Analyze the mapping
    duplicates = {g_id: conn_list for g_id, conn_list in conn_timepoints_by_g_timepoint.items() if len(conn_list) > 1}
    unique_g_timepoints = len(conn_timepoints_by_g_timepoint)

    logger.info(f"Processed {total_timepoints_processed} total timepoints")
    logger.info(f"Matched {timepoints_matched} timepoints to g-factor data")
    logger.info(f"Unique g-factor timepoints matched: {unique_g_timepoints}")
    logger.info(f"Found {len(duplicates)} g-factor timepoints with multiple matches")

    # If duplicates exist, log some examples
    if duplicates:
        logger.info("\nExamples of duplicate matches:")
        example_count = 0
        for g_id, conn_list in duplicates.items():
            if example_count >= 5:  # Show only 5 examples
                break

            logger.info(f"  G-factor timepoint {g_id} matched to {len(conn_list)} connectivity timepoints:")
            for i, conn_info in enumerate(conn_list):
                logger.info(f"    {i + 1}. {conn_info['conn_timepoint_id']} (score: {conn_info['match_score']})")

            example_count += 1

    # Now we need to decide how to handle duplicates
    # Option 1: Keep only the best match for each g-factor timepoint
    # Option 2: Create unique identifiers for duplicates

    # For now, let's go with Option 1: Keep only the best match
    valid_timepoints = {}

    for g_id, conn_list in conn_timepoints_by_g_timepoint.items():
        # Sort by match score (highest first)
        sorted_matches = sorted(conn_list, key=lambda x: x['match_score'], reverse=True)
        best_match = sorted_matches[0]

        # Find which fold/split this belongs to
        for fold in range(5):
            for split in ['train', 'test']:
                matches = [item for item in functional_timepoints[fold][split]
                           if item['g_timepoint_id'] == g_id]

                if matches:
                    # Use the best match's connectivity file
                    valid_timepoints[g_id] = (fold, split, {
                        'subject_id': normalize_subject_id(g_id),
                        'original_file': best_match['conn_file']
                    })
                    break

    logger.info("\n===== TIMEPOINT PROCESSING BREAKDOWN =====")
    logger.info(f"Total connectivity timepoints processed: {total_timepoints_processed}")
    logger.info(f"Timepoints matched to g-factor data: {timepoints_matched}")
    logger.info(f"Unique g-factor timepoints matched: {unique_g_timepoints}")
    logger.info(f"Final valid timepoints: {len(valid_timepoints)}")

    logger.info("\nExclusion reasons:")
    for reason, count in sorted(exclusion_reasons.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {reason}: {count} timepoints")

    # If there are multiple matches, outline the solution
    if timepoints_matched > unique_g_timepoints:
        logger.info(
            f"\nRESOLUTION: {timepoints_matched - unique_g_timepoints} duplicate matches resolved by keeping only the best match for each g-factor timepoint")

        # Calculate how many duplicates we kept vs. discarded
        kept = len(valid_timepoints)
        discarded = timepoints_matched - kept
        logger.info(f"Kept {kept} matches, discarded {discarded} lower-quality duplicate matches")

    logger.info(f"Created master list with {len(valid_timepoints)} valid timepoints from functional connectivity")
    return valid_timepoints


def load_nested_fold_assignments_complete():
    """
    Load complete nested CV assignments where each subject appears in all 5 outer folds.

    Returns:
    --------
    dict : {subject_id: {outer_fold: 'train'/'test'}}
    """
    try:
        df = pd.read_csv(FOLD_ASSIGNMENTS_PATH)
        logger.info(f"Loaded nested fold assignments: {len(df)} entries")

        # Group by subject to see their assignments across all outer folds
        assignments_by_subject = {}

        for _, row in df.iterrows():
            subject_id = normalize_subject_id(row['subject_id'])
            outer_fold = int(row['outer_fold'])
            inner_split = row['inner_split']

            if subject_id not in assignments_by_subject:
                assignments_by_subject[subject_id] = {}

            assignments_by_subject[subject_id][outer_fold] = inner_split

        logger.info(f"Loaded assignments for {len(assignments_by_subject)} subjects")

        # Validate that each subject appears in all 5 outer folds
        complete_subjects = 0
        for subject_id, folds in assignments_by_subject.items():
            if len(folds) == 5:
                complete_subjects += 1
            else:
                logger.warning(f"Subject {subject_id} missing fold assignments: has {len(folds)}, needs 5")

        logger.info(f"Found {complete_subjects} subjects with complete fold assignments")
        return assignments_by_subject

    except Exception as e:
        logger.error(f"Error loading nested fold assignments: {e}")
        return {}


def load_g_factor_data(subject_filter=None):
    """
    Load the g-factor data for each fold to get the timepoint IDs.
    Optionally filter to only include subjects in subject_filter.

    Parameters:
    -----------
    subject_filter : set, optional
        Set of normalized subject IDs to include. If None, all subjects are included.

    Returns:
    --------
    dict : Dictionary mapping fold to train and test timepoint sets
    """
    g_factor_timepoints = {}

    for fold in range(5):
        g_factor_timepoints[fold] = {'train': set(), 'test': set()}

        # Load train data
        train_path = os.path.join(BASE_DIR, f"g_factor/fold_{fold}/g/g_train_regularized_{fold}.csv")
        if os.path.exists(train_path):
            train_df = pd.read_csv(train_path)
            # Extract timepoint IDs and add to set if subject is in filter
            for eid in train_df['eid']:
                subject_id = normalize_subject_id(eid)
                # Include if there's no filter or if subject is in filter
                if subject_filter is None or subject_id in subject_filter:
                    g_factor_timepoints[fold]['train'].add(eid)

            logger.info(f"Fold {fold} train: {len(g_factor_timepoints[fold]['train'])} timepoints" +
                        f" (filtered from {len(train_df)} original)" if subject_filter else "")

        # Load test data
        test_path = os.path.join(BASE_DIR, f"g_factor/fold_{fold}/g/g_test_regularized_{fold}.csv")
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            # Extract timepoint IDs and add to set if subject is in filter
            for eid in test_df['eid']:
                subject_id = normalize_subject_id(eid)
                # Include if there's no filter or if subject is in filter
                if subject_filter is None or subject_id in subject_filter:
                    g_factor_timepoints[fold]['test'].add(eid)

            logger.info(f"Fold {fold} test: {len(g_factor_timepoints[fold]['test'])} timepoints" +
                        f" (filtered from {len(test_df)} original)" if subject_filter else "")

    return g_factor_timepoints


def filter_g_factor_data_from_processed():
    """
    Filter g-factor data to include only subjects that are in final processed modality files.
    """
    logger.info("Filtering g-factor data based on processed modality files...")

    # First, determine what subjects actually made it into the final files
    # We'll use cortical_area as the reference modality
    final_subjects = set()

    # Process each fold
    for fold in range(5):
        for split in ['train', 'test']:
            # Path to cortical_area file
            ref_path = os.path.join(BASE_DIR, "structural/cortical_area", f"fold_{fold}", f"{split}.csv")

            if os.path.exists(ref_path):
                try:
                    df = pd.read_csv(ref_path)
                    for eid in df['eid']:
                        subject_id = normalize_subject_id(eid)
                        if subject_id:
                            final_subjects.add(subject_id)
                except Exception as e:
                    logger.error(f"Error reading reference file {ref_path}: {e}")

    logger.info(f"Found {len(final_subjects)} subjects in final processed files")
    logger.info(f"Sample subjects: {list(final_subjects)[:5]}")

    # Now filter g-factor to match these subjects
    for fold in range(5):
        fold_dir = os.path.join(BASE_DIR, f"g_factor/fold_{fold}/g")
        os.makedirs(fold_dir, exist_ok=True)

        # Process train and test files
        for split in ['train', 'test']:
            # Path to original file
            original_path = os.path.join(fold_dir, f"g_{split}_regularized_{fold}.csv")

            if not os.path.exists(original_path):
                logger.warning(f"Original g-factor file not found: {original_path}")
                continue

            # Load original data
            try:
                df = pd.read_csv(original_path)
                original_count = len(df)

                # Create backup of original file
                backup_path = os.path.join(fold_dir, f"g_{split}_regularized_{fold}_original.csv")
                if not os.path.exists(backup_path):
                    shutil.copy2(original_path, backup_path)
                    logger.info(f"Created backup of original g-factor file: {backup_path}")

                # Filter to keep only subjects in final processed files
                filtered_rows = []
                excluded_count = 0
                for _, row in df.iterrows():
                    eid = row['eid']
                    subject_id = normalize_subject_id(eid)
                    if subject_id in final_subjects:
                        filtered_rows.append(row)
                    else:
                        excluded_count += 1
                        if excluded_count <= 5:  # Log first 5 exclusions
                            logger.info(f"Excluding {eid} (normalized: {subject_id}) from g-factor data")

                # Create filtered DataFrame
                filtered_df = pd.DataFrame(filtered_rows)
                filtered_count = len(filtered_df)

                # Save filtered file
                filtered_df.to_csv(original_path, index=False)
                logger.info(
                    f"Filtered g-factor {split} fold {fold}: {original_count} → {filtered_count} subjects (excluded {excluded_count})")

            except Exception as e:
                logger.error(f"Error filtering g-factor file {original_path}: {e}")
                import traceback
                logger.error(traceback.format_exc())

    logger.info("G-factor filtering complete")


def process_structural_data(valid_timepoints=None):
    """
    Process structural modality data for nested CV structure.
    Each subject appears in all 5 outer folds with appropriate train/test assignments.
    """
    logger.info("Processing structural modality data for nested CV...")

    # Load complete fold assignments
    subject_assignments = load_nested_fold_assignments_complete()

    for modality in STRUCTURAL_MODALITIES:
        input_path = os.path.join(STRUCT_DATA_DIR, f"{modality}.csv")

        if not os.path.exists(input_path):
            logger.warning(f"Structural file not found: {input_path}")
            continue

        logger.info(f"Processing {modality} data from {input_path}")

        # Load the structural data
        df = pd.read_csv(input_path)
        subject_col = df.columns[0]

        # Initialize fold data structures
        fold_data = {fold: {'train': [], 'test': []} for fold in range(5)}

        # Track processing statistics
        processed_count = 0
        no_assignment_count = 0

        # Process each row in the structural data
        for _, row in df.iterrows():
            timepoint_id = str(row[subject_col])
            subject_id, age_str = extract_subject_and_age(timepoint_id)

            if not subject_id:
                continue

            # Check if subject has fold assignments
            if subject_id not in subject_assignments:
                no_assignment_count += 1
                continue

            # Create g-factor style timepoint ID
            if age_str and age_str.isdigit():
                gfactor_style_id = f"{subject_id}_Age{age_str}"
            else:
                gfactor_style_id = f"{subject_id}_AgeUnknown"

            # Add this timepoint to ALL 5 outer folds with appropriate train/test assignment
            for outer_fold in range(5):
                if outer_fold in subject_assignments[subject_id]:
                    split = subject_assignments[subject_id][outer_fold]  # 'train' or 'test'

                    # Create row dictionary with g-factor style ID
                    row_dict = row.to_dict()
                    row_dict['eid'] = gfactor_style_id

                    fold_data[outer_fold][split].append(row_dict)

            processed_count += 1

        logger.info(f"Processed {processed_count} {modality} timepoints")
        if no_assignment_count > 0:
            logger.info(f"No fold assignment: {no_assignment_count} timepoints")

        # Save data for each fold/split
        for fold in range(5):
            for split in ['train', 'test']:
                if fold_data[fold][split]:
                    # Create output directory
                    output_dir = os.path.join(BASE_DIR, "structural", modality, f"fold_{fold}")
                    os.makedirs(output_dir, exist_ok=True)

                    # Create DataFrame
                    split_df = pd.DataFrame(fold_data[fold][split])

                    # Ensure 'eid' is the first column
                    cols = ['eid'] + [c for c in split_df.columns if c != 'eid']
                    split_df = split_df[cols]

                    # Save to CSV
                    output_path = os.path.join(output_dir, f"{split}.csv")
                    split_df.to_csv(output_path, index=False)
                    logger.info(f"Saved {len(split_df)} {modality} entries to fold {fold} {split}")


def process_alff_reho_data(valid_timepoints=None):
    """
    Process ALFF and ReHo data for nested CV structure.
    Each subject appears in all 5 outer folds with appropriate train/test assignments.
    """
    logger.info("Processing ALFF and ReHo data for nested CV...")

    # Load complete fold assignments
    subject_assignments = load_nested_fold_assignments_complete()

    for modality in ['alff', 'reho']:
        input_path = os.path.join(ALFF_REHO_DIR, f"{modality}_aggregated_data_ml.csv")

        if not os.path.exists(input_path):
            logger.warning(f"ALFF/ReHo file not found: {input_path}")
            continue

        logger.info(f"Processing {modality} data from {input_path}")

        # Load the data
        df = pd.read_csv(input_path)
        subject_col = df.columns[0]  # Assume first column is subject ID

        # Initialize fold data structures
        fold_data = {fold: {'train': [], 'test': []} for fold in range(5)}

        # Track processing statistics
        processed_count = 0
        excluded_subject_count = 0
        excluded_timepoint_count = 0
        no_assignment_count = 0

        # Process each row in the data
        for _, row in df.iterrows():
            timepoint_id = str(row[subject_col])

            # Skip excluded subjects/timepoints
            if should_exclude_subject(timepoint_id):
                excluded_subject_count += 1
                continue

            if should_exclude_timepoint(timepoint_id):
                excluded_timepoint_count += 1
                continue

            # Extract subject ID and age
            subject_id, age_str = extract_subject_and_age(timepoint_id)

            if not subject_id:
                continue

            # Check if subject has fold assignments
            if subject_id not in subject_assignments:
                no_assignment_count += 1
                continue

            # Create g-factor style timepoint ID
            if age_str and age_str.isdigit():
                gfactor_style_id = f"{subject_id}_Age{age_str}"
            else:
                gfactor_style_id = f"{subject_id}_AgeUnknown"

            # Add this timepoint to ALL 5 outer folds with appropriate train/test assignment
            for outer_fold in range(5):
                if outer_fold in subject_assignments[subject_id]:
                    split = subject_assignments[subject_id][outer_fold]  # 'train' or 'test'

                    # Create row dictionary with g-factor style ID
                    row_dict = row.to_dict()
                    row_dict['eid'] = gfactor_style_id

                    fold_data[outer_fold][split].append(row_dict)

            processed_count += 1

        logger.info(f"Processed {processed_count} {modality} timepoints")
        logger.info(f"Excluded: {excluded_subject_count} subjects, {excluded_timepoint_count} timepoints")
        if no_assignment_count > 0:
            logger.info(f"No fold assignment: {no_assignment_count} timepoints")

        # Save data for each fold/split
        for fold in range(5):
            for split in ['train', 'test']:
                if fold_data[fold][split]:
                    # Create output directory
                    output_dir = os.path.join(BASE_DIR, "functional", modality, f"fold_{fold}")
                    os.makedirs(output_dir, exist_ok=True)

                    # Create DataFrame
                    split_df = pd.DataFrame(fold_data[fold][split])

                    # Ensure 'eid' is the first column
                    cols = ['eid'] + [c for c in split_df.columns if c != 'eid']
                    split_df = split_df[cols]

                    # Save to CSV
                    output_path = os.path.join(output_dir, f"{split}.csv")
                    split_df.to_csv(output_path, index=False)
                    logger.info(f"Saved {len(split_df)} {modality} entries to fold {fold} {split}")


def process_functional_connectivity(valid_timepoints=None):
    """
    Process functional connectivity data for nested CV structure.
    Each subject appears in all 5 outer folds with appropriate train/test assignments.
    """
    logger.info("Processing functional connectivity data for nested CV...")

    # Load complete fold assignments
    subject_assignments = load_nested_fold_assignments_complete()

    # Initialize fold data structures
    fold_data = {fold: {'train': [], 'test': []} for fold in range(5)}

    # Find all functional connectivity directories
    func_conn_dirs = glob.glob(os.path.join(FUNC_CONN_DIR, "sub-*"))
    logger.info(f"Found {len(func_conn_dirs)} functional connectivity directories")

    # Track processing statistics
    processed_count = 0
    excluded_subject_count = 0
    excluded_timepoint_count = 0
    no_assignment_count = 0
    no_file_count = 0

    for conn_dir in func_conn_dirs:
        # Extract timepoint ID from directory name
        dir_name = os.path.basename(conn_dir)
        timepoint_id = dir_name[4:] if dir_name.startswith('sub-') else dir_name

        # Skip excluded subjects/timepoints
        if should_exclude_subject(timepoint_id):
            excluded_subject_count += 1
            continue

        if should_exclude_timepoint(timepoint_id):
            excluded_timepoint_count += 1
            continue

        # Extract subject ID and age
        subject_id, age_str = extract_subject_and_age(timepoint_id)

        if not subject_id:
            continue

        # Check if subject has fold assignments
        if subject_id not in subject_assignments:
            no_assignment_count += 1
            continue

        # Find the connectivity CSV file
        csv_files = glob.glob(os.path.join(conn_dir, "*_averaged_combined_connectivity.csv"))
        if not csv_files:
            no_file_count += 1
            continue

        conn_file = csv_files[0]

        # Create g-factor style timepoint ID
        if age_str and age_str.isdigit():
            gfactor_style_id = f"{subject_id}_Age{age_str}"
        else:
            gfactor_style_id = f"{subject_id}_AgeUnknown"

        # Add this timepoint to ALL 5 outer folds with appropriate train/test assignment
        for outer_fold in range(5):
            if outer_fold in subject_assignments[subject_id]:
                split = subject_assignments[subject_id][outer_fold]  # 'train' or 'test'

                fold_data[outer_fold][split].append({
                    'eid': gfactor_style_id,
                    'original_file': conn_file,
                    'modality_timepoint': timepoint_id,
                    'subject_id': subject_id
                })

        processed_count += 1

    # Log processing statistics
    logger.info(f"Processed {processed_count} connectivity timepoints")
    logger.info(f"Excluded: {excluded_subject_count} subjects, {excluded_timepoint_count} timepoints")
    logger.info(f"No assignment: {no_assignment_count}, No file: {no_file_count}")

    # Log distribution across folds
    total_entries = 0
    for fold in range(5):
        train_count = len(fold_data[fold]['train'])
        test_count = len(fold_data[fold]['test'])
        fold_total = train_count + test_count
        total_entries += fold_total
        logger.info(f"Fold {fold}: {train_count} train + {test_count} test = {fold_total} entries")

    logger.info(f"Total entries across all folds: {total_entries}")

    # Save processed data for each fold
    for fold in range(5):
        for split in ['train', 'test']:
            if fold_data[fold][split]:
                # Create directory
                output_dir = os.path.join(BASE_DIR, "functional", "functional_connectivity", f"fold_{fold}")
                split_dir = os.path.join(output_dir, split)  #
                os.makedirs(output_dir, exist_ok=True)
                os.makedirs(split_dir, exist_ok=True)


                # Copy connectivity files to individual directories (optional)
                # For now, just create index files

                # Create index DataFrame
                index_data = []
                for item in fold_data[fold][split]:
                    index_data.append({
                        'eid': item['eid']
                    })

                # Save index file
                index_df = pd.DataFrame(index_data)
                index_path = os.path.join(output_dir, f"{split}.csv")
                index_df.to_csv(index_path, index=False)

                logger.info(f"Saved {len(index_df)} connectivity entries to fold {fold} {split}")

                # Also save detailed mapping for reference
                mapping_df = pd.DataFrame(fold_data[fold][split])
                mapping_path = os.path.join(output_dir, f"{split}_mapping.csv")
                mapping_df.to_csv(mapping_path, index=False)
                # Actually copy the files
                logger.info(f"Copying connectivity files for fold {fold} {split}")
                copied_count = 0
                failed_count = 0

                for item in fold_data[fold][split]:
                    eid = item['eid']
                    original_file = item['original_file']

                    # Destination file path (what First_level_model expects)
                    dest_file = os.path.join(split_dir, f"{eid}_connectivity.csv")

                    try:
                        if os.path.exists(original_file):
                            shutil.copy2(original_file, dest_file)
                            copied_count += 1
                        else:
                            logger.warning(f"Original file not found: {original_file}")
                            failed_count += 1
                    except Exception as e:
                        logger.error(f"Failed to copy {original_file} to {dest_file}: {e}")
                        failed_count += 1

                logger.info(f"Fold {fold} {split}: Copied {copied_count} files, {failed_count} failed")
                logger.info(f"Saved {len(index_df)} connectivity entries to fold {fold} {split}")

    return fold_data


# analyze and report on missing data
def analyze_missing_data():
    """
    Analyze which timepoints are missing data and for which modalities,
    providing detailed reports to identify patterns.
    """
    logger.info("\n===== MISSING DATA ANALYSIS =====")

    # Get all timepoints that were processed
    all_timepoints = set()
    for category in ['structural', 'functional']:
        for modality in tracked_matches[category]:
            all_timepoints.update(tracked_matches[category][modality]['matched'])
            all_timepoints.update(tracked_matches[category][modality]['missing'])

    logger.info(f"Total timepoints processed: {len(all_timepoints)}")

    # Find timepoints with all modalities
    complete_timepoints = set()
    for timepoint in all_timepoints:
        has_all_modalities = True
        for category in ['structural', 'functional']:
            for modality in tracked_matches[category]:
                if timepoint not in tracked_matches[category][modality]['matched']:
                    has_all_modalities = False
                    break

        if has_all_modalities:
            complete_timepoints.add(timepoint)

    logger.info(f"Timepoints with all modalities: {len(complete_timepoints)}")
    logger.info(f"Timepoints missing at least one modality: {len(all_timepoints) - len(complete_timepoints)}")

    # Analyze missing data by modality
    logger.info("\nMissing data by modality:")
    for category in ['structural', 'functional']:
        for modality in tracked_matches[category]:
            matched = len(tracked_matches[category][modality]['matched'])
            missing = len(tracked_matches[category][modality]['missing'])
            total = matched + missing
            missing_pct = (missing / total * 100) if total > 0 else 0

            logger.info(f"  {category}/{modality}: {missing} missing of {total} ({missing_pct:.1f}%)")

    # Find subjects most affected by missing data
    logger.info("\nSubjects most affected by missing data:")
    subject_missing_counts = {}
    for timepoint in all_timepoints:
        if timepoint in complete_timepoints:
            continue

        subject_id = normalize_subject_id(timepoint)
        if subject_id not in subject_missing_counts:
            subject_missing_counts[subject_id] = 0
        subject_missing_counts[subject_id] += 1

    # Sort subjects by number of missing timepoints
    sorted_subjects = sorted(subject_missing_counts.items(), key=lambda x: x[1], reverse=True)

    if sorted_subjects:
        logger.info(f"Top 10 subjects with missing data:")
        for subject_id, count in sorted_subjects[:10]:
            logger.info(f"  {subject_id}: {count} timepoints with missing data")

    # Create a detailed report of which modalities are missing for incomplete timepoints
    incomplete_timepoints = all_timepoints - complete_timepoints

    if incomplete_timepoints:
        logger.info("\nDetailed analysis of incomplete timepoints:")

        # Track missing modality combinations
        missing_patterns = {}

        for timepoint in incomplete_timepoints:
            missing_modalities = []
            for category in ['structural', 'functional']:
                for modality in tracked_matches[category]:
                    if timepoint not in tracked_matches[category][modality]['matched']:
                        missing_modalities.append(f"{category}/{modality}")

            pattern = ", ".join(sorted(missing_modalities))
            if pattern not in missing_patterns:
                missing_patterns[pattern] = 0
            missing_patterns[pattern] += 1

        # Log the most common missing patterns
        logger.info("Most common missing modality patterns:")
        for pattern, count in sorted(missing_patterns.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {count} timepoints missing: {pattern}")

    # Save full report to CSV
    report_data = []
    for timepoint in all_timepoints:
        timepoint_data = {
            'timepoint_id': timepoint,
            'subject_id': normalize_subject_id(timepoint),
            'complete': timepoint in complete_timepoints
        }

        # Add status for each modality
        for category in ['structural', 'functional']:
            for modality in tracked_matches[category]:
                status = 'present' if timepoint in tracked_matches[category][modality]['matched'] else 'missing'
                timepoint_data[f"{category}_{modality}"] = status

        report_data.append(timepoint_data)

    # Create and save report
    report_df = pd.DataFrame(report_data)
    report_path = os.path.join(BASE_DIR, "modality_completeness_report.csv")
    report_df.to_csv(report_path, index=False)
    logger.info(f"\nDetailed modality completeness report saved to: {report_path}")


def filter_to_common_subjects_corrected():
    """
    CORRECTED: Find subjects that appear in ALL modalities by checking ALL folds.
    Maintain nested CV structure while removing subjects missing from any modality.
    """
    logger.info("Finding common subjects across all modalities (checking ALL folds)...")

    # Step 1: Find ALL unique subjects per modality across ALL folds
    modality_all_subjects = {}

    for modality in STRUCTURAL_MODALITIES + FUNCTIONAL_MODALITIES:
        all_subjects_this_modality = set()

        # Check ALL folds and splits to get complete subject list
        for fold in range(5):
            for split in ['train', 'test']:
                if modality in STRUCTURAL_MODALITIES:
                    file_path = os.path.join(BASE_DIR, "structural", modality, f"fold_{fold}", f"{split}.csv")
                else:
                    file_path = os.path.join(BASE_DIR, "functional", modality, f"fold_{fold}", f"{split}.csv")

                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    if 'eid' in df.columns:
                        for eid in df['eid']:
                            subject_id = normalize_subject_id(eid)
                            if subject_id:
                                all_subjects_this_modality.add(subject_id)

        modality_all_subjects[modality] = all_subjects_this_modality
        logger.info(f"{modality}: {len(all_subjects_this_modality)} total unique subjects")

    # Step 2: Find intersection (subjects present in ALL modalities)
    if modality_all_subjects:
        common_subjects = set.intersection(*modality_all_subjects.values())
        logger.info(f"Common subjects across ALL modalities: {len(common_subjects)}")

        # Log what's being excluded
        for modality, subjects in modality_all_subjects.items():
            excluded = subjects - common_subjects
            if excluded:
                logger.info(f"{modality}: excluding {len(excluded)} subjects not in all modalities")
                # Show a few examples
                example_excluded = list(excluded)[:5]
                logger.info(f"  Examples: {example_excluded}")

        return common_subjects
    else:
        logger.error("No modality data found")
        return set()


def refilter_modality_files_corrected(common_subjects):
    """
    CORRECTED: Filter each fold file to only include common subjects.
    Preserves nested CV structure.
    """
    logger.info(f"Filtering all modality files to {len(common_subjects)} common subjects...")

    # Process each modality
    for modality in STRUCTURAL_MODALITIES + FUNCTIONAL_MODALITIES:
        logger.info(f"Filtering {modality}...")

        # Determine base path
        if modality in STRUCTURAL_MODALITIES:
            base_path = os.path.join(BASE_DIR, "structural", modality)
        else:
            base_path = os.path.join(BASE_DIR, "functional", modality)

        # Process each fold and split
        for fold in range(5):
            for split in ['train', 'test']:
                file_path = os.path.join(base_path, f"fold_{fold}", f"{split}.csv")

                if os.path.exists(file_path):
                    # Load original data
                    df = pd.read_csv(file_path)
                    original_count = len(df)

                    # Filter to common subjects
                    filtered_rows = []
                    for _, row in df.iterrows():
                        eid = row['eid']
                        subject_id = normalize_subject_id(eid)
                        if subject_id in common_subjects:
                            filtered_rows.append(row)

                    # Save filtered data
                    if filtered_rows:
                        filtered_df = pd.DataFrame(filtered_rows)
                        filtered_df.to_csv(file_path, index=False)

                        # Only log if there was a significant change
                        if abs(original_count - len(filtered_df)) > 2:
                            logger.info(
                                f"{modality} fold {fold} {split}: {original_count} → {len(filtered_df)} timepoints")
                    else:
                        # This should be very rare now
                        logger.warning(
                            f"{modality} fold {fold} {split}: No data after filtering (was {original_count})")


def validate_subject_counts():
    """
    Quick validation that we haven't broken the nested CV structure.
    """
    logger.info("Validating subject counts after filtering...")

    # Check each modality
    for modality in STRUCTURAL_MODALITIES + FUNCTIONAL_MODALITIES:
        if modality in STRUCTURAL_MODALITIES:
            base_path = os.path.join(BASE_DIR, "structural", modality)
        else:
            base_path = os.path.join(BASE_DIR, "functional", modality)

        # Count unique subjects across all folds
        all_subjects = set()
        fold_counts = {}

        for fold in range(5):
            fold_subjects = set()
            for split in ['train', 'test']:
                file_path = os.path.join(base_path, f"fold_{fold}", f"{split}.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    split_subjects = set()
                    for eid in df['eid']:
                        subject_id = normalize_subject_id(eid)
                        if subject_id:
                            all_subjects.add(subject_id)
                            fold_subjects.add(subject_id)
                            split_subjects.add(subject_id)

            fold_counts[fold] = len(fold_subjects)

        logger.info(f"{modality}: {len(all_subjects)} total subjects, folds: {fold_counts}")

        # Check if fold counts are reasonable (should be similar across folds)
        fold_values = list(fold_counts.values())
        if fold_values and (max(fold_values) - min(fold_values)) > len(all_subjects) * 0.1:
            logger.warning(f"{modality}: Large variation in subjects per fold - may indicate filtering problem")


def validate_consistent_subjects():
    """
    Validate that all modalities now have exactly the same subjects.
    """
    logger.info("Validating consistent subjects across all modalities...")

    modality_subjects = {}

    # Get subjects from each modality
    for modality in STRUCTURAL_MODALITIES + FUNCTIONAL_MODALITIES:
        if modality in STRUCTURAL_MODALITIES:
            file_path = os.path.join(BASE_DIR, "structural", modality, "fold_0", "train.csv")
        else:
            file_path = os.path.join(BASE_DIR, "functional", modality, "fold_0", "train.csv")

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            subjects = set()
            for eid in df['eid']:
                subject_id = normalize_subject_id(eid)
                if subject_id:
                    subjects.add(subject_id)
            modality_subjects[modality] = subjects

    # Check that all modalities have exactly the same subjects
    reference_modality = list(modality_subjects.keys())[0]
    reference_subjects = modality_subjects[reference_modality]

    all_consistent = True
    for modality, subjects in modality_subjects.items():
        if subjects != reference_subjects:
            difference = len(subjects.symmetric_difference(reference_subjects))
            logger.error(f"{modality} has different subjects than {reference_modality}: {difference} differences")
            all_consistent = False
        else:
            logger.info(f"✓ {modality}: {len(subjects)} subjects (consistent)")

    if all_consistent:
        logger.info(f"✅ All modalities have consistent subjects: {len(reference_subjects)} subjects")
    else:
        logger.error("❌ Modalities have inconsistent subjects")

    return all_consistent


def verify_fold_consistency():
    """
    Verify the consistency of nested CV fold assignments across all modalities.

    In nested CV, each subject should appear in ALL 5 outer folds:
    - 4 times as 'train' (when their inner fold is used for training)
    - 1 time as 'test' (when their inner fold is used for testing)
    """
    logger.info("\n===== NESTED CV FOLD ASSIGNMENT CONSISTENCY VERIFICATION =====")

    validation_passed = True

    # Load nested fold assignments
    try:
        assignments_df = pd.read_csv(FOLD_ASSIGNMENTS_PATH)

        # Create expected assignments structure
        expected_assignments = {}
        for _, row in assignments_df.iterrows():
            subject_id = normalize_subject_id(row['subject_id'])
            outer_fold = int(row['outer_fold'])
            inner_split = row['inner_split']

            if subject_id not in expected_assignments:
                expected_assignments[subject_id] = {}
            expected_assignments[subject_id][outer_fold] = inner_split

        logger.info(f"Loaded expected assignments for {len(expected_assignments)} subjects")

        # Validate that each subject has assignments for all 5 outer folds
        for subject_id, folds in expected_assignments.items():
            if len(folds) != 5:
                logger.error(f"Subject {subject_id} missing fold assignments: has {len(folds)}, needs 5")
                validation_passed = False
            else:
                # Count train vs test appearances
                train_count = sum(1 for split in folds.values() if split == 'train')
                test_count = sum(1 for split in folds.values() if split == 'test')

                if train_count != 4 or test_count != 1:
                    logger.error(f"Subject {subject_id} has invalid nested CV pattern: "
                                 f"{train_count} train, {test_count} test (should be 4,1)")
                    validation_passed = False

        if validation_passed:
            logger.info("✓ Expected nested CV structure is valid")

    except Exception as e:
        logger.error(f"Error loading nested fold assignments: {e}")
        return False

    # Dict to hold subjects by modality, fold and split
    modality_fold_subjects = {}

    # Process each modality
    all_modalities = []
    all_modalities.extend([f"structural/{m}" for m in STRUCTURAL_MODALITIES])
    all_modalities.extend([f"functional/{m}" for m in FUNCTIONAL_MODALITIES])

    for modality_path in all_modalities:
        modality_name = modality_path.split('/')[-1]
        modality_fold_subjects[modality_name] = {}

        # Check each fold
        for fold in range(5):
            modality_fold_subjects[modality_name][fold] = {'train': set(), 'test': set()}

            # Check train and test
            for split in ['train', 'test']:
                csv_path = os.path.join(BASE_DIR, modality_path, f"fold_{fold}", f"{split}.csv")

                if not os.path.exists(csv_path):
                    logger.warning(f"Missing file: {csv_path}")
                    continue

                # Load data and extract subject IDs
                try:
                    df = pd.read_csv(csv_path)

                    if 'eid' not in df.columns:
                        logger.warning(f"No 'eid' column in {csv_path}")
                        continue

                    # Extract subject IDs from eid column
                    for eid in df['eid']:
                        subject_id = normalize_subject_id(eid)
                        if subject_id:
                            modality_fold_subjects[modality_name][fold][split].add(subject_id)

                    logger.info(
                        f"{modality_name} fold {fold} {split}: {len(modality_fold_subjects[modality_name][fold][split])} subjects")

                except Exception as e:
                    logger.error(f"Error reading {csv_path}: {e}")
                    validation_passed = False

    # Validate nested CV pattern for each modality
    logger.info("\nValidating nested CV pattern for each modality...")

    for modality, fold_data in modality_fold_subjects.items():
        logger.info(f"\nValidating {modality}:")

        # Track each subject's appearances across folds
        subject_appearances = {}

        for fold in range(5):
            for split in ['train', 'test']:
                for subject in fold_data[fold][split]:
                    if subject not in subject_appearances:
                        subject_appearances[subject] = {'train_folds': [], 'test_folds': []}
                    subject_appearances[subject][f'{split}_folds'].append(fold)

        # Check nested CV pattern for each subject
        nested_cv_violations = 0
        assignment_mismatches = 0

        for subject, appearances in subject_appearances.items():
            n_train = len(appearances['train_folds'])
            n_test = len(appearances['test_folds'])

            # Check nested CV pattern (4 train, 1 test)
            if n_train != 4 or n_test != 1:
                logger.error(f"NESTED CV VIOLATION in {modality}: Subject {subject} appears in "
                             f"{n_train} train folds and {n_test} test folds (should be 4,1)")
                nested_cv_violations += 1
                validation_passed = False
                continue

            # Check that assignments match expected pattern
            if subject in expected_assignments:
                test_fold = appearances['test_folds'][0]
                expected_test_split = expected_assignments[subject].get(test_fold)

                if expected_test_split != 'test':
                    logger.error(f"ASSIGNMENT MISMATCH in {modality}: Subject {subject} is test in fold {test_fold} "
                                 f"but expected assignment is '{expected_test_split}'")
                    assignment_mismatches += 1
                    validation_passed = False

                # Check train folds
                for train_fold in appearances['train_folds']:
                    expected_train_split = expected_assignments[subject].get(train_fold)
                    if expected_train_split != 'train':
                        logger.error(
                            f"ASSIGNMENT MISMATCH in {modality}: Subject {subject} is train in fold {train_fold} "
                            f"but expected assignment is '{expected_train_split}'")
                        assignment_mismatches += 1
                        validation_passed = False

        # Summary for this modality
        total_subjects = len(subject_appearances)
        if nested_cv_violations == 0 and assignment_mismatches == 0:
            logger.info(f"✓ {modality}: {total_subjects} subjects with valid nested CV pattern")
        else:
            logger.error(
                f"✗ {modality}: {nested_cv_violations} nested CV violations, {assignment_mismatches} assignment mismatches")

    # Check consistency across modalities
    logger.info("\nChecking consistency across modalities...")

    if len(modality_fold_subjects) > 1:
        # Compare first modality with all others
        reference_modality = list(modality_fold_subjects.keys())[0]

        for modality in list(modality_fold_subjects.keys())[1:]:
            for fold in range(5):
                for split in ['train', 'test']:
                    # Get reference and comparison sets
                    ref_subjects = modality_fold_subjects.get(reference_modality, {}).get(fold, {}).get(split, set())
                    comp_subjects = modality_fold_subjects.get(modality, {}).get(fold, {}).get(split, set())

                    # Check for differences
                    only_in_ref = ref_subjects - comp_subjects
                    only_in_comp = comp_subjects - ref_subjects

                    if only_in_ref:
                        logger.warning(f"{len(only_in_ref)} subjects in {reference_modality} fold {fold} {split} "
                                       f"but not in {modality}")
                        if len(only_in_ref) <= 10:  # Show examples for small differences
                            logger.warning(f"Missing subjects: {only_in_ref}")

                    if only_in_comp:
                        logger.warning(f"{len(only_in_comp)} subjects in {modality} fold {fold} {split} "
                                       f"but not in {reference_modality}")
                        if len(only_in_comp) <= 10:
                            logger.warning(f"Extra subjects: {only_in_comp}")

                    # Large differences indicate a problem
                    if len(only_in_ref) > 5 or len(only_in_comp) > 5:
                        validation_passed = False

    # Summary statistics
    logger.info("\n--- Fold Assignment Summary ---")

    for modality, fold_data in modality_fold_subjects.items():
        total_entries = 0
        fold_counts = {}

        for fold in range(5):
            train_count = len(fold_data[fold]['train'])
            test_count = len(fold_data[fold]['test'])
            fold_total = train_count + test_count
            total_entries += fold_total
            fold_counts[fold] = {'train': train_count, 'test': test_count}

        logger.info(f"{modality}: Total {total_entries} entries")
        for fold in range(5):
            train_count = fold_counts[fold]['train']
            test_count = fold_counts[fold]['test']
            logger.info(f"  Fold {fold}: {train_count} train, {test_count} test")

    # Final verdict
    if validation_passed:
        logger.info("\n✅ All nested CV consistency checks PASSED")
    else:
        logger.error("\n❌ Some nested CV consistency checks FAILED - see warnings and errors above")

    logger.info("===== NESTED CV VALIDATION COMPLETE =====")
    return validation_passed


def track_subject_assignments():
    """
    Create comprehensive report of subject assignments and exclusions.
    Generates a CSV with detailed breakdown of all subjects.
    """
    logger.info("Generating comprehensive subject assignment report...")

    # Data structure to track all subjects
    subject_tracking = {}

    # Step 1: Collect all potential subjects from all sources

    # From functional connectivity directories
    subject_dirs = glob.glob(os.path.join(FUNC_CONN_DIR, "sub-*"))
    for subject_dir in subject_dirs:
        basename = os.path.basename(subject_dir)
        subject_timepoint_id = basename[4:] if basename.startswith("sub-") else basename
        subject_id = normalize_subject_id(subject_timepoint_id)

        if subject_id and subject_id not in subject_tracking:
            subject_tracking[subject_id] = {
                "status": "unprocessed",
                "reason": "",
                "fold": None,
                "split": None,
                "modalities_present": ["functional_connectivity"],
                "modalities_missing": [],
                "timepoints": [subject_timepoint_id]
            }

    # Add subjects from structural data
    for modality in STRUCTURAL_MODALITIES:
        input_path = os.path.join(STRUCT_DATA_DIR, f"{modality}.csv")
        if os.path.exists(input_path):
            df = pd.read_csv(input_path)
            subject_col = df.columns[0]

            for _, row in df.iterrows():
                subject_timepoint_id = str(row[subject_col])
                subject_id = normalize_subject_id(subject_timepoint_id)

                if subject_id:
                    if subject_id not in subject_tracking:
                        subject_tracking[subject_id] = {
                            "status": "unprocessed",
                            "reason": "",
                            "fold": None,
                            "split": None,
                            "modalities_present": [modality],
                            "modalities_missing": [],
                            "timepoints": [subject_timepoint_id]
                        }
                    else:
                        if modality not in subject_tracking[subject_id]["modalities_present"]:
                            subject_tracking[subject_id]["modalities_present"].append(modality)
                        if subject_timepoint_id not in subject_tracking[subject_id]["timepoints"]:
                            subject_tracking[subject_id]["timepoints"].append(subject_timepoint_id)

    # Add ALFF and ReHo subjects
    for modality in ['alff', 'reho']:
        input_path = os.path.join(ALFF_REHO_DIR, f"{modality}_aggregated_data_ml.csv")
        if os.path.exists(input_path):
            df = pd.read_csv(input_path)
            subject_col = df.columns[0]

            for _, row in df.iterrows():
                subject_timepoint_id = str(row[subject_col])
                subject_id = normalize_subject_id(subject_timepoint_id)

                if subject_id:
                    if subject_id not in subject_tracking:
                        subject_tracking[subject_id] = {
                            "status": "unprocessed",
                            "reason": "",
                            "fold": None,
                            "split": None,
                            "modalities_present": [modality],
                            "modalities_missing": [],
                            "timepoints": [subject_timepoint_id]
                        }
                    else:
                        if modality not in subject_tracking[subject_id]["modalities_present"]:
                            subject_tracking[subject_id]["modalities_present"].append(modality)
                        if subject_timepoint_id not in subject_tracking[subject_id]["timepoints"]:
                            subject_tracking[subject_id]["timepoints"].append(subject_timepoint_id)

    # Step 2: Mark excluded subjects
    normalized_exclusions = set()
    for subject in SUBJECT_EXCLUDE:
        normalized_id = normalize_subject_id(subject)
        normalized_exclusions.add(normalized_id)

        if normalized_id in subject_tracking:
            subject_tracking[normalized_id]["status"] = "excluded"
            subject_tracking[normalized_id]["reason"] = "In global exclusion list"

    # Step 3: Check fold assignments
    assignments_df = pd.read_csv(FOLD_ASSIGNMENTS_PATH)

    # Track subjects with fold assignments
    assigned_count = 0
    for _, row in assignments_df.iterrows():
        subject_id = normalize_subject_id(row['subject_id'])
        fold = int(row['outer_fold'])
        split = row['inner_split']

        if subject_id:
            if subject_id in subject_tracking:
                if subject_id not in normalized_exclusions:
                    subject_tracking[subject_id]["status"] = "assigned"
                    subject_tracking[subject_id]["fold"] = fold
                    subject_tracking[subject_id]["split"] = split
                    assigned_count += 1
            else:
                # Subject in fold assignments but not found in any modality
                subject_tracking[subject_id] = {
                    "status": "missing_data",
                    "reason": "Has fold assignment but no modality data found",
                    "fold": fold,
                    "split": split,
                    "modalities_present": [],
                    "modalities_missing": STRUCTURAL_MODALITIES + FUNCTIONAL_MODALITIES,
                    "timepoints": []
                }

    # Step 4: Check for subjects in data that don't have fold assignments
    for subject_id, info in subject_tracking.items():
        if info["status"] == "unprocessed":
            subject_tracking[subject_id]["status"] = "no_assignment"
            subject_tracking[subject_id]["reason"] = "No fold assignment found"

    # Step 5: Check modality completeness for assigned subjects
    all_modalities = STRUCTURAL_MODALITIES + FUNCTIONAL_MODALITIES
    for subject_id, info in subject_tracking.items():
        present_modalities = set(info["modalities_present"])
        missing_modalities = [m for m in all_modalities if m not in present_modalities]
        subject_tracking[subject_id]["modalities_missing"] = missing_modalities

        # Update status for incomplete subjects
        if info["status"] == "assigned" and missing_modalities:
            subject_tracking[subject_id]["reason"] += f"Missing data: {', '.join(missing_modalities)}"

    # Step 6: Generate report
    report_data = []
    for subject_id, info in subject_tracking.items():
        report_data.append({
            "subject_id": subject_id,
            "status": info["status"],
            "reason": info["reason"],
            "fold": info["fold"],
            "split": info["split"],
            "modalities_present": ", ".join(info["modalities_present"]),
            "modalities_missing": ", ".join(info["modalities_missing"]),
            "timepoints": ", ".join(info["timepoints"][:3]) + ("..." if len(info["timepoints"]) > 3 else "")
        })

    # Create DataFrame and save to CSV
    report_df = pd.DataFrame(report_data)
    # Sort by status (assigned first, then others)
    status_order = {"assigned": 0, "excluded": 1, "no_assignment": 2, "missing_data": 3, "unprocessed": 4}
    report_df["status_order"] = report_df["status"].map(status_order)
    report_df = report_df.sort_values(["status_order", "subject_id"]).drop("status_order", axis=1)

    report_path = os.path.join(BASE_DIR, "subject_assignment_report.csv")
    report_df.to_csv(report_path, index=False)

    # Generate summary statistics
    status_counts = report_df["status"].value_counts()
    logger.info("\n===== SUBJECT ASSIGNMENT SUMMARY =====")
    logger.info(f"Total subjects found: {len(report_df)}")
    for status, count in status_counts.items():
        logger.info(f"{status}: {count} subjects")

    # Add fold distribution summary
    assigned_df = report_df[report_df["status"] == "assigned"]
    fold_counts = assigned_df.groupby(["fold", "split"]).size().reset_index(name="count")

    logger.info("\n===== TIMEPOINT ANALYSIS =====")
    # Count subjects with multiple timepoints in raw data
    subject_timepoint_counts = defaultdict(int)
    for subject_dir in glob.glob(os.path.join(FUNC_CONN_DIR, "sub-*")):
        basename = os.path.basename(subject_dir)
        subject_timepoint_id = basename[4:] if basename.startswith("sub-") else basename
        subject_id = normalize_subject_id(subject_timepoint_id)
        if subject_id:
            subject_timepoint_counts[subject_id] += 1

    multi_timepoint_counts = {count: 0 for count in range(1, 6)}
    multi_timepoint_counts["6+"] = 0
    for subject_id, count in subject_timepoint_counts.items():
        if count >= 6:
            multi_timepoint_counts["6+"] += 1
        else:
            multi_timepoint_counts[count] += 1

    logger.info("Timepoints per subject in raw data:")
    # First handle the integer counts in sorted order
    for count in range(1, 6):
        num_subjects = multi_timepoint_counts[count]
        logger.info(f"  {num_subjects} subjects have {count} timepoint{'s' if count > 1 else ''}")

    # Then handle the "6+" count at the end
    logger.info(f"  {multi_timepoint_counts['6+']} subjects have 6+ timepoints")


    # Count how many timepoints were retained for subjects with multiple timepoints
    # in the final processed data
    logger.info("\nSubjects with multiple timepoints in processed data:")
    processed_subject_timepoints = defaultdict(list)
    for category in ["structural", "functional"]:
        for modality in (STRUCTURAL_MODALITIES if category == "structural" else FUNCTIONAL_MODALITIES):
            for fold in range(5):
                for split in ["train", "test"]:
                    file_path = os.path.join(BASE_DIR, category, modality, f"fold_{fold}", f"{split}.csv")
                    if os.path.exists(file_path):
                        try:
                            df = pd.read_csv(file_path)
                            if 'eid' in df.columns:
                                for eid in df['eid']:
                                    subject_id = normalize_subject_id(eid)
                                    if subject_id:
                                        processed_subject_timepoints[subject_id].append(eid)
                        except Exception as e:
                            logger.error(f"Error reading {file_path}: {e}")

    multi_processed_counts = {count: 0 for count in range(1, 6)}
    multi_processed_counts["6+"] = 0
    for subject_id, timepoints in processed_subject_timepoints.items():
        count = len(set(timepoints))  # Use set to count unique timepoints
        if count >= 6:
            multi_processed_counts["6+"] += 1
        else:
            multi_processed_counts[count] += 1

    # First handle the integer counts in sorted order
    for count in range(1, 6):
        num_subjects = multi_processed_counts[count]
        logger.info(f"  {num_subjects} subjects have {count} timepoint{'s' if count > 1 else ''}")
    # Then handle the "6+" count at the end
    logger.info(f"  {multi_processed_counts['6+']} subjects have 6+ timepoints")

    # Calculate how many potential timepoints were lost
    total_raw_timepoints = sum(subject_timepoint_counts.values())
    total_processed_timepoints = sum(len(set(tps)) for tps in processed_subject_timepoints.values())
    logger.info(
        f"\nTimepoint retention: {total_processed_timepoints} of {total_raw_timepoints} timepoints retained ({total_processed_timepoints / total_raw_timepoints * 100:.1f}%)")

    # For more detailed analysis of final assignments, check actual data files
    logger.info("\n===== FINAL PROCESSED FILE COUNTS =====")
    final_counts = {
        "structural": {m: {fold: {"train": 0, "test": 0} for fold in range(5)} for m in STRUCTURAL_MODALITIES},
        "functional": {m: {fold: {"train": 0, "test": 0} for fold in range(5)} for m in FUNCTIONAL_MODALITIES}
    }

    # Count subjects in final processed files for each modality and fold
    for category in ["structural", "functional"]:
        for modality in final_counts[category].keys():
            for fold in range(5):
                for split in ["train", "test"]:
                    file_path = os.path.join(BASE_DIR, category, modality, f"fold_{fold}", f"{split}.csv")
                    if os.path.exists(file_path):
                        try:
                            df = pd.read_csv(file_path)
                            count = len(df)
                            final_counts[category][modality][fold][split] = count
                        except Exception as e:
                            logger.error(f"Error reading {file_path}: {e}")

    # Output summary of processed file counts
    processed_total = 0
    for category in ["structural", "functional"]:
        for modality in final_counts[category].keys():
            modality_total = 0
            logger.info(f"\n{category.upper()}/{modality}:")
            for fold in range(5):
                fold_train = final_counts[category][modality][fold]["train"]
                fold_test = final_counts[category][modality][fold]["test"]
                fold_total = fold_train + fold_test
                modality_total += fold_total
                logger.info(f"  Fold {fold}: {fold_train} train + {fold_test} test = {fold_total} timepoints")
            logger.info(f"  Total for {modality}: {modality_total} timepoints")
            if modality == STRUCTURAL_MODALITIES[0] or modality == FUNCTIONAL_MODALITIES[0]:
                processed_total = modality_total

    logger.info(f"\nTotal processed timepoints (based on primary modality): {processed_total}")

    # Create a second CSV with the final processed counts
    processed_data = []
    for category in ["structural", "functional"]:
        for modality in final_counts[category].keys():
            for fold in range(5):
                for split in ["train", "test"]:
                    processed_data.append({
                        "category": category,
                        "modality": modality,
                        "fold": fold,
                        "split": split,
                        "count": final_counts[category][modality][fold][split]
                    })

    processed_df = pd.DataFrame(processed_data)
    processed_path = os.path.join(BASE_DIR, "processed_timepoints_summary.csv")
    processed_df.to_csv(processed_path, index=False)
    logger.info(f"Processed timepoints summary saved to: {processed_path}")

    logger.info(f"Detailed subject assignment report saved to: {report_path}")
    return report_df

def main():
    """Updated main function with corrected subject filtering."""
    logger.info("Starting preprocessing script with nested cross-validation")

    # Set up directory structure
    setup_directory_structure()

    # Process each modality using nested CV fold assignments
    process_structural_data()
    process_alff_reho_data()
    process_functional_connectivity()

    # CORRECTED: Filter to common subjects using ALL folds
    common_subjects = filter_to_common_subjects_corrected()
    if common_subjects:
        refilter_modality_files_corrected(common_subjects)
        validate_subject_counts()
        consistency_result = validate_consistent_subjects()
    else:
        logger.error("No common subjects found")
        return

    # Verify fold consistency
    if consistency_result:
        fold_consistency = verify_fold_consistency()
        if fold_consistency:
            logger.info("✅ Preprocessing complete with verified nested CV consistency and matched subjects")
        else:
            logger.warning("⚠️ Preprocessing complete but fold consistency verification failed")
    else:
        logger.warning("⚠️ Preprocessing complete but subject consistency failed")

    # Generate detailed subject assignment report
    # track_subject_assignments() #disabling for now.

if __name__ == "__main__":
    main()