#!/usr/bin/env python3

"""
Oregon REST Connectivity Visualization Script using Time Series Data
with Cole Network Ordering

This script processes parcellated time series data from XCP-D outputs,
computes correlations directly from the time series, and organizes
connectivity matrices according to the Cole network ordering.
This ensures proper correlation computation between cortical and
subcortical regions and correct network-based visualization.
"""

#########################################################################################
#                          CONFIGURATION VARIABLES - EDIT FOR YOUR NEEDS                #
#########################################################################################

# XCP-D data directory (where time series files are located)
DATA_DIR = "/media/hcs-sci-psy-narun/OREGON_ADHD_1000/OREGON_ONE_SINGLE/derivatives/xcp_d-0.10.6_acompcor_censorMT_redoMay_12"

# Directory to save visualizations
OUTPUT_DIR = "/media/hcs-sci-psy-narun/Jack/Functional_connectivity_XCPD_Censor_May"

# Path to the NDAR phenotype file (CSV format)
NDAR_FILE = "/media/hcs-sci-psy-narun/Jack/ndar_subject01.csv"

# Path to the Cole network mapping file
COLE_NETWORK_FILE = "/media/hcs-sci-psy-narun/Jack/groi_colenetwork_mapping.csv"

# Task name (for OREGON it's typically 'rest')
TASK = "rest"

# List of specific subject IDs to analyze (if None, analyzes all subjects)
SUBJECT_LIST = None
# ^ Format: ["sub-001", "sub-002", "sub-003"]

# List of subject IDs to exclude (if None, excludes no subjects)
SUBJECT_EXCLUDE = [
    "NDARINVKPGRKEZW", "NDARINVCGJ4G85F", "NDARINV0H2MNXAU", "NDARINVE5F245ZC",
    "NDARINVD0UHEXV5", "NDARINVKKE5M36H", "NDARINVDPJEMFNZ", "NDARINV9U08EKVL",
    "NDARINVEPTLP4ED", "NDARINVT4FZHCTW", "NDARINV4E1GLBYM", "NDARINV8LZY8RBZ",
    "NDARINVUYKLT0YE", "NDARINVXF3RN4HD", "NDARINV09MPGC1A", "NDARINVPF66BBJZ",
    "NDARINVZZVK3FXR", "NDARINVT3V0N9JR", "NDARINV9LMBWJ8Y", "NDARINVKF5TZ56H",
    "NDARINVRD9739F6", "NDARINVXGH2TU13", "NDARINVDGLFV9JE", "NDARINV31PK79NV",
    "NDARINV4AWRRTML", "NDARINVKRWMZUZ0", "NDARINVC597YE96", "NDARINVT5HEHVWA",
    "NDARINVM5GDJH10"
]
# ^ Format: ["sub-001", "sub-002", "sub-003"]

# List of specific subject-timepoint combinations to exclude
# This is for the additional list which may have valid data at other timepoints
SUBJECT_TIMEPOINT_EXCLUDE = [
    "NDARAB554KWNAge156MonthsStudyYear05",
    "NDARBL381WXTAge186MonthsStudyYear05",
    "NDARBL853AM1Age189MonthsStudyYear05",
    "NDAREA400JB9Age136MonthsStudyYear05",
    "NDARFG935JBWAge147MonthsStudyYear05",
    "NDARHC389FNBAge164MonthsStudyYear05",
    "NDARHM950VRNAge172MonthsStudyYear05",
    "NDARINV0TMRP8W9Age110MonthsStudyYear02",
    "NDARINV7P60663LAge89MonthsStudyYear01",
    "NDARINVEMKF9W2FAge143MonthsStudyYear05",
    "NDARINVFNCBBB3ZAge177MonthsStudyYear05",
    "NDARINVVEXNDKF5Age87MonthsStudyYear01",
    "NDARAN054EUFAge99MonthsStudyYear02",
    "NDARBE279VUGAge109MonthsStudyYear02",
    "NDARBF078JZHAge111MonthsStudyYear01",
    "NDARBF078JZHAge123MonthsStudyYear02",
    "NDAREH600YYDAge137MonthsStudyYear01",
    "NDAREH600YYDAge161MonthsStudyYear03",
    "NDARGW477DZUAge107MonthsStudyYear01",
    "NDARHC389FNBAge141MonthsStudyYear03",
    "NDARINV09MPGC1AAge116MonthsStudyYear01",
    "NDARINV0H2MNXAUAge104MonthsStudyYear01",
    "NDARINV0X7T7YPRAge126MonthsStudyYear01",
    "NDARINV4FCNLAA6Age124MonthsStudyYear02",
    "NDARINV5VML6URXAge125MonthsStudyYear02",
    "NDARINV8R5NVBTHAge098MonthsStudyYear01",
    "NDARINVELE7W2WVAge101MonthsStudyYear01",
    "NDARINVHD6646BYAge132MonthsStudyYear01",
    "NDARINVL9PBUMPVAge151MonthsStudyYear01",
    "NDARINVM78JYC7GAge104MonthsStudyYear01",
    "NDARINVXVR7YLZZAge119MonthsStudyYear01",
    "NDARINV31PK79NVAge100MonthsStudyYear01",
    "NDARINV3TRYNB4FAge124MonthsStudyYear03",
    "NDARINV3ZN3A79XAge103MonthsStudyYear01",
    "NDARINV5W4MYDL2Age104MonthsStudyYear01",
    "NDARINV6KGN3D1CAge092MonthsStudyYear01",
    "NDARINV7XANUXTMAge097MonthsStudyYear01",
    "NDARINV8BP9WA01Age134MonthsStudyYear03",
    "NDARINV8LZY8RBZAge098MonthsStudyYear01",
    "NDARINV8UJ769XBAge132MonthsStudyYear03",
    "NDARINV9D6YLNDCAge088MonthsStudyYear01",
    "NDARINV9LMBWJ8YAge97MonthsStudyYear01",
    "NDARINV9P3TB42ZAge093MonthsStudyYear01",
    "NDARINV9U08EKVLAge108MonthsStudyYear01",
    "NDARINV9U08EKVLAge120MonthsStudyYear02",
    "NDARINV9U08EKVLAge135MonthsStudyYear03",
    "NDARINV9VDT98JWAge120MonthsStudyYear01",
    "NDARINVA1BEG2CLAge105MonthsStudyYear01",
    "NDARINVA7LXRXJWAge124MonthsStudyYear02",
    "NDARINVA7LXRXJWAge131MonthsStudyYear03",
    "NDARINVBDZMWL9JAge118MonthsStudyYear01",
    "NDARINVBDZMWL9JAge127MonthsStudyYear02",
    "NDARINVBDZMWL9JAge139MonthsStudyYear03",
    "NDARINVBT2NFBE5Age092MonthsStudyYear01",
    "NDARINVCGJ4G85FAge101MonthsStudyYear01",
    "NDARINVCGJ4G85FAge126MonthsStudyYear03",
    "NDARINVD140CNXVAge131MonthsStudyYear02",
    "NDARINVE5F245ZCAge103MonthsStudyYear01",
    "NDARINVEPTLP4EDAge109MonthsStudyYear01",
    "NDARINVFUJJ03CHAge108MonthsStudyYear02",
    "NDARINVG19CZ4NUAge123MonthsStudyYear01",
    "NDARINVHZCJVCZ9Age129MonthsStudyYear02",
    "NDARINVHZCJVCZ9Age141MonthsStudyYear03",
    "NDARINVJ2GPW7RHAge095MonthsStudyYear01",
    "NDARINVJ8TJZ9BCAge088MonthsStudyYear01",
    "NDARINVJ8TJZ9BCAge100MonthsStudyYear02",
    "NDARINVKF5TZ56HAge101MonthsStudyYear01",
    "NDARINVKM864ZFCAge110MonthsStudyYear02",
    "NDARINVL50ABG4TAge145MonthsStudyYear02",
    "NDARINVM41MAHX2Age135MonthsStudyYear01",
    "NDARINVM5GDJH10Age110MonthsStudyYear01",
    "NDARINVM5PGAF1WAge107MonthsStudyYear02",
    "NDARINVNPHR7KZGAge151MonthsStudyYear06",
    "NDARINVNZ8Y8Y9MAge102MonthsStudyYear01",
    "NDARINVNZ8Y8Y9MAge126MonthsStudyYear03",
    "NDARINVPF66BBJZAge112MonthsStudyYear01",
    "NDARINVRD9739F6Age106MonthsStudyYear01",
    "NDARINVT38Y2MU1Age096MonthsStudyYear01",
    "NDARINVT3V0N9JRAge100MonthsStudyYear01",
    "NDARINVT5HEHVWAAge108MonthsStudyYear01",
    "NDARINVTXYCDHAWAge112MonthsStudyYear01",
    "NDARINVUYKLT0YEAge097MonthsStudyYear01",
    "NDARINVV4TFAYPZAge137MonthsStudyYear01",
    "NDARINVVWT2EXNRAge125MonthsStudyYear01",
    "NDARINVWU4B1BEJAge105MonthsStudyYear01",
    "NDARINVXF3RN4HDAge118MonthsStudyYear01",
    "NDARINVXGH2TU13Age114MonthsStudyYear01",
    "NDARINVYHRR74C3Age095MonthsStudyYear01",
    "NDARINVZBNJXU7MAge124MonthsStudyYear02",
    "NDARLW133FWGAge097MonthsStudyYear01",
    "NDARMA471CL4Age112MonthsStudyYear01",
    "NDARPC423JTCAge147MonthsStudyYear01",
    "NDARPJ373TU7Age102MonthsStudyYear02",
    "NDARPJ373TU7Age114MonthsStudyYear03",
    "NDARTD417EC4Age199MonthsStudyYear05",
    "NDARTE304MDHAge100MonthsStudyYear01",
    "NDARUE152ELBAge104MonthsStudyYear01",
    "NDARUN588MBCAge105MonthsStudyYear01",
    "NDARUN588MBCAge128MonthsStudyYear03",
    "NDARWM302NLPAge211MonthsStudyYear06",
    "NDARXK430GTQAge116MonthsStudyYear03",
    "NDARYD918PBGAge120MonthsStudyYear02",
    "NDARZN519WNHAge098MonthsStudyYear01",
    "NDARZL911CDTAge158MonthsStudyYear05"
]

# Generate subject-level visualizations? (False to only generate group visualizations)
GENERATE_SUBJECT_VISUALIZATIONS = False

# Directory to store log files (if None, logs saved to script directory)
LOG_DIR = "/media/hcs-sci-psy-narun/Jack/XCP-D_test/logs/"

#########################################################################################
#                                   IMPORT LIBRARIES                                    #
#########################################################################################

import os
import sys
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
import gc
import traceback
from datetime import datetime
import seaborn as sns
from nilearn import plotting
import concurrent.futures
import psutil
import xml.etree.ElementTree as ET

# Try to use nilearn for CIFTI, but import nibabel as fallback
try:
    from nilearn import image
    from nilearn.connectome import ConnectivityMeasure

    HAS_NILEARN_CIFTI = False  # We'll check this later in the code
except ImportError:
    HAS_NILEARN_CIFTI = False

try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("WARNING: nibabel is not installed. This is required for CIFTI file handling.")

# Try to import tqdm for progress bars if available
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

########################
# Additional variables #
########################
# Configuration for parallelization and testing
# Configure how many subjects to process in parallel - adjust as needed
# Default to 75% of available cores for parallel processing
MAX_WORKERS = max(int(os.cpu_count() * 0.75), 1)
# How much memory to reserve (GB) - prevent out-of-memory errors
MEMORY_RESERVE_GB = 32  # Reserve 32GB to avoid system instability

# Process only a fraction of subjects (1.0 = all, 0.1 = 10%, etc.)
# Set to None to process all subjects
PROCESS_FRACTION = None  # Process x% for testing

#########################################################################################
#                                  HELPER FUNCTIONS                                     #
#########################################################################################

def setup_logging(log_dir=None):
    """Set up logging configuration with both file and console output"""
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'connectivity_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    else:
        log_file = f'connectivity_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    # Configure logging to file and console
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    return logging.getLogger(__name__)


def clean_subject_id(sub_label):
    """
    Clean the subject ID by removing any file extensions or suffixes.

    Parameters:
    -----------
    sub_label : str
        Original subject label

    Returns:
    --------
    str : Cleaned subject ID
    """
    # Remove .html extension if present
    if sub_label.endswith('.html'):
        sub_label = sub_label[:-5]

    # Remove _executive_summary suffix if present
    if '_executive_summary' in sub_label:
        sub_label = sub_label.replace('_executive_summary', '')

    # Remove any other common extensions
    for ext in ['.txt', '.csv', '.tsv', '.json']:
        if sub_label.endswith(ext):
            sub_label = sub_label[:-len(ext)]

    return sub_label

def parse_timepoint_info(timepoint_id):
    """
    Parse a timepoint ID to extract age and study year information.

    Parameters:
    -----------
    timepoint_id : str
        Timepoint ID (e.g., "NDARXXXAge112MonthsStudyYear02" where 112 means 112 months of age)

    Returns:
    --------
    tuple : (age_months, study_year) or (None, None) if not parsable
    """
    if timepoint_id is None:
        return None, None

    # Split by "Age" to get age part
    parts = timepoint_id.split("Age")
    if len(parts) < 2:
        return None, None

    # Extract age digits from the format
    age_match = re.search(r'(\d+)Months', parts[1])
    if not age_match:
        return None, None

    age_str = age_match.group(1)

    # Convert YYM or YYMM format to total months
    if len(age_str) == 3:  # YYM format (e.g., 112 = 11 years, 2 months)
        years = int(age_str[:2])
        months = int(age_str[2])
    elif len(age_str) == 4:  # YYMM format (e.g., 1102 = 11 years, 2 months)
        years = int(age_str[:2])
        months = int(age_str[2:])
    else:
        # Handle unexpected format - try to parse the whole thing as months
        try:
            return int(age_str), None
        except:
            return None, None

    # Convert to total months
    age_months = (years * 12) + months

    # Extract study year
    year_match = re.search(r'StudyYear(\d+)', timepoint_id)
    study_year = int(year_match.group(1)) if year_match else None

    return age_months, study_year

def should_exclude_timepoint(subject_id, logger, age_months=None, interview_age=None):
    """
    Determine if a specific subject-timepoint combination should be excluded.

    Parameters:
    -----------
    subject_id : str
        Subject identifier
    age_months : int or None
        Age in months from timepoint ID parsing
    interview_age : float or None
        Interview age from data (in months)

    Returns:
    --------
    bool : True if should exclude, False otherwise
    """
    # First check if subject should be completely excluded (all timepoints)
    normalized_id = normalize_subject_id(subject_id)

    # Get normalized complete exclusion list
    normalized_complete = [normalize_subject_id(s) for s in SUBJECT_EXCLUDE]

    if normalized_id in normalized_complete:
        logger.info(f"Excluding subject {subject_id} (normalized: {normalized_id}) - in complete exclusion list")
        return True

    # If we don't have age information, we can't match specific timepoints
    if age_months is None and interview_age is None:
        return False

    # Use interview_age if age_months is not available
    if age_months is None and interview_age is not None:
        age_months = int(interview_age)

    # Check each timepoint exclusion entry
    for excluded_entry in SUBJECT_TIMEPOINT_EXCLUDE:
        # Extract subject ID from the excluded entry
        excluded_id = normalize_subject_id(excluded_entry)

        # Skip if normalized IDs don't match
        if normalized_id != excluded_id:
            continue

        # Extract age from the excluded entry
        excluded_age, _ = parse_timepoint_info(excluded_entry)

        # Skip if we couldn't parse the age
        if excluded_age is None:
            logger.warning(f"Couldn't parse age from exclusion entry: {excluded_entry}")
            continue

        # Check if the ages match (within 1 month tolerance)
        if abs(age_months - excluded_age) <= 1:
            logger.info(f"Excluding timepoint for subject {subject_id} at age {age_months} months - " +
                        f"matched exclusion entry {excluded_entry} (age: {excluded_age} months)")
            return True

    # If we get here, this subject-timepoint should not be excluded
    return False

def normalize_subject_id(subject_id):
    """
    Normalize subject ID by extracting just the core identifier (e.g., NDARINVXYZ)
    without any prefixes, suffixes, or additional information.

    Parameters:
    -----------
    subject_id : str
        Original subject ID which might include prefixes, age info, etc.

    Returns:
    --------
    str : Normalized subject ID
    """
    if subject_id is None:
        return None

    # Convert to string if not already
    subject_id = str(subject_id)

    # Remove 'sub-' prefix if present
    if subject_id.startswith('sub-'):
        subject_id = subject_id[4:]

    # Strip underscores
    subject_id = subject_id.replace('_', '')

    # If it includes "Age", extract just the part before it
    if "Age" in subject_id:
        subject_id = subject_id.split("Age")[0]

    return subject_id

def load_cole_network_mapping(file_path, logger):
    """
    Load Cole network mapping for region ordering and network information.

    Parameters:
    -----------
    file_path : str
        Path to Cole network mapping CSV file
    logger : Logger
        Logger object

    Returns:
    --------
    DataFrame : Loaded Cole network mapping
    """
    try:
        logger.info(f"Loading Cole network mapping from {file_path}")
        mapping_df = pd.read_csv(file_path, index_col=0)

        # Check for required columns
        required_cols = ['name_Glasser', 'Cole_net_name', 'orig_order', 'Cole_net_order']
        missing_cols = [col for col in required_cols if col not in mapping_df.columns]

        if missing_cols:
            logger.error(f"Missing required columns in Cole network mapping file: {missing_cols}")
            # Try to create Cole_net_order if missing but we have other network info
            if 'Cole_net_order' in missing_cols and 'Cole_net_name' in mapping_df.columns:
                logger.info("Attempting to create Cole_net_order from Cole_net_name")
                # Get unique networks and assign ordering
                networks = mapping_df['Cole_net_name'].unique()
                network_order = {network: i for i, network in enumerate(networks)}

                # Create a temporary ordering within each network
                mapping_df['temp_order'] = mapping_df.groupby('Cole_net_name').cumcount()

                # Combine network order and temp order to create Cole_net_order
                mapping_df['Cole_net_order'] = mapping_df.apply(
                    lambda row: network_order[row['Cole_net_name']] * 1000 + row['temp_order'],
                    axis=1
                )
                logger.info("Created Cole_net_order based on network membership")

        # Count networks and regions per network
        if 'Cole_net_name' in mapping_df.columns:
            networks = mapping_df['Cole_net_name'].value_counts().to_dict()
            networks_str = ", ".join([f"{net}: {count}" for net, count in networks.items()])
            logger.info(f"Networks distribution: {networks_str}")

        # Check if reordering will work
        if 'orig_order' in mapping_df.columns and 'Cole_net_order' in mapping_df.columns:
            logger.info("Network mapping contains both original and Cole network ordering")
            # Analyze if there are duplicate values
            orig_duplicates = mapping_df['orig_order'].duplicated().sum()
            cole_duplicates = mapping_df['Cole_net_order'].duplicated().sum()
            if orig_duplicates > 0:
                logger.warning(f"Found {orig_duplicates} duplicate values in orig_order")
            if cole_duplicates > 0:
                logger.warning(f"Found {cole_duplicates} duplicate values in Cole_net_order")

        return mapping_df

    except Exception as e:
        logger.error(f"Error loading Cole network mapping: {e}")
        logger.error(traceback.format_exc())
        return None


def find_all_subjects(data_dir, logger):
    """
    Find all subjects available in the data directory.

    Parameters:
    -----------
    data_dir : str
        Path to XCP-D output directory
    logger : Logger
        Logger object

    Returns:
    --------
    list : List of subject IDs found in the directory
    """
    logger.info(f"No subject list provided, searching for all subjects in {data_dir}")

    # Look for subject directories using both possible patterns
    subject_dirs = []

    # Pattern 1: data_dir/xcp_d/sub-*
    pattern1 = os.path.join(data_dir, 'xcp_d', 'sub-*')
    dirs1 = glob.glob(pattern1)
    subject_dirs.extend(dirs1)

    # Pattern 2: data_dir/sub-*
    pattern2 = os.path.join(data_dir, 'sub-*')
    dirs2 = glob.glob(pattern2)
    subject_dirs.extend(dirs2)

    # Extract unique subject IDs from directory paths
    subject_ids = set()

    for dir_path in subject_dirs:
        # Extract the subject ID from the path
        dir_name = os.path.basename(dir_path)
        if dir_name.startswith('sub-'):
            subject_id = dir_name[4:]  # Remove 'sub-' prefix
            subject_ids.add(subject_id)

    # Convert to sorted list
    subject_list = sorted(list(subject_ids))

    if not subject_list:
        logger.warning(f"No subjects found in {data_dir}")
    else:
        logger.info(f"Found {len(subject_list)} subjects in {data_dir}")

    return subject_list


def extract_ndar_id(subject_label):
    """
    Extract NDAR ID from subject label, with improved matching for various formats.

    Parameters:
    -----------
    subject_label : str
        Subject label (e.g., 'NDARZZ753JNTAge183MonthsStudyYear06', 'sub-NDARINV3BXPG4B7Age109MonthsStudyYear02')

    Returns:
    --------
    list : List of potential NDAR IDs to try matching
    """
    # Remove 'sub-' prefix if present
    if subject_label.startswith('sub-'):
        subject_label = subject_label[4:]

    # Extract the core NDAR ID using various patterns
    potential_ids = []

    # Pattern 1: Extract the part before "Age" if it's present
    if "Age" in subject_label:
        ndar_part = subject_label.split("Age")[0]
        # Remove any underscores - they need to be stripped out
        ndar_part = ndar_part.replace("_", "")
        potential_ids.append(ndar_part)

    # Pattern 2: Extract just the NDAR part using regex
    ndar_match = re.search(r'(NDAR[A-Za-z0-9]+)', subject_label, re.IGNORECASE)
    if ndar_match:
        ndar_id = ndar_match.group(1)
        potential_ids.append(ndar_id)

    # Pattern 3: Add variations with and without "NDAR" prefix
    for id in list(potential_ids):  # Create a copy to avoid modifying during iteration
        if id.upper().startswith("NDAR"):
            # Add version without NDAR prefix
            potential_ids.append(id[4:])
        else:
            # Add version with NDAR prefix
            potential_ids.append("NDAR" + id)

    # Add uppercase and lowercase variations
    for id in list(potential_ids):  # Create a copy to avoid modifying during iteration
        potential_ids.append(id.upper())
        potential_ids.append(id.lower())

    # Remove duplicates and return
    return list(set(potential_ids))


def load_ndar_phenotypes(ndar_file, logger):
    """Load NDAR subjects file and extract phenotype information."""
    logger.info(f"Loading phenotype data from {ndar_file}...")

    try:
        ndar_df = pd.read_csv(ndar_file)
        logger.info(f"Loaded NDAR file with {len(ndar_df)} subjects and {len(ndar_df.columns)} columns")

        # Create mapping from NDAR ID to phenotype
        phenotype_map = {}
        raw_phenotype_map = {}  # For debugging

        for _, row in ndar_df.iterrows():
            # Extract subject key from either 'subjectkey' or 'ndar_subject01_id'
            if 'subjectkey' in row and not pd.isna(row['subjectkey']):
                subject_key = str(row['subjectkey'])
            elif 'ndar_subject01_id' in row and not pd.isna(row['ndar_subject01_id']):
                subject_key = str(row['ndar_subject01_id'])
            else:
                continue  # Skip if no valid ID

            # Store original for debugging
            raw_subject_key = subject_key

            # Standardize format (remove underscore after NDAR if present)
            subject_key = subject_key.replace("_", "")

            # Ensure it starts with NDAR (case insensitive)
            if not subject_key.upper().startswith("NDAR"):
                subject_key = "NDAR" + subject_key

            # Extract age in months for matching if available
            age_months = None
            if 'interview_age' in row and not pd.isna(row['interview_age']):
                try:
                    age_months = int(row['interview_age'])
                except (ValueError, TypeError):
                    age_months = None

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
                    if phenotype < 1 or phenotype > 4:  # Validate the value
                        phenotype = None
                except (ValueError, TypeError):
                    phenotype = None

            # Store the phenotype if valid
            if phenotype is not None:
                # Store with standardized key
                phenotype_map[subject_key.upper()] = phenotype

                # Store raw entry for debugging
                raw_phenotype_map[raw_subject_key] = phenotype

                # Also store with variations to improve matching
                phenotype_map[subject_key.lower()] = phenotype
                phenotype_map[subject_key.replace("NDAR", "")] = phenotype

        # Log detailed debugging information
        logger.info(f"Found valid phenotype information for {len(raw_phenotype_map)} subjects")
        logger.info(f"Standardized to {len(phenotype_map)} phenotype entries for matching")

        # Print some sample entries for debugging
        sample_entries = list(raw_phenotype_map.items())[:5]
        logger.info(f"Sample raw entries: {sample_entries}")

        # Print distribution of phenotypes
        phenotype_counts = {
            1: sum(1 for p in raw_phenotype_map.values() if p == 1),
            2: sum(1 for p in raw_phenotype_map.values() if p == 2),
            3: sum(1 for p in raw_phenotype_map.values() if p == 3),
            4: sum(1 for p in raw_phenotype_map.values() if p == 4)
        }
        logger.info(f"Phenotype distribution: Control: {phenotype_counts[1]}, Subthreshold: {phenotype_counts[2]}, "
                    f"ADHD: {phenotype_counts[3]}, Not ADHD/Not Clean Control: {phenotype_counts[4]}")

        return phenotype_map, raw_phenotype_map

    except Exception as e:
        logger.error(f"Error loading NDAR phenotypes: {e}")
        logger.error(traceback.format_exc())
        return {}, {}


def get_phenotype_label(phenotype):
    """Convert phenotype code to text label"""
    labels = {
        1: "Control",
        2: "Subthreshold",
        3: "ADHD",
        4: "Not ADHD/Not Clean Control"
    }
    return labels.get(phenotype, "Unknown")


def find_timeseries_files(sub_label, task_label, data_dir, logger):
    """
    Find parcellated time series files for a subject with much more efficient searching.
    STRICTLY filters to only include Glasser and HCP atlas files.

    Parameters:
    -----------
    sub_label : str
        Subject label
    task_label : str
        Task label
    data_dir : str
        Path to XCP-D output directory
    logger : Logger
        Logger object

    Returns:
    --------
    list : List of time series files found, filtered to only Glasser and HCP
    """
    # Start timer for performance tracking
    start_time = datetime.now()

    # Define the most likely locations for time series files
    specific_paths = [
        # Primary XCP-D output location
        os.path.join(data_dir, f'sub-{sub_label}', 'func'),
        # Alternative location with xcp_d subdirectory
        os.path.join(data_dir, 'xcp_d', f'sub-{sub_label}', 'func')
    ]

    # Only patterns that EXPLICITLY include Glasser or HCP atlas
    patterns = [
        # Glasser atlas patterns - using different variations of how it might appear in filenames
        f"sub-{sub_label}_task-{task_label}_*_seg-Glasser_*_timeseries.ptseries.nii",
        f"sub-{sub_label}_task-{task_label}_*_atlas-Glasser_*_timeseries.ptseries.nii",
        f"sub-{sub_label}_task-{task_label}_*Glasser*_timeseries.ptseries.nii",

        # HCP atlas patterns
        f"sub-{sub_label}_task-{task_label}_*_seg-HCP_*_timeseries.ptseries.nii",
        f"sub-{sub_label}_task-{task_label}_*_atlas-HCP_*_timeseries.ptseries.nii",
        f"sub-{sub_label}_task-{task_label}_*HCP*_timeseries.ptseries.nii",
    ]

    # Try each specific path and pattern combination
    timeseries_files = []
    logger.info(f"Searching for Glasser or HCP time series files for subject {sub_label}")

    for path in specific_paths:
        if not os.path.exists(path):
            continue

        logger.debug(f"Checking directory: {path}")

        for pattern in patterns:
            full_pattern = os.path.join(path, pattern)
            files = glob.glob(full_pattern)

            if files:
                logger.info(f"Found {len(files)} files matching {pattern} in {path}")
                timeseries_files.extend(files)

    # Extra filtering step to ensure ONLY Glasser or HCP files
    filtered_by_atlas = []
    for file_path in timeseries_files:
        file_name = os.path.basename(file_path)
        # Only include if "Glasser" or "HCP" is in the filename
        if "Glasser" in file_name or "HCP" in file_name:
            filtered_by_atlas.append(file_path)
        else:
            logger.debug(f"Excluding non-Glasser/HCP file: {file_path}")

    # Replace original list with strictly filtered list
    timeseries_files = filtered_by_atlas

    # Remove duplicate files (if any)
    unique_files = list(set(timeseries_files))
    if len(timeseries_files) != len(unique_files):
        logger.info(f"Removed {len(timeseries_files) - len(unique_files)} duplicate files")
        timeseries_files = unique_files

    # Filter out combined runs (files without 'run-XX')
    run_files = []
    other_files = []

    for file_path in timeseries_files:
        # Check if this is a per-run file (contains run-XX)
        file_name = os.path.basename(file_path)
        run_match = re.search(r'run-\d+', file_name)

        if run_match:
            run_files.append(file_path)
        else:
            other_files.append(file_path)

    # If we found specific run files, use those. Otherwise use whatever we found.
    filtered_files = run_files if run_files else other_files

    # Performance measurement
    search_time = (datetime.now() - start_time).total_seconds()

    if not filtered_files:
        logger.warning(f"No Glasser or HCP time series files found for subject {sub_label}")
    else:
        logger.info(
            f"Found {len(filtered_files)} Glasser/HCP time series files for subject {sub_label} in {search_time:.2f} seconds")

        # Log files found for verification
        logger.info("Files found:")
        for file in filtered_files:
            logger.info(f"  - {file}")

    return filtered_files


def load_cifti_timeseries(file_path, phenotype_map, logger, is_hcp=False):
    try:
        logger.info(f"Loading time series data from {file_path}")
        if HAS_NIBABEL:
            cifti = nib.load(file_path)
            time_series = cifti.get_fdata()
            if is_hcp or "HCP" in file_path:
                if time_series.shape[0] > 30 and time_series.shape[1] < 30:
                    logger.info(f"HCP file detected, transposing from {time_series.shape}")
                    time_series = time_series.T
            else:
                if len(cifti.shape) > 1 and cifti.shape[0] < cifti.shape[1]:
                    logger.info(f"Transposing time series from {time_series.shape}")
                    time_series = time_series.T

            # Extract parcel names from the XML header
            xml_header = cifti.header.to_xml()
            root = ET.fromstring(xml_header)
            mim = root.find('.//MatrixIndicesMap[@IndicesMapToDataType="CIFTI_INDEX_TYPE_PARCELS"]')
            if mim is not None:
                parcels = mim.findall('.//Parcel')
                region_labels = [parcel.get('Name') for parcel in parcels if parcel.get('Name') is not None]

                # For Glasser files, identify hippocampus regions for later filtering
                hippocampus_indices = []
                if not (is_hcp or "HCP" in file_path):
                    # Keep track of hippocampus regions and remove _ROI suffix
                    region_labels_filtered = []
                    for i, label in enumerate(region_labels):
                        # Remove '_ROI' suffix for Glasser regions
                        clean_label = label.replace('_ROI', '')

                        # Check if this is a hippocampus region (L_H or R_H)
                        if clean_label in ['L_H', 'R_H'] or clean_label.endswith(
                                '_Hippocampus') or 'hippocampus' in clean_label.lower():
                            hippocampus_indices.append(i)
                            logger.info(f"Identified hippocampus region to filter: {label} (index {i})")

                        region_labels_filtered.append(clean_label)

                    region_labels = region_labels_filtered

            else:
                logger.warning("No parcel names found in CIFTI header, using generic labels")
                region_labels = [f"Region_{i}" for i in range(time_series.shape[0])]
                hippocampus_indices = []  # Empty since we're using generic labels

            if len(region_labels) != time_series.shape[0]:
                logger.warning(
                    f"Mismatch between number of labels ({len(region_labels)}) and regions ({time_series.shape[0]}), using generic labels")
                region_labels = [f"Region_{i}" for i in range(time_series.shape[0])]
                hippocampus_indices = []  # Reset since we're using generic labels

            del cifti
            gc.collect()
            logger.info(f"Extracted {len(region_labels)} region labels: {region_labels[:5]} ... {region_labels[-5:]}")
            return time_series, region_labels, hippocampus_indices
        else:
            logger.error("nibabel is required for CIFTI file loading")
            return None, None, []
    except Exception as e:
        logger.error(f"Error loading CIFTI file: {e}")
        return None, None, []


def match_and_combine_timeseries(timeseries_files, logger):
    """
    Match Glasser and HCP time series files by run and combine them.

    Parameters:
    -----------
    timeseries_files : list
        List of time series file paths
    logger : Logger
        Logger object

    Returns:
    --------
    dict : Dictionary with matched and grouped time series files by run
    """
    # Group files by run
    runs = {}

    for file_path in timeseries_files:
        file_name = os.path.basename(file_path)

        # Determine atlas type (Glasser or HCP)
        if "Glasser" in file_name:
            atlas_type = "Glasser"
        elif "HCP" in file_name:
            atlas_type = "HCP"
        else:
            logger.warning(f"Unknown atlas type for file: {file_path}")
            continue

        # Extract run information
        run_match = re.search(r'run-(\d+)', file_name)
        run_info = f"run-{run_match.group(1)}" if run_match else "run-1"  # Default to run-1 if not found

        # Initialize run entry if needed
        if run_info not in runs:
            runs[run_info] = {"Glasser": None, "HCP": None}

        # Store file path
        runs[run_info][atlas_type] = file_path

    # Filter runs to only those with both Glasser and HCP files
    combined_runs = {}
    for run_info, run_data in runs.items():
        if run_data["Glasser"] is not None and run_data["HCP"] is not None:
            combined_runs[run_info] = run_data

            # Add diagnostic logging
            glasser_file = run_data["Glasser"]
            hcp_file = run_data["HCP"]
            logger.info(f"Will combine: {os.path.basename(glasser_file)} with {os.path.basename(hcp_file)}")

    logger.info(f"Found {len(combined_runs)} runs with matching Glasser and HCP files")
    return combined_runs


def compute_correlation_matrix(time_series, logger):
    """
    Compute correlation matrix from time series data.

    Parameters:
    -----------
    time_series : numpy.ndarray
        Time series data (regions x time points)
    logger : Logger
        Logger object

    Returns:
    --------
    numpy.ndarray : Correlation matrix
    """
    try:
        logger.info(f"Computing correlation matrix from time series with shape {time_series.shape}")

        # Log memory usage before computation
        mem_usage_before = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
        logger.debug(f"Memory usage before correlation: {mem_usage_before:.2f} MB")

        # Check for NaN or Inf values in time series
        nan_count = np.isnan(time_series).sum()
        inf_count = np.isinf(time_series).sum()
        if nan_count > 0 or inf_count > 0:
            logger.warning(f"Found {nan_count} NaN and {inf_count} Inf values in time series. Cleaning data.")
            time_series = np.nan_to_num(time_series, nan=0.0, posinf=0.0, neginf=0.0)

        # Manual computation using numpy for better memory control
        # This avoids creating intermediate large arrays
        n_regions = time_series.shape[0]
        correlation_matrix = np.zeros((n_regions, n_regions), dtype=np.float32)  # Use float32 to save memory

        # Normalize each time series (subtract mean, divide by std)
        time_series_normalized = np.zeros_like(time_series, dtype=np.float32)
        for i in range(n_regions):
            ts = time_series[i]
            ts_mean = np.mean(ts)
            ts_std = np.std(ts)
            if ts_std > 0:
                time_series_normalized[i] = (ts - ts_mean) / ts_std
            else:
                time_series_normalized[i] = 0  # Handle flat lines

        # Compute correlation as dot product of normalized time series
        for i in range(n_regions):
            # Use matrix multiplication for efficiency
            correlation_matrix[i] = np.dot(time_series_normalized[i], time_series_normalized.T) / time_series.shape[1]

        # Ensure diagonal is exactly 1.0
        np.fill_diagonal(correlation_matrix, 1.0)

        # Clean up to free memory
        del time_series_normalized
        gc.collect()

        logger.info(f"Computed correlation matrix with shape {correlation_matrix.shape}")

        # Log memory usage after computation
        mem_usage_after = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB
        logger.debug(f"Memory usage after correlation: {mem_usage_after:.2f} MB")

        # Check for NaN values
        nan_count = np.isnan(correlation_matrix).sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in correlation matrix, replacing with zeros")
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0)

        return correlation_matrix

    except MemoryError:
        logger.error(f"Out of memory while computing correlation matrix")
        # Try to recover
        gc.collect()
        return None
    except Exception as e:
        logger.error(f"Error computing correlation matrix: {e}")
        logger.error(traceback.format_exc())
        return None


def apply_fisher_z_transform(correlation_matrix, logger):
    """
    Apply Fisher Z-transformation (arctanh) to correlation matrix.

    Parameters:
    -----------
    correlation_matrix : numpy.ndarray
        Correlation matrix to transform
    logger : Logger
        Logger object

    Returns:
    --------
    numpy.ndarray : Fisher Z-transformed matrix
    """
    try:
        # Clip values to avoid arctanh at boundaries (-1, 1)
        clipped_matrix = np.clip(correlation_matrix, -0.99999, 0.99999)

        # Apply arctanh (Fisher Z-transformation)
        transformed_matrix = np.arctanh(clipped_matrix)

        logger.info(f"Applied Fisher Z-transformation to correlation matrix")
        return transformed_matrix

    except Exception as e:
        logger.error(f"Error applying Fisher Z-transformation: {e}")
        logger.error(traceback.format_exc())
        return correlation_matrix  # Return original if transformation fails


def reorder_by_cole_networks(correlation_matrix, region_labels, cole_mapping, logger, atlas_type=None, combined=False):
    """
    Reorder correlation matrix based on Cole network ordering using region labels.
    """
    try:
        logger.info(f"Reordering correlation matrix by Cole networks (atlas_type={atlas_type}, combined={combined})")
        n_regions = correlation_matrix.shape[0]
        logger.info(f"Matrix has {n_regions} regions with shape {correlation_matrix.shape}")
        logger.info(f"First 5 labels: {region_labels[:5]}, Last 5 labels: {region_labels[-5:]}")

        # Labels should already match name_Glasser format
        actual_labels = region_labels.copy()

        # Create mapping from name_Glasser to Cole_net_order and network name
        name_to_cole = {}
        name_to_network = {}
        for _, row in cole_mapping.iterrows():
            name = row['name_Glasser']
            if not pd.isna(name) and not pd.isna(row['Cole_net_order']):
                cole_order = float(row['Cole_net_order'])
                network = row['Cole_net_name'] if not pd.isna(row['Cole_net_name']) else "Unknown"
                name_to_cole[name] = cole_order
                name_to_network[name] = network
        logger.info(f"Created mapping for {len(name_to_cole)} regions from name_Glasser to Cole_net_order")

        # Create reordering pairs
        sort_pairs = []
        network_assignments = []
        for i, label in enumerate(actual_labels):
            if label in name_to_cole:
                cole_order = name_to_cole[label]
                network = name_to_network[label]
            else:
                logger.warning(f"Region label {label} not found in Cole mapping, assigning to end")
                cole_order = 9999
                network = "Unknown"
            sort_pairs.append((i, cole_order))
            network_assignments.append(network)

        # Sort by Cole_net_order
        sort_pairs.sort(key=lambda x: x[1])
        sorted_indices = [pair[0] for pair in sort_pairs]

        # Reorder the matrix and labels
        reordered_matrix = correlation_matrix[sorted_indices, :][:, sorted_indices]
        reordered_labels = [actual_labels[i] for i in sorted_indices]
        sorted_networks = [network_assignments[i] for i in sorted_indices]

        # Log network distribution
        network_counts = {}
        for network in sorted_networks:
            network_counts[network] = network_counts.get(network, 0) + 1  # Fixed line
        logger.info(f"Network distribution in reordered matrix: {network_counts}")

        return reordered_matrix, reordered_labels, sorted_networks
    except Exception as e:
        logger.error(f"Error reordering matrix by Cole networks: {e}")
        return correlation_matrix, region_labels, ["Unknown"] * len(region_labels)


def plot_connectivity_matrix_cole(correlation_matrix, network_assignments, title=None, save_path=None, logger=None):
    """
    Create a brain connectivity matrix visualization with Cole lab network annotations.
    With improved title positioning and specialized handling for subcortical matrices.
    """
    try:
        if logger:
            logger.info("Creating Cole network connectivity visualization")

        # Define network colors
        manual_network_colors = {
            'Visual1': '#000023',  # Dark Blue
            'Visual2': '#1E90FF',  # Dodger Blue
            'Somatomotor': '#32CD32',  # Lime Green
            'Cingulo-Opercular': '#FFD700',  # Gold
            'Dorsal_Attention': '#12670f',  # Dark Green
            'Language': '#FF69B4',  # Hot Pink
            'Frontoparietal': '#FF6347',  # Tomato
            'Auditory': '#00CED1',  # Dark Turquoise
            'Posterior_Multimodal': '#9400D3',  # Dark Violet
            'Default': '#800000',  # Maroon
            'Orbito-Affective': '#FF8C00',  # Dark Orange
            'Ventral_Multimodal': '#8B4513',  # Saddle Brown
            'Subcortex': '#808080',  # Gray for subcortical regions
            'Unknown': '#CCCCCC'  # Light Gray for unknown regions
        }

        # Check if this is a small subcortical matrix (HCP)
        is_subcortical = correlation_matrix.shape[0] < 30 and all(
            net == 'Subcortex' for net in network_assignments if net is not None)
        matrix_size = correlation_matrix.shape[0]

        # Specialized handling for subcortical matrices
        if is_subcortical:
            # Create a specialized figure for subcortical matrices
            fig = plt.figure(figsize=(8, 6))

            # Add title first, BEFORE creating the subplot
            if title:
                fig.suptitle(title, fontsize=14, y=0.98)

            # Add a single axis that takes most of the figure space with minimal margins
            ax = fig.add_subplot(111)

            # Set diagonal to 0 for better visualization
            masked_matrix = correlation_matrix.copy()
            np.fill_diagonal(masked_matrix, 0)

            # Plot the heatmap with no color bars on the side (they're not needed for all-subcortical)
            sns.heatmap(
                masked_matrix,
                cmap='bwr',
                vmin=-1.0,
                vmax=1.0,
                xticklabels=False,
                yticklabels=False,
                ax=ax,
                cbar_kws={'label': 'Connectivity Strength (Fisher Z)'}
            )

            # Add a simple label on the left
            ax.text(-0.15, 0.5, 'Subcortex',
                    va='center', ha='right', rotation=90,
                    transform=ax.transAxes, fontsize=12,
                    color=manual_network_colors.get('Subcortex', '#808080'))
        else:
            # Original handling for larger matrices
            if matrix_size < 30:  # Small but not all subcortical
                fig = plt.figure(figsize=(8, 6))
                margin_size = -8
                text_offset = 2
            else:
                # For large matrices, especially the combined ones, add more top margin
                fig = plt.figure(figsize=(12, 11))  # Increased height for title space
                margin_size = -17
                text_offset = 5

            # Add title BEFORE creating subplot, with properly positioned y-coordinate
            # This prevents the title from overlapping with the network color bar
            if title:
                fig.suptitle(title, fontsize=16, y=0.98)

            # Create subplot with proper positioning to leave room for title
            if matrix_size >= 200:  # Combined matrices need more space for title
                ax = fig.add_subplot(111)
                plt.subplots_adjust(top=0.92)  # Leave more space at top for title
            else:
                ax = fig.add_subplot(111)

            # Set diagonal to 0 for better visualization
            masked_matrix = correlation_matrix.copy()
            np.fill_diagonal(masked_matrix, 0)

            # Plot the heatmap
            sns.heatmap(
                masked_matrix,
                cmap='bwr',
                vmin=-1.0,
                vmax=1.0,
                xticklabels=False,
                yticklabels=False,
                ax=ax
            )

            # Handle missing or incorrect network_assignments
            if network_assignments is None or len(network_assignments) != correlation_matrix.shape[0]:
                if logger:
                    logger.warning(f"Network assignments missing or incorrect size. Creating defaults.")
                network_assignments = ["Unknown"] * correlation_matrix.shape[0]

            # Create color bars based on network assignments
            top_colors = [manual_network_colors.get(net, '#CCCCCC') for net in network_assignments]
            side_colors = top_colors.copy()

            # Add color bar to the top
            for i, color in enumerate(top_colors):
                ax.add_patch(patches.Rectangle((i, margin_size), 1, abs(margin_size) - 2,
                                               facecolor=color, transform=ax.transData,
                                               clip_on=False, linewidth=0))

            # Add color bar to the left
            for i, color in enumerate(side_colors):
                ax.add_patch(patches.Rectangle((margin_size, i), abs(margin_size) - 2, 1,
                                               facecolor=color, transform=ax.transData,
                                               clip_on=False, linewidth=0))

            # Find network boundaries for labeling
            network_boundaries = []
            current_network = None
            for i, network in enumerate(network_assignments):
                if network != current_network:
                    network_boundaries.append((i, network))
                    current_network = network

            # Add network labels
            for i, (boundary, network) in enumerate(network_boundaries):
                if network is not None:
                    color = manual_network_colors.get(network, '#CCCCCC')
                    ax.text(margin_size - 2, boundary + text_offset, network, ha='right', fontsize=10,
                            color=color, transform=ax.transData)

        # Save or show the figure
        if save_path:
            # Use a layout that gives more space to the title
            if matrix_size >= 200:  # Combined matrices
                plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave top 8% for title
            else:
                plt.tight_layout(rect=[0, 0, 1, 0.95])  # Normal spacing

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if logger:
                logger.info(f"Saved connectivity visualization to {save_path}")
            plt.close(fig)
        else:
            if matrix_size >= 200:
                plt.tight_layout(rect=[0, 0, 1, 0.92])
            else:
                plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
            plt.close(fig)

        return fig

    except Exception as e:
        if logger:
            logger.error(f"Error creating Cole network visualization: {e}")
            logger.error(traceback.format_exc())
        return None


def process_subject(sub_label, task_label, data_dir, timeseries_file, output_dir, timeseries_files, cole_mapping, phenotype_map, raw_phenotype_map,
                    generate_subject_vis, logger):
    """
    Process a single subject's connectivity data from time series.
    """
    subject_data = {"combined": [], "separate": []}
    try:
        # Get subject phenotype if available with improved matching
        subject_phenotype = None

        # Extract potential NDAR IDs for matching
        potential_ndar_ids = extract_ndar_id(sub_label)

        # Log the matching attempt for debugging
        logger.info(f"Trying to match subject {sub_label} with extracted NDAR IDs: {potential_ndar_ids}")

        # Try matching with each potential ID
        for ndar_id in potential_ndar_ids:
            if ndar_id in phenotype_map:
                subject_phenotype = phenotype_map[ndar_id]
                logger.info(
                    f"Found phenotype for subject {sub_label} using ID {ndar_id}: {get_phenotype_label(subject_phenotype)} (Code: {subject_phenotype})")
                break

        if subject_phenotype is None:
            logger.info(f"No phenotype match found for subject {sub_label}")

        # Find potential timepoint information from filename patterns
        run_matches = [re.search(r'run-(\d+)', os.path.basename(f)) for f in timeseries_files]
        age_months = None

        # Try to extract age information from filenames or metadata
        for file_path in timeseries_files:
            file_name = os.path.basename(file_path)
            # Check for age pattern in filename
            age_match = re.search(r'Age(\d+)Months', file_name)
            if age_match:
                try:
                    age_months = int(age_match.group(1))
                    break
                except (ValueError, TypeError):
                    pass

        # Check if this specific timepoint should be excluded
        if should_exclude_timepoint(sub_label, age_months=age_months):
            logger.info(f"Excluding timepoint for subject {sub_label} based on specific timepoint exclusion criteria")
            return {sub_label: subject_data}

        # Continue with the rest of the function
        logger.info(f"Processing subject {sub_label}")
        timeseries_files = find_timeseries_files(sub_label, task_label, data_dir, logger)

        # Filter to only include files with '_run-' in them
        run_files = [f for f in timeseries_files if '_run-' in os.path.basename(f)]
        if run_files:
            logger.info(f"Found {len(run_files)} run-specific files out of {len(timeseries_files)} total files")
            timeseries_files = run_files
        else:
            logger.warning(
                f"No run-specific files found for subject {sub_label}, using all {len(timeseries_files)} files")

        if not timeseries_files:
            logger.warning(f"No time series files found for subject {sub_label}")
            return {sub_label: subject_data}

        matched_runs = match_and_combine_timeseries(timeseries_files, logger)

        # For averaging matrices later
        all_combined_data = []

        for run_info, run_data in matched_runs.items():
            glasser_file = run_data["Glasser"]
            hcp_file = run_data["HCP"]

            # Load Glasser data with hippocampus indices identification
            glasser_time_series, glasser_labels, glasser_hippo_indices = load_cifti_timeseries(
                glasser_file, phenotype_map, logger, is_hcp=False)
            if glasser_time_series is None:
                continue

            # Load HCP data (hippocampus regions should be kept here)
            hcp_time_series, hcp_labels, _ = load_cifti_timeseries(
                hcp_file, phenotype_map, logger, is_hcp=True)
            if hcp_time_series is None:
                continue

            # Ensure HCP data is properly oriented
            if hcp_time_series.shape[0] > 20 and hcp_time_series.shape[1] < 30:
                hcp_time_series = hcp_time_series.T

            # Filter out hippocampus regions from Glasser data if any were found
            if glasser_hippo_indices:
                logger.info(
                    f"Filtering {len(glasser_hippo_indices)} hippocampus regions from Glasser data for subject {sub_label}")
                # Create mask to keep non-hippocampus regions
                keep_mask = np.ones(glasser_time_series.shape[0], dtype=bool)
                keep_mask[glasser_hippo_indices] = False

                # Filter timeseries and labels
                glasser_time_series = glasser_time_series[keep_mask, :]
                filtered_glasser_labels = [label for i, label in enumerate(glasser_labels)
                                           if i not in glasser_hippo_indices]

                logger.info(
                    f"After filtering: Glasser data has {glasser_time_series.shape[0]} regions (removed L_H and R_H)")

                # Update the labels list after filtering
                glasser_labels = filtered_glasser_labels

            # Process Glasser data for separate analysis
            glasser_matrix = compute_correlation_matrix(glasser_time_series, logger)
            if glasser_matrix is not None:
                glasser_fisher_z = apply_fisher_z_transform(glasser_matrix, logger)
                glasser_reordered, glasser_labels_reordered, glasser_networks = reorder_by_cole_networks(
                    glasser_fisher_z, glasser_labels, cole_mapping, logger, atlas_type="Glasser"
                )
                subject_data["separate"].append({
                    "matrix": glasser_reordered, "labels": glasser_labels_reordered, "networks": glasser_networks,
                    "run_info": run_info, "file": glasser_file, "phenotype": subject_phenotype, "atlas": "Glasser",
                    "matrix_size": glasser_reordered.shape[0]
                })

            # Process HCP data for separate analysis
            hcp_matrix = compute_correlation_matrix(hcp_time_series, logger)
            if hcp_matrix is not None:
                hcp_fisher_z = apply_fisher_z_transform(hcp_matrix, logger)
                hcp_reordered, hcp_labels_reordered, hcp_networks = reorder_by_cole_networks(
                    hcp_fisher_z, hcp_labels, cole_mapping, logger, atlas_type="HCP"
                )
                subject_data["separate"].append({
                    "matrix": hcp_reordered, "labels": hcp_labels_reordered, "networks": hcp_networks,
                    "run_info": run_info, "file": hcp_file, "phenotype": subject_phenotype, "atlas": "HCP",
                    "matrix_size": hcp_reordered.shape[0]
                })

            # Combine Glasser and HCP data (now with hippocampus regions filtered from Glasser)
            min_time_points = min(glasser_time_series.shape[1], hcp_time_series.shape[1])
            if min_time_points < 60:
                logger.warning(f"Insufficient time points ({min_time_points}) for robust correlation")
                continue

            # Create DataFrames and combine
            glasser_df = pd.DataFrame(glasser_time_series.T, columns=glasser_labels)
            hcp_df = pd.DataFrame(hcp_time_series.T, columns=hcp_labels)
            glasser_df.index = range(glasser_df.shape[0])
            hcp_df.index = range(hcp_df.shape[0])
            combined_df = pd.concat([glasser_df, hcp_df], axis=1).fillna(0)
            combined_df = combined_df.iloc[:min_time_points, :]

            # Store original column headers for later use in CSV output
            combined_headers = list(combined_df.columns)
            logger.debug(f"Combined headers: first 5: {combined_headers[:5]}, last 5: {combined_headers[-5:]}")

            combined_array = combined_df.values.T
            combined_labels = glasser_labels + hcp_labels

            # Compute combined correlation matrix
            combined_matrix = compute_correlation_matrix(combined_array, logger)
            if combined_matrix is not None:
                fisher_z_matrix = apply_fisher_z_transform(combined_matrix, logger)
                reordered_matrix, reordered_labels, network_assignments = reorder_by_cole_networks(
                    fisher_z_matrix, combined_labels, cole_mapping, logger, combined=True
                )

                # Store data for this run
                run_data = {
                    "matrix": reordered_matrix,
                    "labels": reordered_labels,
                    "original_labels": combined_headers,
                    "networks": network_assignments,
                    "run_info": run_info,
                    "files": [glasser_file, hcp_file],
                    "phenotype": subject_phenotype,
                    "matrix_size": reordered_matrix.shape[0]
                }

                subject_data["combined"].append(run_data)
                all_combined_data.append(run_data)

                # Generate visualization if requested
                if generate_subject_vis:
                    subject_dir = os.path.join(output_dir, f"sub-{sub_label}")
                    os.makedirs(subject_dir, exist_ok=True)
                    title = f"{sub_label} - {run_info} - Combined Connectivity (Fisher Z)"
                    output_path = os.path.join(subject_dir, f"{sub_label}_{run_info}_combined_connectivity.png")
                    plot_connectivity_matrix_cole(reordered_matrix, network_assignments, title, output_path, logger)

                # Export individual run CSV if requested
                if generate_subject_vis:
                    subject_dir = os.path.join(output_dir, f"sub-{sub_label}")
                    os.makedirs(subject_dir, exist_ok=True)
                    csv_path = os.path.join(subject_dir, f"{sub_label}_{run_info}_combined_connectivity.csv")

                    matrix_df = pd.DataFrame(
                        reordered_matrix,
                        index=reordered_labels,
                        columns=reordered_labels
                    )
                    matrix_df.to_csv(csv_path)
                    logger.info(f"Saved individual run connectivity matrix to {csv_path}")

        # Average matrices across all runs for this subject
        if all_combined_data:
            # Initialize with first run's data
            first_run = all_combined_data[0]
            summed_matrix = first_run["matrix"].copy()
            labels = first_run["labels"]
            networks = first_run["networks"]

            # Add matrices from other runs
            for run_data in all_combined_data[1:]:
                # Check if matrices have the same shape
                if summed_matrix.shape == run_data["matrix"].shape:
                    summed_matrix += run_data["matrix"]
                else:
                    logger.warning(
                        f"Matrix shape mismatch: {summed_matrix.shape} vs {run_data['matrix'].shape}. Skipping this run for averaging.")

            # Calculate average
            avg_matrix = summed_matrix / len(all_combined_data)

            # Create subject directory for the averaged matrix
            subject_dir = os.path.join(output_dir, f"sub-{sub_label}")
            os.makedirs(subject_dir, exist_ok=True)

            # Extract the core ID
            subject_core = normalize_subject_id(sub_label)

            # Extract age information
            age_match = re.search(r'Age(\d+)Months', sub_label)
            age_str = age_match.group(1) if age_match else "unknown"

            # Format filename to match g-score format
            gscore_format_id = f"{subject_core}_Age{age_str}"
            csv_path = os.path.join(subject_dir, f"{gscore_format_id}_averaged_combined_connectivity.csv")

            # Create DataFrame with proper region labels
            matrix_df = pd.DataFrame(
                avg_matrix,
                index=labels,
                columns=labels
            )
            matrix_df.to_csv(csv_path)

            logger.info(
                f"Created averaged connectivity matrix from {len(all_combined_data)} runs for subject {sub_label}")
            logger.info(f"Saved ML-ready averaged connectivity matrix to {csv_path}")

            # Store averaged matrix in subject data
            subject_data["averaged_matrix"] = {
                "matrix": avg_matrix,
                "labels": labels,
                "networks": networks,
                "phenotype": subject_phenotype,
                "num_runs_averaged": len(all_combined_data)
            }
        else:
            logger.warning(f"No valid runs found for subject {sub_label}, cannot create averaged matrix")

        return {sub_label: subject_data}
    except Exception as e:
        logger.error(f"Error processing subject {sub_label}: {e}")
        logger.error(traceback.format_exc())
        return {sub_label: subject_data}


def aggregate_matrices_by_phenotype(subject_data, phenotype_map, logger):
    """
    Aggregate correlation matrices by phenotype.
    """
    # Initialize result structure
    combined_aggregated = {
        "phenotypes": {},
        "all": {
            "matrix_sum": None,
            "count": 0,
            "network_assignments": None,
            "atlas_designations": None
        }
    }

    separate_aggregated = {}

    # Initialize counters for tracking
    counts = {
        "total": 0,
        "with_phenotype": 0,
        "combined_matrices": 0,
        "by_phenotype": {},  # Use a dictionary that will automatically handle any phenotype value
        "by_atlas": {}  # Track counts by atlas type
    }

    # Process each subject
    for subject_id, subject_data_by_type in subject_data.items():
        counts["total"] += 1

        # Process combined matrices first (these include cortical-subcortical correlations)
        if "combined" in subject_data_by_type and subject_data_by_type["combined"]:
            for ts_data in subject_data_by_type["combined"]:
                if "matrix" not in ts_data or ts_data["matrix"] is None:
                    continue

                matrix = ts_data["matrix"]
                phenotype = ts_data.get("phenotype", None)
                network_assignments = ts_data.get("networks", None)
                atlas_designations = ts_data.get("atlas_designations", None)
                matrix_size = matrix.shape[0]  # Should be consistent for combined matrices

                # Debug log for each combined matrix
                logger.info(
                    f"Adding combined matrix with size {matrix_size} from subject {subject_id}, phenotype: {phenotype}")

                # Track first valid network assignments and atlas designations
                if network_assignments is not None and combined_aggregated["all"]["network_assignments"] is None:
                    combined_aggregated["all"]["network_assignments"] = network_assignments

                if atlas_designations is not None and combined_aggregated["all"]["atlas_designations"] is None:
                    combined_aggregated["all"]["atlas_designations"] = atlas_designations

                counts["combined_matrices"] += 1

                if phenotype is not None:
                    counts["with_phenotype"] += 1

                    # Ensure the phenotype is in our counter dictionary
                    if phenotype not in counts["by_phenotype"]:
                        counts["by_phenotype"][phenotype] = 0
                    counts["by_phenotype"][phenotype] += 1

                    # Initialize phenotype entry if needed
                    if phenotype not in combined_aggregated["phenotypes"]:
                        combined_aggregated["phenotypes"][phenotype] = {
                            "matrix_sum": None,
                            "count": 0,
                            "label": get_phenotype_label(phenotype)
                        }

                    # Add to phenotype-specific average
                    if combined_aggregated["phenotypes"][phenotype]["matrix_sum"] is None:
                        combined_aggregated["phenotypes"][phenotype]["matrix_sum"] = matrix.copy()
                    else:
                        # Check matrix sizes match before adding
                        if combined_aggregated["phenotypes"][phenotype]["matrix_sum"].shape == matrix.shape:
                            combined_aggregated["phenotypes"][phenotype]["matrix_sum"] += matrix
                        else:
                            logger.warning(f"Matrix size mismatch in combined phenotype {phenotype}: "
                                           f"{combined_aggregated['phenotypes'][phenotype]['matrix_sum'].shape} vs {matrix.shape}")

                            # Try to resize matrices to match
                            try:
                                # Use the smaller dimensions
                                min_rows = min(combined_aggregated["phenotypes"][phenotype]["matrix_sum"].shape[0],
                                               matrix.shape[0])
                                min_cols = min(combined_aggregated["phenotypes"][phenotype]["matrix_sum"].shape[1],
                                               matrix.shape[1])

                                # Truncate both matrices to the same size
                                truncated_sum = combined_aggregated["phenotypes"][phenotype]["matrix_sum"][:min_rows,
                                                :min_cols]
                                truncated_new = matrix[:min_rows, :min_cols]

                                # Replace the matrix sum with the truncated version
                                combined_aggregated["phenotypes"][phenotype][
                                    "matrix_sum"] = truncated_sum + truncated_new
                                logger.info(f"Resized matrices to {min_rows}x{min_cols} for addition")
                            except Exception as e:
                                logger.error(f"Error resizing matrices: {e}")
                                continue

                    combined_aggregated["phenotypes"][phenotype]["count"] += 1

                # Add to overall average
                if combined_aggregated["all"]["matrix_sum"] is None:
                    combined_aggregated["all"]["matrix_sum"] = matrix.copy()
                else:
                    # Check matrix sizes match before adding
                    if combined_aggregated["all"]["matrix_sum"].shape == matrix.shape:
                        combined_aggregated["all"]["matrix_sum"] += matrix
                    else:
                        logger.warning(f"Matrix size mismatch in combined: "
                                       f"{combined_aggregated['all']['matrix_sum'].shape} vs {matrix.shape}")

                        # Try to resize matrices to match
                        try:
                            # Use the smaller dimensions
                            min_rows = min(combined_aggregated["all"]["matrix_sum"].shape[0], matrix.shape[0])
                            min_cols = min(combined_aggregated["all"]["matrix_sum"].shape[1], matrix.shape[1])

                            # Truncate both matrices to the same size
                            truncated_sum = combined_aggregated["all"]["matrix_sum"][:min_rows, :min_cols]
                            truncated_new = matrix[:min_rows, :min_cols]

                            # Replace the matrix sum with the truncated version
                            combined_aggregated["all"]["matrix_sum"] = truncated_sum + truncated_new
                            logger.info(f"Resized matrices to {min_rows}x{min_cols} for addition in all")
                        except Exception as e:
                            logger.error(f"Error resizing matrices: {e}")
                            continue

                combined_aggregated["all"]["count"] += 1

        # Process separate matrices (individual atlas types)
        if "separate" in subject_data_by_type and subject_data_by_type["separate"]:
            for ts_data in subject_data_by_type["separate"]:
                if "matrix" not in ts_data or ts_data["matrix"] is None:
                    continue

                matrix = ts_data["matrix"]
                phenotype = ts_data.get("phenotype", None)
                network_assignments = ts_data.get("networks", None)
                atlas = ts_data.get("atlas", "Unknown")
                matrix_size = ts_data.get("matrix_size", matrix.shape[0])

                # Create a unique key for this atlas (without size to avoid fragmentation)
                atlas_key = f"{atlas}"

                # Count by atlas
                if atlas not in counts["by_atlas"]:
                    counts["by_atlas"][atlas] = 0
                counts["by_atlas"][atlas] += 1

                # Initialize atlas entry in separate_aggregated if needed
                if atlas_key not in separate_aggregated:
                    separate_aggregated[atlas_key] = {
                        "phenotypes": {},
                        "all": {
                            "matrix_sum": None,
                            "count": 0,
                            "network_assignments": None
                        },
                        "atlas": atlas
                    }

                # Track first valid network assignments for each atlas type
                if network_assignments is not None and separate_aggregated[atlas_key]["all"][
                    "network_assignments"] is None:
                    separate_aggregated[atlas_key]["all"]["network_assignments"] = network_assignments

                if phenotype is not None:
                    # Initialize phenotype entry if needed
                    if phenotype not in separate_aggregated[atlas_key]["phenotypes"]:
                        separate_aggregated[atlas_key]["phenotypes"][phenotype] = {
                            "matrix_sum": None,
                            "count": 0,
                            "label": get_phenotype_label(phenotype)
                        }

                    # Add to phenotype-specific average
                    if separate_aggregated[atlas_key]["phenotypes"][phenotype]["matrix_sum"] is None:
                        separate_aggregated[atlas_key]["phenotypes"][phenotype]["matrix_sum"] = matrix.copy()
                    else:
                        # Check matrix sizes match before adding
                        if separate_aggregated[atlas_key]["phenotypes"][phenotype]["matrix_sum"].shape == matrix.shape:
                            separate_aggregated[atlas_key]["phenotypes"][phenotype]["matrix_sum"] += matrix
                        else:
                            logger.warning(f"Matrix size mismatch in {atlas} phenotype {phenotype}: "
                                           f"{separate_aggregated[atlas_key]['phenotypes'][phenotype]['matrix_sum'].shape} vs {matrix.shape}")

                            # Try to resize matrices to match
                            try:
                                # Use the smaller dimensions
                                min_rows = min(
                                    separate_aggregated[atlas_key]["phenotypes"][phenotype]["matrix_sum"].shape[0],
                                    matrix.shape[0])
                                min_cols = min(
                                    separate_aggregated[atlas_key]["phenotypes"][phenotype]["matrix_sum"].shape[1],
                                    matrix.shape[1])

                                # Truncate both matrices to the same size
                                truncated_sum = separate_aggregated[atlas_key]["phenotypes"][phenotype]["matrix_sum"][
                                                :min_rows, :min_cols]
                                truncated_new = matrix[:min_rows, :min_cols]

                                # Replace the matrix sum with the truncated version
                                separate_aggregated[atlas_key]["phenotypes"][phenotype][
                                    "matrix_sum"] = truncated_sum + truncated_new
                                logger.info(
                                    f"Resized matrices to {min_rows}x{min_cols} for addition in {atlas} phenotype {phenotype}")
                            except Exception as e:
                                logger.error(f"Error resizing matrices: {e}")
                                continue

                    separate_aggregated[atlas_key]["phenotypes"][phenotype]["count"] += 1

                # Add to overall average for this atlas
                if separate_aggregated[atlas_key]["all"]["matrix_sum"] is None:
                    separate_aggregated[atlas_key]["all"]["matrix_sum"] = matrix.copy()
                else:
                    # Check matrix sizes match before adding
                    if separate_aggregated[atlas_key]["all"]["matrix_sum"].shape == matrix.shape:
                        separate_aggregated[atlas_key]["all"]["matrix_sum"] += matrix
                    else:
                        logger.warning(f"Matrix size mismatch in {atlas}: "
                                       f"{separate_aggregated[atlas_key]['all']['matrix_sum'].shape} vs {matrix.shape}")

                        # Try to resize matrices to match
                        try:
                            # Use the smaller dimensions
                            min_rows = min(separate_aggregated[atlas_key]["all"]["matrix_sum"].shape[0],
                                           matrix.shape[0])
                            min_cols = min(separate_aggregated[atlas_key]["all"]["matrix_sum"].shape[1],
                                           matrix.shape[1])

                            # Truncate both matrices to the same size
                            truncated_sum = separate_aggregated[atlas_key]["all"]["matrix_sum"][:min_rows, :min_cols]
                            truncated_new = matrix[:min_rows, :min_cols]

                            # Replace the matrix sum with the truncated version
                            separate_aggregated[atlas_key]["all"]["matrix_sum"] = truncated_sum + truncated_new
                            logger.info(f"Resized matrices to {min_rows}x{min_cols} for addition in {atlas}")
                        except Exception as e:
                            logger.error(f"Error resizing matrices: {e}")
                            continue

                separate_aggregated[atlas_key]["all"]["count"] += 1

    # Calculate averages for combined matrices
    if combined_aggregated["all"]["count"] > 0:
        combined_aggregated["all"]["matrix_avg"] = combined_aggregated["all"]["matrix_sum"] / \
                                                   combined_aggregated["all"]["count"]

    # Phenotype-specific averages for combined matrices
    for phenotype in combined_aggregated["phenotypes"]:
        if combined_aggregated["phenotypes"][phenotype]["count"] > 0:
            combined_aggregated["phenotypes"][phenotype]["matrix_avg"] = (
                    combined_aggregated["phenotypes"][phenotype]["matrix_sum"] /
                    combined_aggregated["phenotypes"][phenotype]["count"]
            )
            combined_aggregated["phenotypes"][phenotype]["network_assignments"] = combined_aggregated["all"][
                "network_assignments"]
            combined_aggregated["phenotypes"][phenotype]["atlas_designations"] = combined_aggregated["all"][
                "atlas_designations"]

    # Calculate averages for separate matrices by atlas
    for atlas_key in separate_aggregated:
        # Overall average
        if separate_aggregated[atlas_key]["all"]["count"] > 0:
            separate_aggregated[atlas_key]["all"]["matrix_avg"] = separate_aggregated[atlas_key]["all"]["matrix_sum"] / \
                                                                  separate_aggregated[atlas_key]["all"]["count"]

        # Phenotype-specific averages
        for phenotype in separate_aggregated[atlas_key]["phenotypes"]:
            if separate_aggregated[atlas_key]["phenotypes"][phenotype]["count"] > 0:
                separate_aggregated[atlas_key]["phenotypes"][phenotype]["matrix_avg"] = (
                        separate_aggregated[atlas_key]["phenotypes"][phenotype]["matrix_sum"] /
                        separate_aggregated[atlas_key]["phenotypes"][phenotype]["count"]
                )
                separate_aggregated[atlas_key]["phenotypes"][phenotype]["network_assignments"] = \
                    separate_aggregated[atlas_key]["all"]["network_assignments"]

    # Log summary
    logger.info(f"Processed {counts['total']} subjects for aggregation")
    logger.info(f"Found {counts['combined_matrices']} combined matrices (with cortical-subcortical correlations)")
    logger.info(f"Found phenotypes for {counts['with_phenotype']} subjects")

    # Log phenotype distribution
    for phenotype, count in counts["by_phenotype"].items():
        if phenotype is not None:
            logger.info(f"  {get_phenotype_label(phenotype)}: {count} subjects")
        else:
            logger.info(f"  Unknown phenotype: {count} subjects")

    # Log atlas distribution
    logger.info("Atlas distribution:")
    for atlas, count in counts["by_atlas"].items():
        logger.info(f"  {atlas}: {count} matrices")

    # Return both aggregation results
    return {
        "combined": combined_aggregated,
        "separate": separate_aggregated
    }


def visualize_group_matrices(aggregated, output_dir, logger):
    """
    Create visualizations of group-level correlation matrices.
    """
    # Create group directory
    group_dir = os.path.join(output_dir, "group_results")
    os.makedirs(group_dir, exist_ok=True)

    # Create a directory for combined visualizations
    combined_dir = os.path.join(group_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)

    # Process combined matrices first (with cortical-subcortical correlations)
    combined_aggregated = aggregated.get("combined", {})
    if combined_aggregated and combined_aggregated["all"]["count"] > 0:
        logger.info(f"Creating group visualizations for combined cortical-subcortical matrices")

        # Create overall average visualization
        if "matrix_avg" in combined_aggregated["all"]:
            matrix = combined_aggregated["all"]["matrix_avg"]
            count = combined_aggregated["all"]["count"]
            network_assignments = combined_aggregated["all"]["network_assignments"]
            atlas_designations = combined_aggregated["all"]["atlas_designations"]

            # Create connectivity matrix visualization
            title = f"Group Average Combined Connectivity (n={count})"
            output_path = os.path.join(combined_dir, "group_average_combined_connectivity.png")

            plot_connectivity_matrix_cole(
                matrix,
                network_assignments,
                title=title,
                save_path=output_path,
                logger=logger
            )

            logger.info(f"Saved group average combined visualization to {output_path}")

            # Extract and visualize just the cortical-subcortical block
            if atlas_designations:
                # Identify cortical and subcortical regions
                is_cortical = np.array([designation == 'Glasser' for designation in atlas_designations])
                is_subcortical = np.array([designation == 'HCP' for designation in atlas_designations])

                # Extract the cortical-subcortical block
                cortico_subcortical = matrix[is_cortical, :][:, is_subcortical]

                # Create visualization for cortico-subcortical block
                plt.figure(figsize=(10, 8))
                plt.imshow(cortico_subcortical, cmap='bwr', vmin=-1, vmax=1)
                plt.colorbar(label='Correlation (Fisher Z)')
                plt.title(f"Group Average Cortical-Subcortical Correlations (n={count})")
                plt.xlabel('Subcortical Regions')
                plt.ylabel('Cortical Regions')
                plt.tight_layout()

                cs_output_path = os.path.join(combined_dir, "group_average_cortico_subcortical.png")
                plt.savefig(cs_output_path, dpi=300)
                plt.close()

                logger.info(f"Saved group average cortical-subcortical block to {cs_output_path}")

            # Save as CSV for further analysis
            csv_path = os.path.join(combined_dir, "group_average_combined_connectivity.csv")
            pd.DataFrame(matrix).to_csv(csv_path)
            logger.info(f"Saved group average combined matrix to {csv_path}")

        # Create phenotype-specific visualizations
        for phenotype, pheno_data in combined_aggregated["phenotypes"].items():
            if phenotype is None:
                pheno_label = "Unknown"
            else:
                pheno_label = get_phenotype_label(phenotype)

            if "matrix_avg" in pheno_data:
                matrix = pheno_data["matrix_avg"]
                count = pheno_data["count"]
                network_assignments = pheno_data.get("network_assignments",
                                                     combined_aggregated["all"]["network_assignments"])
                atlas_designations = pheno_data.get("atlas_designations",
                                                    combined_aggregated["all"]["atlas_designations"])

                if count == 0:
                    continue

                # Create connectivity matrix visualization
                title = f"Group Average Combined Connectivity - {pheno_label} (n={count})"
                output_path = os.path.join(combined_dir, f"phenotype_{phenotype}_combined_connectivity.png")

                plot_connectivity_matrix_cole(
                    matrix,
                    network_assignments,
                    title=title,
                    save_path=output_path,
                    logger=logger
                )

                logger.info(f"Saved {pheno_label} combined average visualization to {output_path}")

                # Extract and visualize just the cortical-subcortical block
                if atlas_designations:
                    # Identify cortical and subcortical regions
                    is_cortical = np.array([designation == 'Glasser' for designation in atlas_designations])
                    is_subcortical = np.array([designation == 'HCP' for designation in atlas_designations])

                    # Extract the cortical-subcortical block
                    cortico_subcortical = matrix[is_cortical, :][:, is_subcortical]

                    # Create visualization for cortico-subcortical block
                    plt.figure(figsize=(10, 8))
                    plt.imshow(cortico_subcortical, cmap='bwr', vmin=-1, vmax=1)
                    plt.colorbar(label='Correlation (Fisher Z)')
                    plt.title(f"{pheno_label} Group Average Cortical-Subcortical Correlations (n={count})")
                    plt.xlabel('Subcortical Regions')
                    plt.ylabel('Cortical Regions')
                    plt.tight_layout()

                    cs_output_path = os.path.join(combined_dir, f"phenotype_{phenotype}_cortico_subcortical.png")
                    plt.savefig(cs_output_path, dpi=300)
                    plt.close()

                    logger.info(f"Saved {pheno_label} cortical-subcortical block to {cs_output_path}")

                # Save as CSV for further analysis
                csv_path = os.path.join(combined_dir, f"phenotype_{phenotype}_combined_connectivity.csv")
                pd.DataFrame(matrix).to_csv(csv_path)
                logger.info(f"Saved {pheno_label} combined average matrix to {csv_path}")

        # Create difference maps between phenotypes (e.g., ADHD vs Control)
        create_phenotype_difference_maps_combined(combined_aggregated, combined_dir, logger)

    # Process separate matrices by atlas type (not by size)
    separate_aggregated = aggregated.get("separate", {})
    if separate_aggregated:
        # Process each atlas_key separately (atlas without size)
        for atlas_key, atlas_data in separate_aggregated.items():
            atlas = atlas_data.get("atlas", "Unknown")

            logger.info(f"Creating group visualizations for {atlas}")

            # Create atlas-specific directory (without size in the path)
            atlas_dir = os.path.join(group_dir, f"{atlas}")
            os.makedirs(atlas_dir, exist_ok=True)

            # Create overall average visualization
            if "matrix_avg" in atlas_data["all"]:
                matrix = atlas_data["all"]["matrix_avg"]
                count = atlas_data["all"]["count"]
                network_assignments = atlas_data["all"]["network_assignments"]

                # Create connectivity matrix visualization
                title = f"Group Average Connectivity - {atlas} (n={count})"
                output_path = os.path.join(atlas_dir, f"group_average_{atlas}_connectivity.png")

                plot_connectivity_matrix_cole(
                    matrix,
                    network_assignments,
                    title=title,
                    save_path=output_path,
                    logger=logger
                )

                logger.info(f"Saved group average {atlas} visualization to {output_path}")

                # Save as CSV for further analysis
                csv_path = os.path.join(atlas_dir, f"group_average_{atlas}_connectivity.csv")
                pd.DataFrame(matrix).to_csv(csv_path)
                logger.info(f"Saved group average {atlas} matrix to {csv_path}")

            # Create phenotype-specific visualizations
            for phenotype, pheno_data in atlas_data["phenotypes"].items():
                if phenotype is None:
                    pheno_label = "Unknown"
                else:
                    pheno_label = get_phenotype_label(phenotype)

                if "matrix_avg" in pheno_data:
                    matrix = pheno_data["matrix_avg"]
                    count = pheno_data["count"]
                    network_assignments = pheno_data.get("network_assignments",
                                                         atlas_data["all"]["network_assignments"])

                    if count == 0:
                        continue

                    # Create connectivity matrix visualization
                    title = f"Group Average Connectivity - {atlas} - {pheno_label} (n={count})"
                    output_path = os.path.join(atlas_dir, f"phenotype_{phenotype}_{atlas}_connectivity.png")

                    plot_connectivity_matrix_cole(
                        matrix,
                        network_assignments,
                        title=title,
                        save_path=output_path,
                        logger=logger
                    )

                    logger.info(f"Saved {pheno_label} {atlas} average visualization to {output_path}")

                    # Save as CSV for further analysis
                    csv_path = os.path.join(atlas_dir, f"phenotype_{phenotype}_{atlas}_connectivity.csv")
                    pd.DataFrame(matrix).to_csv(csv_path)
                    logger.info(f"Saved {pheno_label} {atlas} average matrix to {csv_path}")

            # Create difference maps between phenotypes (e.g., ADHD vs Control)
            create_phenotype_difference_maps_separate(atlas_data, atlas_dir, atlas, None, logger)


def create_phenotype_difference_maps_combined(combined_aggregated, output_dir, logger):
    """
    Create difference maps between phenotypes for combined matrices.

    Parameters:
    -----------
    combined_aggregated : dict
        Dictionary of aggregated combined matrices by phenotype
    output_dir : str
        Directory to save visualizations
    logger : Logger
        Logger object
    """
    # Focus on comparing Control (1) vs ADHD (3)
    control_code = 1
    adhd_code = 3

    # Check if we have both phenotypes
    if (control_code in combined_aggregated["phenotypes"] and
            adhd_code in combined_aggregated["phenotypes"] and
            "matrix_avg" in combined_aggregated["phenotypes"][control_code] and
            "matrix_avg" in combined_aggregated["phenotypes"][adhd_code]):

        control_matrix = combined_aggregated["phenotypes"][control_code]["matrix_avg"]
        adhd_matrix = combined_aggregated["phenotypes"][adhd_code]["matrix_avg"]
        control_count = combined_aggregated["phenotypes"][control_code]["count"]
        adhd_count = combined_aggregated["phenotypes"][adhd_code]["count"]
        network_assignments = combined_aggregated["phenotypes"][control_code].get(
            "network_assignments", combined_aggregated["all"]["network_assignments"]
        )
        atlas_designations = combined_aggregated["phenotypes"][control_code].get(
            "atlas_designations", combined_aggregated["all"]["atlas_designations"]
        )

        # Create difference matrix (ADHD - Control)
        diff_matrix = adhd_matrix - control_matrix

        # Create connectivity matrix visualization
        title = f"Difference Map - Combined (ADHD n={adhd_count} vs Control n={control_count})"
        output_path = os.path.join(output_dir, "adhd_vs_control_difference_combined_map.png")

        plot_connectivity_matrix_cole(
            diff_matrix,
            network_assignments,
            title=title,
            save_path=output_path,
            logger=logger
        )

        logger.info(f"Saved difference map combined visualization to {output_path}")

        # Extract and visualize just the cortical-subcortical block
        if atlas_designations:
            # Identify cortical and subcortical regions
            is_cortical = np.array([designation == 'Glasser' for designation in atlas_designations])
            is_subcortical = np.array([designation == 'HCP' for designation in atlas_designations])

            # Extract the cortical-subcortical block
            cortico_subcortical = diff_matrix[is_cortical, :][:, is_subcortical]

            # Create visualization for cortico-subcortical block
            plt.figure(figsize=(10, 8))
            plt.imshow(cortico_subcortical, cmap='bwr', vmin=-0.5, vmax=0.5)  # Smaller range for difference
            plt.colorbar(label='Difference in Correlation (Fisher Z)')
            plt.title(f"Difference in Cortical-Subcortical Correlations (ADHD - Control)")
            plt.xlabel('Subcortical Regions')
            plt.ylabel('Cortical Regions')
            plt.tight_layout()

            cs_output_path = os.path.join(output_dir, "adhd_vs_control_difference_cortico_subcortical.png")
            plt.savefig(cs_output_path, dpi=300)
            plt.close()

            logger.info(f"Saved difference in cortical-subcortical block to {cs_output_path}")

        # Save as CSV for further analysis
        csv_path = os.path.join(output_dir, "adhd_vs_control_difference_combined_matrix.csv")
        pd.DataFrame(diff_matrix).to_csv(csv_path)
        logger.info(f"Saved difference combined matrix to {csv_path}")
    else:
        logger.info(f"Cannot create difference maps for combined matrices - missing phenotype data")


def create_phenotype_difference_maps_separate(atlas_data, output_dir, atlas, matrix_size, logger):
    """
    Create difference maps between phenotypes for a specific atlas.
    Handles matrices of different sizes by finding the common dimensions.
    """
    # Focus on comparing Control (1) vs ADHD (3)
    control_code = 1
    adhd_code = 3

    # Check if we have both phenotypes
    if (control_code in atlas_data["phenotypes"] and
            adhd_code in atlas_data["phenotypes"] and
            "matrix_avg" in atlas_data["phenotypes"][control_code] and
            "matrix_avg" in atlas_data["phenotypes"][adhd_code]):

        control_matrix = atlas_data["phenotypes"][control_code]["matrix_avg"]
        adhd_matrix = atlas_data["phenotypes"][adhd_code]["matrix_avg"]
        control_count = atlas_data["phenotypes"][control_code]["count"]
        adhd_count = atlas_data["phenotypes"][adhd_code]["count"]

        # Get network assignments, being careful about potential mismatches
        network_assignments = atlas_data["phenotypes"][control_code].get(
            "network_assignments", atlas_data["all"]["network_assignments"]
        )

        # Handle matrix size mismatches
        if control_matrix.shape != adhd_matrix.shape:
            logger.warning(
                f"Matrix shape mismatch for {atlas}: Control {control_matrix.shape} vs ADHD {adhd_matrix.shape}")

            # Find common dimensions
            min_rows = min(control_matrix.shape[0], adhd_matrix.shape[0])
            min_cols = min(control_matrix.shape[1], adhd_matrix.shape[1])

            logger.info(f"Resizing matrices to common size {min_rows}x{min_cols} for difference calculation")

            # Resize both matrices to common dimensions
            control_matrix_resized = control_matrix[:min_rows, :min_cols]
            adhd_matrix_resized = adhd_matrix[:min_rows, :min_cols]

            # Resize network assignments if needed
            if network_assignments and len(network_assignments) > min_rows:
                network_assignments = network_assignments[:min_rows]
                logger.info(f"Truncated network assignments to {min_rows} entries")

            # Create difference matrix using resized matrices
            diff_matrix = adhd_matrix_resized - control_matrix_resized
            actual_matrix_size = min_rows  # Update matrix size for the title
        else:
            # Matrices have same shape, proceed as normal
            diff_matrix = adhd_matrix - control_matrix
            actual_matrix_size = control_matrix.shape[0]

        # Create connectivity matrix visualization
        title = f"Difference Map - {atlas} (ADHD n={adhd_count} vs Control n={control_count})"
        output_path = os.path.join(output_dir, f"adhd_vs_control_difference_{atlas}_map.png")

        # Make sure network_assignments matches the matrix dimensions
        if network_assignments and len(network_assignments) != diff_matrix.shape[0]:
            logger.warning(
                f"Network assignments length mismatch: have {len(network_assignments)}, need {diff_matrix.shape[0]}")
            if len(network_assignments) > diff_matrix.shape[0]:
                network_assignments = network_assignments[:diff_matrix.shape[0]]
            else:
                # Extend with Unknown
                network_assignments = network_assignments + ["Unknown"] * (
                            diff_matrix.shape[0] - len(network_assignments))

        plot_connectivity_matrix_cole(
            diff_matrix,
            network_assignments,
            title=title,
            save_path=output_path,
            logger=logger
        )

        logger.info(f"Saved difference map {atlas} visualization to {output_path}")

        # Save as CSV for further analysis
        csv_path = os.path.join(output_dir, f"adhd_vs_control_difference_{atlas}_matrix.csv")
        pd.DataFrame(diff_matrix).to_csv(csv_path)
        logger.info(f"Saved difference {atlas} matrix to {csv_path}")
    else:
        logger.info(f"Cannot create difference maps for {atlas} - missing phenotype data")



def process_all_subjects(subject_list, task_label, data_dir, output_dir, cole_mapping,
                         phenotype_map, raw_phenotype_map, generate_subject_vis, logger):
    """
    Process all subjects in parallel and create group-level visualizations.

    Parameters:
    -----------
    subject_list : list
        List of subject IDs to process
    task_label : str
        Task label
    data_dir : str
        Path to XCP-D output directory
    output_dir : str
        Directory to save visualizations
    cole_mapping : pandas.DataFrame
        Cole network mapping DataFrame
    phenotype_map : dict
        Dictionary mapping subject IDs to phenotypes
    raw_phenotype_map : dict
        Dictionary with raw phenotype mappings for debugging
    generate_subject_vis : bool
        Whether to generate subject-level visualizations
    logger : Logger
        Logger object
    """
    # Initialize data structure for all subjects
    all_subject_data = {}

    # Track progress
    successful = 0
    failed = 0

    # Create a progress bar if tqdm is available
    if HAS_TQDM:
        progress_bar = tqdm(total=len(subject_list), desc="Processing subjects")

    # Function to process a single subject for parallel execution
    def process_subject_parallel(sub_label):
        try:
            # Check available memory before processing
            avail_mem_gb = psutil.virtual_memory().available / (1024 ** 3)
            if avail_mem_gb < MEMORY_RESERVE_GB:
                logger.warning(f"Low memory ({avail_mem_gb:.2f} GB available) - skipping subject {sub_label}")
                return None

            result = process_subject(
                sub_label=sub_label,
                task_label=task_label,
                data_dir=data_dir,
                output_dir=output_dir,
                cole_mapping=cole_mapping,
                phenotype_map=phenotype_map,
                raw_phenotype_map=raw_phenotype_map,
                generate_subject_vis=generate_subject_vis,
                logger=logger
            )

            # Clean up memory
            gc.collect()

            return result
        except Exception as e:
            logger.error(f"Error in parallel processing for subject {sub_label}: {e}")
            logger.error(traceback.format_exc())
            return None

    # Process subjects in parallel using a thread pool
    logger.info(f"Processing {len(subject_list)} subjects using up to {MAX_WORKERS} parallel workers")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_subject = {executor.submit(process_subject_parallel, sub_label): sub_label
                                 for sub_label in subject_list}

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_subject):
                sub_label = future_to_subject[future]
                try:
                    result = future.result()
                    if result:
                        # Successfully processed
                        all_subject_data.update(result)
                        successful += 1
                    else:
                        # Failed to process
                        failed += 1

                    # Update progress
                    if HAS_TQDM:
                        progress_bar.update(1)
                        progress_bar.set_postfix(successful=successful, failed=failed)
                    else:
                        # Log progress every 10 subjects
                        if (successful + failed) % 10 == 0:
                            logger.info(f"Progress: {successful + failed}/{len(subject_list)} subjects processed")

                except Exception as e:
                    logger.error(f"Error getting result for subject {sub_label}: {e}")
                    failed += 1
                    if HAS_TQDM:
                        progress_bar.update(1)

    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        logger.error(traceback.format_exc())

    finally:
        if HAS_TQDM:
            progress_bar.close()

    # Create group-level visualizations
    if successful > 0:
        logger.info("Creating group-level visualizations")

        # Aggregate matrices by phenotype
        aggregated = aggregate_matrices_by_phenotype(all_subject_data, phenotype_map, logger)

        # Create visualizations
        visualize_group_matrices(aggregated, output_dir, logger)

    # Output summary
    logger.info("=" * 50)
    logger.info(f"Processing complete!")
    logger.info(f"Total subjects: {len(subject_list)}")
    logger.info(f"Successfully processed: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info("=" * 50)


def main():
    """Main function to run the script"""
    # Set up logging
    logger = setup_logging(LOG_DIR)
    logger.info("Starting Oregon REST Connectivity Visualization Script using Time Series Data")

    try:
        # Log system information
        logger.info(
            f"System information: {os.cpu_count()} CPUs, {psutil.virtual_memory().total / (1024 ** 3):.1f} GB RAM")
        logger.info(f"Using up to {MAX_WORKERS} parallel workers")

        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Check for CIFTI support
        if not HAS_NIBABEL:
            logger.error("nibabel is required for CIFTI file processing but is not installed")
            return 1

        # Load Cole network mapping
        cole_mapping = load_cole_network_mapping(COLE_NETWORK_FILE, logger)
        if cole_mapping is None:
            logger.error(f"Failed to load Cole network mapping from {COLE_NETWORK_FILE}")
            return 1

        # Get subject list
        if SUBJECT_LIST is not None:
            # Use provided list
            logger.info(f"Using provided subject list with {len(SUBJECT_LIST)} subjects")
            subject_list = SUBJECT_LIST
        else:
            # Find all subjects in the data directory
            subject_list = find_all_subjects(DATA_DIR, logger)

        if not subject_list:
            logger.error("No subjects found to process")
            return 1

        # Clean subject IDs in the subject list
        cleaned_subject_list = [clean_subject_id(sub) for sub in subject_list]
        # Remove duplicates that might have been created by cleaning
        cleaned_subject_list = list(set(cleaned_subject_list))
        # Sort for consistency
        cleaned_subject_list.sort()

        # Process exclusion list if provided
        if SUBJECT_EXCLUDE is not None:
            exclude_list = [clean_subject_id(sub) for sub in SUBJECT_EXCLUDE]
            exclude_list = list(set(exclude_list))
            original_count = len(cleaned_subject_list)
            cleaned_subject_list = [sub for sub in cleaned_subject_list if sub not in exclude_list]
            logger.info(
                f"Excluded {len(exclude_list)} subjects, processing remaining {len(cleaned_subject_list)} subjects ({original_count - len(cleaned_subject_list)} removed)")

        # Process only a fraction of subjects if configured
        if PROCESS_FRACTION is not None and 0 < PROCESS_FRACTION < 1:
            original_count = len(cleaned_subject_list)
            sample_size = max(1, int(original_count * PROCESS_FRACTION))
            cleaned_subject_list = cleaned_subject_list[:sample_size]
            logger.info(f"Processing {sample_size} subjects ({PROCESS_FRACTION:.1%} of {original_count} total)")
        else:
            logger.info(f"Processing all {len(cleaned_subject_list)} subjects")

        # Load phenotype data if available
        phenotype_map = {}
        raw_phenotype_map = {}
        if os.path.exists(NDAR_FILE):
            phenotype_map, raw_phenotype_map = load_ndar_phenotypes(NDAR_FILE, logger)
        else:
            logger.warning(f"NDAR phenotype file not found: {NDAR_FILE}")

        # Process all subjects
        start_time = datetime.now()
        process_all_subjects(
            subject_list=cleaned_subject_list,
            task_label=TASK,
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            cole_mapping=cole_mapping,
            phenotype_map=phenotype_map,
            raw_phenotype_map=raw_phenotype_map,
            generate_subject_vis=GENERATE_SUBJECT_VISUALIZATIONS,
            logger=logger
        )

        # Log total execution time
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Total execution time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)")

        return 0

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())