#!/usr/bin/env python3

"""
Oregon ALFF and ReHo Analysis Script

This script processes ALFF and ReHo data from XCP-D outputs,
combines cortical (Glasser) and subcortical (HCP) regions,
maps them to Cole networks, and creates visualizations by phenotype.
Most importantly, creates a CSV of all subject-timepoints with parcels and ALFF and ReHo values (two CSVs)
"""

#########################################################################################
#                          CONFIGURATION VARIABLES - EDIT FOR YOUR NEEDS                #
#########################################################################################

# XCP-D data directory for ALFF (UNCENSORED data)
DATA_DIR1 = "/media/hcs-sci-psy-narun/OREGON_ADHD_1000/OREGON_ONE_SINGLE/derivatives/xcp_d-0.10.6_acompcor_no_censor_redoMay_10"

# XCP-D data directory for ReHo (CENSORED data)
DATA_DIR2 = "/media/hcs-sci-psy-narun/OREGON_ADHD_1000/OREGON_ONE_SINGLE/derivatives/xcp_d-0.10.6_acompcor_censorMT_redoMay_12"

# Directory to save visualizations and outputs
OUTPUT_DIR = "/media/hcs-sci-psy-narun/Jack/ALFF_ReHo_Analysis_v3_final_w_CSV"

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
import logging
import gc
import traceback
from datetime import datetime
import seaborn as sns
from scipy import stats
import concurrent.futures
import psutil

# Try to import tqdm for progress bars if available
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

########################
# Additional variables #
########################
# Configuration for parallelization
# Default to 75% of available cores for parallel processing
MAX_WORKERS = max(int(os.cpu_count() * 0.75), 1)
# How much memory to reserve (GB) - prevent out-of-memory errors
MEMORY_RESERVE_GB = 8  # Reserve 8GB to avoid system instability

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
        log_file = os.path.join(log_dir, f'alff_reho_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    else:
        log_file = f'alff_reho_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

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
        required_cols = ['name_Glasser', 'Cole_net_name']
        missing_cols = [col for col in required_cols if col not in mapping_df.columns]

        if missing_cols:
            logger.error(f"Missing required columns in Cole network mapping file: {missing_cols}")
            return None

        # Count networks and regions per network
        if 'Cole_net_name' in mapping_df.columns:
            networks = mapping_df['Cole_net_name'].value_counts().to_dict()
            networks_str = ", ".join([f"{net}: {count}" for net, count in networks.items()])
            logger.info(f"Networks distribution: {networks_str}")

        return mapping_df

    except Exception as e:
        logger.error(f"Error loading Cole network mapping: {e}")
        logger.error(traceback.format_exc())
        return None


def find_alff_reho_files(sub_label, task_label, data_dir1, data_dir2, metric_type, logger):
    """
    Find ALFF or ReHo TSV files for a subject using the appropriate data directory.

    Parameters:
    -----------
    sub_label : str
        Subject label
    task_label : str
        Task label
    data_dir1 : str
        Path to uncensored XCP-D output directory (for ALFF)
    data_dir2 : str
        Path to censored XCP-D output directory (for ReHo)
    metric_type : str
        Type of metric to find ('alff' or 'reho')
    logger : Logger
        Logger object

    Returns:
    --------
    dict : Dictionary with 'Glasser' and 'HCP' file paths
    """
    # Start timer for performance tracking
    start_time = datetime.now()

    # Select appropriate data directory based on metric type
    if metric_type.lower() == 'alff':
        data_dir = data_dir1
        logger.info(f"Searching for ALFF files in uncensored data directory for subject {sub_label}")
    else:  # Use censored data for ReHo
        data_dir = data_dir2
        logger.info(f"Searching for ReHo files in censored data directory for subject {sub_label}")

    # Define the most likely locations for TSV files
    specific_paths = [
        # Primary XCP-D output location
        os.path.join(data_dir, f'sub-{sub_label}', 'func'),
        # Alternative location with xcp_d subdirectory
        os.path.join(data_dir, 'xcp_d', f'sub-{sub_label}', 'func')
    ]

    # Patterns specifically for ALFF or ReHo TSV files
    glasser_pattern = f"sub-{sub_label}_task-{task_label}_*_space-fsLR_seg-Glasser_stat-{metric_type}_bold.tsv"
    hcp_pattern = f"sub-{sub_label}_task-{task_label}_*_space-fsLR_seg-HCP_stat-{metric_type}_bold.tsv"

    # Initialize result
    result = {
        'Glasser': None,
        'HCP': None
    }

    # Try each specific path and pattern combination
    logger.info(f"Searching for {metric_type.upper()} files for subject {sub_label}")

    for path in specific_paths:
        if not os.path.exists(path):
            continue

        logger.debug(f"Checking directory: {path}")

        # Check for Glasser file
        glasser_files = glob.glob(os.path.join(path, glasser_pattern))
        if glasser_files:
            result['Glasser'] = glasser_files[0]
            logger.info(f"Found Glasser {metric_type.upper()} file: {result['Glasser']}")

        # Check for HCP file
        hcp_files = glob.glob(os.path.join(path, hcp_pattern))
        if hcp_files:
            result['HCP'] = hcp_files[0]
            logger.info(f"Found HCP {metric_type.upper()} file: {result['HCP']}")

        # If we found both files, we can stop searching
        if result['Glasser'] and result['HCP']:
            break

    # Performance measurement
    search_time = (datetime.now() - start_time).total_seconds()

    if not (result['Glasser'] or result['HCP']):
        logger.warning(f"No {metric_type.upper()} files found for subject {sub_label}")
    else:
        logger.info(
            f"Found {sum(1 for v in result.values() if v)} {metric_type.upper()} files for subject {sub_label} in {search_time:.2f} seconds")

    return result


def parse_timepoint_info(timepoint_id):
    """
    Parse a timepoint ID to extract age and study year information.

    Parameters:
    -----------
    timepoint_id : str
        Timepoint ID (e.g., "sub-NDARXXXAge112MonthsStudyYear02" where 112 means 11 years, 2 months)

    Returns:
    --------
    tuple : (age_months, study_year) or (None, None) if not parsable
    """
    if timepoint_id is None:
        return None, None

    # Remove 'sub-' prefix if present
    if timepoint_id.startswith('sub-'):
        timepoint_id = timepoint_id[4:]

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


def should_exclude_timepoint(subject_id, timepoint_id, logger):
    """
    Determine if a specific subject-timepoint combination should be excluded.

    Parameters:
    -----------
    subject_id : str
        Subject identifier (with or without 'sub-' prefix)
    timepoint_id : str
        Complete timepoint identifier (e.g., "sub-NDARXXXAge112MonthsStudyYear02")
    logger : Logger
        Logger object

    Returns:
    --------
    bool : True if should exclude, False otherwise
    """
    # Normalize subject IDs for comparison
    if subject_id.startswith('sub-'):
        subject_id = subject_id[4:]

    # Check if complete subject should be excluded
    for excluded_subject in SUBJECT_EXCLUDE:
        excluded_id = excluded_subject[4:] if excluded_subject.startswith('sub-') else excluded_subject
        if subject_id == excluded_id:
            logger.info(f"Excluding subject {subject_id} - in complete exclusion list")
            return True

    # If timepoint_id is not provided, we can't check timepoint-specific exclusions
    if not timepoint_id:
        return False

    # Normalize timepoint ID for comparison
    normalized_timepoint = timepoint_id
    if normalized_timepoint.startswith('sub-'):
        normalized_timepoint = normalized_timepoint[4:]

    # Check if this specific timepoint should be excluded
    for excluded_entry in SUBJECT_TIMEPOINT_EXCLUDE:
        excluded_entry_normalized = excluded_entry
        if excluded_entry_normalized.startswith('sub-'):
            excluded_entry_normalized = excluded_entry_normalized[4:]

        # Extract subject part from excluded entry
        if "Age" in excluded_entry_normalized:
            excluded_subject = excluded_entry_normalized.split("Age")[0]
        else:
            excluded_subject = excluded_entry_normalized

        # If subject doesn't match, skip this entry
        if subject_id != excluded_subject:
            continue

        # Parse age and study year from both timepoints
        excluded_age, excluded_year = parse_timepoint_info(excluded_entry)
        current_age, current_year = parse_timepoint_info(timepoint_id)

        # If we can't parse either timepoint, skip
        if excluded_age is None or current_age is None:
            continue

        # Check if ages match (within 1 month tolerance)
        age_match = abs(excluded_age - current_age) <= 1

        # Check if study years match (if available in both)
        year_match = True
        if excluded_year is not None and current_year is not None:
            year_match = excluded_year == current_year

        # If both age and year match, exclude this timepoint
        if age_match and year_match:
            logger.info(f"Excluding timepoint {timepoint_id} - matched exclusion entry {excluded_entry}")
            return True

    # If we get here, this subject-timepoint should not be excluded
    return False

def load_tsv_data(file_path, logger):
    """
    Load data from a TSV file with region names as column headers.

    Parameters:
    -----------
    file_path : str
        Path to TSV file
    logger : Logger
        Logger object

    Returns:
    --------
    tuple : (DataFrame of data, list of region labels, list of metric values)
    """
    try:
        logger.info(f"Loading data from {file_path}")

        # Read the TSV file - regions are column headers, values are in a single row
        df = pd.read_csv(file_path, sep='\t')

        # Log the shape and structure
        logger.info(f"TSV file has shape: {df.shape} - {len(df.columns)} columns (regions) and {len(df)} rows")

        # Get the region names from column headers (skip any index column)
        if 'Unnamed: 0' in df.columns:
            region_labels = df.columns[1:].tolist()
        else:
            region_labels = df.columns.tolist()

        # Get the values from the first (and only) data row
        if len(df) > 0:
            if 'Unnamed: 0' in df.columns:
                metric_values = df.iloc[0, 1:].tolist()
            else:
                metric_values = df.iloc[0, :].tolist()
        else:
            logger.error(f"TSV file {file_path} has no data rows")
            return None, [], []

        # Create a DataFrame with regions as rows for consistent processing
        result_df = pd.DataFrame({
            'Region': region_labels,
            'Value': metric_values
        })

        # Show samples for debugging
        logger.info(f"Extracted {len(region_labels)} regions from column headers")
        logger.info(f"Sample regions: {region_labels[:5]}")
        logger.info(f"Sample values: {metric_values[:5]}")

        return result_df, region_labels, metric_values

    except Exception as e:
        logger.error(f"Error loading TSV file {file_path}: {e}")
        logger.error(traceback.format_exc())
        return None, [], []


def combine_glasser_hcp_data(glasser_file, hcp_file, logger):
    """
    Combine data from Glasser and HCP files, filtering out hippocampus regions from Glasser.
    """
    try:
        # Load data from both files
        glasser_df, glasser_labels, glasser_values = load_tsv_data(glasser_file, logger)
        hcp_df, hcp_labels, hcp_values = load_tsv_data(hcp_file, logger)

        if glasser_df is None or hcp_df is None:
            logger.error("Could not load both Glasser and HCP data")
            return None, [], []

        # Filter out hippocampus regions from Glasser data
        hippo_indices = []
        filtered_glasser_labels = []
        filtered_glasser_values = []

        for i, label in enumerate(glasser_labels):
            # Check if this is a hippocampus region
            if label in ['L_H', 'R_H', 'Left_H', 'Right_H'] or label.endswith('_Hippocampus') or 'hippocampus' in label.lower():
                hippo_indices.append(i)
                logger.info(f"Filtering hippocampus region from Glasser: {label}")
            else:
                filtered_glasser_labels.append(label)
                filtered_glasser_values.append(glasser_values[i])

        if hippo_indices:
            logger.info(f"Filtered out {len(hippo_indices)} hippocampus regions from Glasser data")

        # Use the filtered Glasser data
        glasser_labels = filtered_glasser_labels
        glasser_values = filtered_glasser_values

        # Combine the filtered Glasser data with HCP data
        combined_labels = glasser_labels + hcp_labels
        combined_values = glasser_values + hcp_values

        # Create a new DataFrame with combined data
        combined_data = pd.DataFrame({
            'Region': combined_labels,
            'Value': combined_values,
            'Atlas': ['Glasser'] * len(glasser_labels) + ['HCP'] * len(hcp_labels)
        })

        logger.info(
            f"Combined data has {len(combined_labels)} regions: {len(glasser_labels)} Glasser + {len(hcp_labels)} HCP")
        return combined_data, combined_labels, combined_values

    except Exception as e:
        logger.error(f"Error combining Glasser and HCP data: {e}")
        logger.error(traceback.format_exc())
        return None, [], []


def map_regions_to_cole_networks(combined_data, cole_mapping, logger):
    """
    Map regions to Cole networks by matching column headers to names in Cole mapping.

    Parameters:
    -----------
    combined_data : DataFrame
        DataFrame with 'Region' column containing region labels (column headers from TSV)
    cole_mapping : DataFrame
        Cole network mapping DataFrame
    logger : Logger
        Logger object

    Returns:
    --------
    DataFrame : DataFrame with added 'Network' column
    """
    try:
        if combined_data.empty:
            logger.warning("Combined data is empty, cannot map to networks")
            return combined_data

        # Create a dictionary to map region names to networks
        region_to_network = {}
        region_to_order = {}

        # Log some sample region names for debugging
        sample_regions = combined_data['Region'].head(10).tolist()
        logger.info(f"Sample region names to match: {sample_regions}")

        # Create mappings for different region name formats
        for _, row in cole_mapping.iterrows():
            glasser_name = row['name_Glasser']
            network_name = row['Cole_net_name']

            if pd.notna(glasser_name) and pd.notna(network_name):
                # Original name format (e.g., "R_V1")
                region_to_network[glasser_name] = network_name

                # Convert R_ to Right_ for standard format (e.g., "Right_V1")
                if isinstance(glasser_name, str):
                    if glasser_name.startswith('R_'):
                        alt_name = f"Right_{glasser_name[2:]}"
                        region_to_network[alt_name] = network_name
                    elif glasser_name.startswith('L_'):
                        alt_name = f"Left_{glasser_name[2:]}"
                        region_to_network[alt_name] = network_name

                # Network order
                if 'Cole_net_order' in row and pd.notna(row['Cole_net_order']):
                    region_to_order[glasser_name] = row['Cole_net_order']

                    # Also add for alternative names
                    if isinstance(glasser_name, str):
                        if glasser_name.startswith('R_'):
                            region_to_order[f"Right_{glasser_name[2:]}"] = row['Cole_net_order']
                        elif glasser_name.startswith('L_'):
                            region_to_order[f"Left_{glasser_name[2:]}"] = row['Cole_net_order']

        # Add special mapping for subcortical structures which follow HCP naming
        for region in combined_data['Region']:
            if isinstance(region, str) and ('_LEFT' in region or '_RIGHT' in region):
                region_to_network[region] = 'Subcortex'
                region_to_order[region] = 9000  # High number for sorting

        # Apply the mapping
        combined_data['Network'] = combined_data['Region'].map(region_to_network)

        # Count matches before further processing
        matched_count = combined_data['Network'].notna().sum()
        total_count = len(combined_data)
        logger.info(
            f"Initial matching: {matched_count}/{total_count} regions matched ({(matched_count / total_count) * 100:.1f}%)")

        # For unmatched regions, try more aggressive matching
        if matched_count < total_count:
            unmatched = combined_data[combined_data['Network'].isna()]

            for idx, row in unmatched.iterrows():
                region = row['Region']
                if not isinstance(region, str):
                    continue

                # Try various transformations to match
                matched = False

                # 1. Try exact match with region_to_network keys
                if not matched and region in region_to_network:
                    combined_data.at[idx, 'Network'] = region_to_network[region]
                    matched = True

                # 2. Try with different cases (upper/lower)
                if not matched:
                    for key in region_to_network.keys():
                        if not isinstance(key, str):
                            continue
                        if region.lower() == key.lower():
                            combined_data.at[idx, 'Network'] = region_to_network[key]
                            matched = True
                            logger.info(f"Matched {region} to {key} using case-insensitive matching")
                            break

                # 3. Try partial matching
                if not matched:
                    for key in region_to_network.keys():
                        if not isinstance(key, str):
                            continue
                        # Check if region is contained in key or key is contained in region
                        if (region.lower() in key.lower()) or (key.lower() in region.lower()):
                            combined_data.at[idx, 'Network'] = region_to_network[key]
                            matched = True
                            logger.info(f"Matched {region} to {key} using partial matching")
                            break

                # 4. Directly map known patterns for HCP regions
                if not matched and any(x in region.upper() for x in
                                       ['CEREBELLUM', 'THALAMUS', 'CAUDATE', 'PUTAMEN', 'PALLIDUM', 'HIPPOCAMPUS',
                                        'AMYGDALA', 'ACCUMBENS', 'DIENCEPHALON', 'BRAIN']):
                    combined_data.at[idx, 'Network'] = 'Subcortex'
                    matched = True
                    logger.info(f"Matched {region} to Subcortex based on region name")

        # Fill any remaining NaN values with 'Unknown'
        combined_data['Network'] = combined_data['Network'].fillna('Unknown')

        # Add network order if available
        if region_to_order:
            combined_data['Network_Order'] = combined_data['Region'].map(region_to_order)
            max_order = max(region_to_order.values()) if region_to_order else 1000
            combined_data['Network_Order'] = combined_data['Network_Order'].fillna(max_order + 1)

        # Final logging
        final_matched = combined_data['Network'].ne('Unknown').sum()
        logger.info(
            f"Final matching: {final_matched}/{total_count} regions matched ({(final_matched / total_count) * 100:.1f}%)")

        # Log network distribution
        network_counts = combined_data['Network'].value_counts().to_dict()
        networks_str = ", ".join([f"{net}: {count}" for net, count in network_counts.items()])
        logger.info(f"Network distribution in mapped data: {networks_str}")

        # If network distribution shows many unknowns, log some of them for debugging
        unknown_count = network_counts.get('Unknown', 0)
        if unknown_count > 0 and unknown_count / total_count > 0.2:  # More than 20% unknown
            unknown_regions = combined_data[combined_data['Network'] == 'Unknown']['Region'].tolist()
            logger.warning(f"High percentage of unknown networks. Sample unmatched regions: {unknown_regions[:10]}")

        return combined_data

    except Exception as e:
        logger.error(f"Error mapping regions to Cole networks: {e}")
        logger.error(traceback.format_exc())
        return combined_data


def process_subject(sub_label, task_label, data_dir1, data_dir2, cole_mapping, phenotype_map, raw_phenotype_map, logger):
    """
    Process a single subject's ALFF and ReHo data.

    Parameters:
    -----------
    sub_label : str
        Subject label
    task_label : str
        Task label
    data_dir1 : str
        Path to uncensored XCP-D output directory (for ALFF)
    data_dir2 : str
        Path to censored XCP-D output directory (for ReHo)
    cole_mapping : DataFrame
        Cole network mapping DataFrame
    phenotype_map : dict
        Dictionary mapping subject IDs to phenotypes
    raw_phenotype_map : dict
        Dictionary with raw phenotype mappings for debugging
    logger : Logger
        Logger object

    Returns:
    --------
    dict : Dictionary with ALFF and ReHo data for the subject
    """
    subject_data = {
        'subject_id': sub_label,
        'alff': None,
        'reho': None,
        'phenotype': None
    }

    try:
        # Check if this subject should be excluded
        if should_exclude_timepoint(sub_label, f"sub-{sub_label}", logger):
            logger.info(f"Subject {sub_label} excluded based on exclusion criteria")
            return subject_data

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

        subject_data['phenotype'] = subject_phenotype

        # Process ALFF data - use uncensored data directory
        alff_files = find_alff_reho_files(sub_label, task_label, data_dir1, data_dir2, 'alff', logger)
        if alff_files['Glasser'] and alff_files['HCP']:
            alff_combined_data, _, _ = combine_glasser_hcp_data(alff_files['Glasser'], alff_files['HCP'], logger)
            if alff_combined_data is not None:
                # Map regions to Cole networks
                alff_combined_data = map_regions_to_cole_networks(alff_combined_data, cole_mapping, logger)
                subject_data['alff'] = alff_combined_data
                logger.info(f"Successfully processed ALFF data for subject {sub_label}")

        # Process ReHo data - use censored data directory
        reho_files = find_alff_reho_files(sub_label, task_label, data_dir1, data_dir2, 'reho', logger)
        if reho_files['Glasser'] and reho_files['HCP']:
            reho_combined_data, _, _ = combine_glasser_hcp_data(reho_files['Glasser'], reho_files['HCP'], logger)
            if reho_combined_data is not None:
                # Map regions to Cole networks
                reho_combined_data = map_regions_to_cole_networks(reho_combined_data, cole_mapping, logger)
                subject_data['reho'] = reho_combined_data
                logger.info(f"Successfully processed ReHo data for subject {sub_label}")

        return subject_data

    except Exception as e:
        logger.error(f"Error processing subject {sub_label}: {e}")
        logger.error(traceback.format_exc())
        return subject_data


def aggregate_data_by_phenotype(subject_data_list, logger):
    """
    Aggregate ALFF and ReHo data by phenotype.

    Parameters:
    -----------
    subject_data_list : list
        List of dictionaries with subject data
    logger : Logger
        Logger object

    Returns:
    --------
    dict : Dictionary with aggregated data by phenotype
    """
    # Initialize result structure
    result = {
        'alff': {
            'by_phenotype': {},
            'by_network': {},
            'all_data': pd.DataFrame()
        },
        'reho': {
            'by_phenotype': {},
            'by_network': {},
            'all_data': pd.DataFrame()
        }
    }

    # Process each subject
    logger.info(f"Aggregating data from {len(subject_data_list)} subjects")

    # Concatenate all subject data
    alff_data_frames = []
    reho_data_frames = []

    for subject_data in subject_data_list:
        subject_id = subject_data['subject_id']
        phenotype = subject_data['phenotype']
        phenotype_label = get_phenotype_label(phenotype) if phenotype is not None else "Unknown"

        # Process ALFF data
        if subject_data['alff'] is not None:
            # Add subject and phenotype info
            subject_data['alff']['Subject'] = subject_id
            subject_data['alff']['Phenotype'] = phenotype
            subject_data['alff']['Phenotype_Label'] = phenotype_label
            alff_data_frames.append(subject_data['alff'])

        # Process ReHo data
        if subject_data['reho'] is not None:
            # Add subject and phenotype info
            subject_data['reho']['Subject'] = subject_id
            subject_data['reho']['Phenotype'] = phenotype
            subject_data['reho']['Phenotype_Label'] = phenotype_label
            reho_data_frames.append(subject_data['reho'])

    # Combine all subjects' data
    if alff_data_frames:
        result['alff']['all_data'] = pd.concat(alff_data_frames, ignore_index=True)
        logger.info(f"Combined ALFF data from {len(alff_data_frames)} subjects")

    if reho_data_frames:
        result['reho']['all_data'] = pd.concat(reho_data_frames, ignore_index=True)
        logger.info(f"Combined ReHo data from {len(reho_data_frames)} subjects")

    # Compute summary statistics by phenotype
    for metric_type in ['alff', 'reho']:
        if result[metric_type]['all_data'].empty:
            continue

        # Group by phenotype and region
        grouped_by_phenotype = result[metric_type]['all_data'].groupby(['Phenotype', 'Region'])

        # Calculate mean, std, sem, etc.
        phenotype_stats = grouped_by_phenotype['Value'].agg(['mean', 'std', 'sem', 'count'])
        result[metric_type]['by_phenotype'] = phenotype_stats

        # Group by phenotype and network
        grouped_by_network = result[metric_type]['all_data'].groupby(['Phenotype', 'Network'])

        # Calculate network statistics
        network_stats = grouped_by_network['Value'].agg(['mean', 'std', 'sem', 'count'])
        result[metric_type]['by_network'] = network_stats

        # Log stats for each phenotype
        for phenotype in result[metric_type]['all_data']['Phenotype'].unique():
            if pd.isna(phenotype):
                continue

            phenotype_label = get_phenotype_label(phenotype)
            phenotype_data = result[metric_type]['all_data'][result[metric_type]['all_data']['Phenotype'] == phenotype]

            logger.info(
                f"{metric_type.upper()} - {phenotype_label}: {len(phenotype_data)} data points from {phenotype_data['Subject'].nunique()} subjects")

    return result


def create_aggregated_csv(all_subject_data, metric_type, output_dir, logger):
    """
    Create aggregated CSV files with one row per subject-timepoint:
    1. Full version with phenotype information for analysis
    2. ML-ready version without phenotype columns for unbiased machine learning
    """
    try:
        logger.info(f"Creating aggregated {metric_type.upper()} CSV files")

        # Create a list to hold all rows
        rows = []

        # Create a set to track all unique region names
        all_regions = set()

        # Process each subject
        for subject_data in all_subject_data:
            subject_id = subject_data['subject_id']
            phenotype = subject_data['phenotype']
            phenotype_label = get_phenotype_label(phenotype) if phenotype is not None else "Unknown"

            # Get the data for this metric
            metric_data = subject_data.get(metric_type)

            if metric_data is not None:
                # Extract regions and values
                regions = metric_data['Region'].tolist()
                values = metric_data['Value'].tolist()

                # Track all region names
                all_regions.update(regions)

                # Create a row with metadata and values
                row = {
                    'subject_id': subject_id,
                    'phenotype': phenotype,
                    'phenotype_label': phenotype_label
                }

                # Add each region's value to the row
                for region, value in zip(regions, values):
                    row[region] = value

                # Add this row to our list
                rows.append(row)

        if not rows:
            logger.warning(f"No data found for {metric_type}, cannot create aggregated CSV")
            return

        # Convert to DataFrame
        df = pd.DataFrame(rows)

        # Ensure all columns exist (filling missing values with NaN)
        for region in all_regions:
            if region not in df.columns:
                df[region] = np.nan

        # Reorder columns: metadata first, then regions
        metadata_cols = ['subject_id', 'phenotype', 'phenotype_label']
        region_cols = [col for col in df.columns if col not in metadata_cols]

        # Check if we need to filter out hippocampus regions again at the CSV level
        # (belt and suspenders approach)
        hippo_cols = [col for col in region_cols if col in ['L_H', 'R_H', 'Left_H', 'Right_H']]

        if hippo_cols:
            logger.warning(f"Found hippocampus columns in final CSV that weren't filtered earlier: {hippo_cols}")
            region_cols = [col for col in region_cols if col not in hippo_cols]

        # Sort region columns for consistency
        region_cols.sort()

        # Combine metadata and region columns for the complete CSV
        ordered_cols = metadata_cols + region_cols
        df_complete = df[ordered_cols]

        # Save the complete CSV with phenotype information
        csv_path = os.path.join(output_dir, f"{metric_type}_aggregated_data.csv")
        df_complete.to_csv(csv_path, index=False)
        logger.info(f"Saved complete aggregated {metric_type.upper()} data for {len(rows)} subjects to {csv_path}")

        # Create a version without phenotype information for ML
        ml_cols = ['subject_id'] + region_cols
        df_ml = df[ml_cols]

        # Save the ML-ready CSV without phenotype information
        ml_csv_path = os.path.join(output_dir, f"{metric_type}_aggregated_data_ml.csv")
        df_ml.to_csv(ml_csv_path, index=False)
        logger.info(f"Saved ML-ready {metric_type.upper()} data (without phenotype info) to {ml_csv_path}")

        logger.info(f"CSV contains {len(region_cols)} region columns")

        return csv_path

    except Exception as e:
        logger.error(f"Error creating aggregated CSV: {e}")
        logger.error(traceback.format_exc())
        return None


def create_region_scatter_plot(data, phenotype_order, metric_type, output_dir, logger):
    """
    Create a scatter plot of individual regions colored by Cole network.

    Parameters:
    -----------
    data : DataFrame
        DataFrame with region data
    phenotype_order : list
        Order of phenotypes to display
    metric_type : str
        Type of metric ('alff' or 'reho')
    output_dir : str
        Directory to save visualization
    logger : Logger
        Logger object

    Returns:
    --------
    None
    """
    try:
        logger.info(f"Creating region scatter plot for {metric_type}")

        # Define network colors
        network_colors = {
            'Visual1': '#000023',
            'Visual2': '#1E90FF',
            'Somatomotor': '#32CD32',
            'Cingulo-Opercular': '#FFD700',
            'Dorsal_Attention': '#12670f',
            'Language': '#FF69B4',
            'Frontoparietal': '#FF6347',
            'Auditory': '#00CED1',
            'Posterior_Multimodal': '#9400D3',
            'Default': '#800000',
            'Orbito-Affective': '#FF8C00',
            'Ventral_Multimodal': '#8B4513',
            'Subcortex': '#808080',
            'Unknown': '#CCCCCC'
        }

        # Create a mapping from phenotype code to label
        phenotype_labels = {p: get_phenotype_label(p) for p in phenotype_order if p is not None}
        phenotype_labels[None] = "Unknown"

        # Create figure
        plt.figure(figsize=(14, 10))

        # Add jittered points for each phenotype
        x_positions = []
        for i, phenotype in enumerate(phenotype_order):
            # Get data for this phenotype
            phenotype_data = data[data['Phenotype'] == phenotype]

            if phenotype_data.empty:
                continue

            # Use x-position with jitter
            x_pos = i
            x_positions.append(x_pos)

            # Get unique networks and sort them (to ensure consistent legend)
            networks = sorted(phenotype_data['Network'].unique())

            # Add points colored by network
            for network in networks:
                network_data = phenotype_data[phenotype_data['Network'] == network]

                # Create jittered x-positions with controlled random seed for reproducibility
                np.random.seed(42 + i)  # Different seed for each phenotype but reproducible
                x_jittered = np.random.normal(x_pos, 0.05, size=len(network_data))

                # Add to plot with appropriate color
                plt.scatter(
                    x_jittered,
                    network_data['Value'],
                    c=network_colors.get(network, '#CCCCCC'),
                    alpha=0.7,  # Slightly increased opacity
                    s=35,  # Slightly larger points
                    label=network if i == 0 else None,  # Only add to legend once
                    edgecolors='none'  # No edge to make dense plots cleaner
                )

        # Add a box plot at each phenotype position to show distribution
        plt.boxplot(
            [data[data['Phenotype'] == p]['Value'].values for p in phenotype_order if p in data['Phenotype'].values],
            positions=x_positions,
            widths=0.3,
            patch_artist=True,
            boxprops={'facecolor': 'white', 'alpha': 0.7},
            medianprops={'color': 'black', 'linewidth': 1.5},
            whiskerprops={'color': 'black', 'linewidth': 1.0},
            capprops={'color': 'black', 'linewidth': 1.0},
            flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markersize': 4, 'alpha': 0.5}
        )

        # Customize the plot
        plt.xlabel('Phenotype', fontsize=14)
        plt.ylabel(f'{metric_type.upper()} Value', fontsize=14)
        plt.title(f'Individual Region {metric_type.upper()} Values by Phenotype', fontsize=16)

        # Set x-ticks
        plt.xticks(x_positions,
                   [phenotype_labels.get(p, "Unknown") for p in phenotype_order if p in data['Phenotype'].values])

        # Add legend with network colors
        # Get unique networks in the data
        used_networks = sorted(data['Network'].unique())

        # Create legend handles and labels only for networks that exist in the data
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=network_colors.get(network, '#CCCCCC'),
                              markersize=10)
                   for network in used_networks]
        labels = used_networks

        # Place legend outside of plot area, or underneath if too many networks
        if len(used_networks) > 8:
            # Place legend below plot with multiple columns
            plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                       ncol=min(4, len(used_networks)), frameon=True)
        else:
            # Place legend to the right of the plot
            plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(output_dir, f'{metric_type}_region_scatter_plot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved region scatter plot to {output_path}")

    except Exception as e:
        logger.error(f"Error creating region scatter plot: {e}")
        logger.error(traceback.format_exc())


def create_network_bar_plot(data, phenotype_order, metric_type, output_dir, logger):
    """
    Create a bar plot of network averages by phenotype.

    Parameters:
    -----------
    data : DataFrame
        DataFrame with network-level data
    phenotype_order : list
        Order of phenotypes to display
    metric_type : str
        Type of metric ('alff' or 'reho')
    output_dir : str
        Directory to save visualization
    logger : Logger
        Logger object

    Returns:
    --------
    None
    """
    try:
        logger.info(f"Creating network bar plot for {metric_type}")

        # Define network colors
        network_colors = {
            'Visual1': '#000023',
            'Visual2': '#1E90FF',
            'Somatomotor': '#32CD32',
            'Cingulo-Opercular': '#FFD700',
            'Dorsal_Attention': '#12670f',
            'Language': '#FF69B4',
            'Frontoparietal': '#FF6347',
            'Auditory': '#00CED1',
            'Posterior_Multimodal': '#9400D3',
            'Default': '#800000',
            'Orbito-Affective': '#FF8C00',
            'Ventral_Multimodal': '#8B4513',
            'Subcortex': '#808080',
            'Unknown': '#CCCCCC'
        }

        # Create a mapping from phenotype code to label
        phenotype_labels = {p: get_phenotype_label(p) for p in phenotype_order if p is not None}
        phenotype_labels[None] = "Unknown"

        # Reset index to have phenotype and network as columns
        if isinstance(data.index, pd.MultiIndex):
            plot_data = data.reset_index()
        else:
            plot_data = data.copy()

        # Filter to only include specified phenotypes
        plot_data = plot_data[plot_data['Phenotype'].isin(phenotype_order)]

        # Set a categorical phenotype column with the desired order
        valid_phenotypes = [p for p in phenotype_order if p in plot_data['Phenotype'].unique()]
        plot_data['Phenotype_Cat'] = pd.Categorical(
            plot_data['Phenotype'].apply(lambda x: x if x in valid_phenotypes else None),
            categories=valid_phenotypes,
            ordered=True
        )

        # Create a figure with subplots for each network
        networks = sorted(plot_data['Network'].unique())
        n_networks = len(networks)

        # Calculate grid dimensions
        n_cols = min(3, n_networks)
        n_rows = (n_networks + n_cols - 1) // n_cols  # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), squeeze=False)
        axes = axes.flatten()

        # Plot each network
        for i, network in enumerate(networks):
            if i >= len(axes):
                break

            ax = axes[i]
            network_data = plot_data[plot_data['Network'] == network]

            # Skip empty networks
            if network_data.empty:
                ax.set_visible(False)
                continue

            # Create the bar plot
            sns.barplot(
                data=network_data,
                x='Phenotype_Cat',
                y='mean',
                ax=ax,
                color=network_colors.get(network, '#CCCCCC'),
                alpha=0.7
            )

            # Add error bars
            for j, row in network_data.iterrows():
                x = network_data.index.get_loc(j)
                ax.errorbar(
                    x, row['mean'],
                    yerr=row['sem'],
                    fmt='none',
                    ecolor='black',
                    capsize=5
                )

            # Customize the subplot
            ax.set_title(f'{network}', fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel(f'{metric_type.upper()} Value')

            # Set x-tick labels
            ax.set_xticklabels([phenotype_labels.get(p, "Unknown") for p in valid_phenotypes])

            # Add the sample size to the bars
            for j, p in enumerate(valid_phenotypes):
                if p in network_data['Phenotype'].values:
                    count = network_data[network_data['Phenotype'] == p]['count'].values[0]
                    ax.text(j, 0, f'n={int(count)}', ha='center', va='bottom', fontsize=10)

        # Hide unused subplots
        for i in range(n_networks, len(axes)):
            axes[i].set_visible(False)

        # Add a main title
        plt.suptitle(f'{metric_type.upper()} Network Averages by Phenotype', fontsize=16, y=1.02)

        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(output_dir, f'{metric_type}_network_bar_plot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved network bar plot to {output_path}")

    except Exception as e:
        logger.error(f"Error creating network bar plot: {e}")
        logger.error(traceback.format_exc())


def fdr_correction(p_values, alpha=0.05, method='indep'):
    """
    Benjamini-Hochberg FDR correction for multiple comparisons.

    Parameters:
    -----------
    p_values : array-like
        List of p-values to correct
    alpha : float
        Significance level
    method : str
        Method for FDR correction ('indep' or 'negcorr')

    Returns:
    --------
    array : Adjusted p-values
    """
    p_values = np.array(p_values)
    n = len(p_values)

    if n == 0:
        return np.array([])

    # Sort p-values in ascending order
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]

    # Calculate adjusted p-values
    if method == 'indep':
        # For independent tests (original BH)
        adjusted_p_values = sorted_p_values * n / np.arange(1, n + 1)
    elif method == 'negcorr':
        # For negatively correlated tests (BY)
        adjusted_p_values = sorted_p_values * n * np.sum(1.0 / np.arange(1, n + 1)) / np.arange(1, n + 1)
    else:
        raise ValueError("Method must be 'indep' or 'negcorr'")

    # Ensure monotonicity (p-values must increase)
    for i in range(n - 1, 0, -1):
        adjusted_p_values[i - 1] = min(adjusted_p_values[i - 1], adjusted_p_values[i])

    # Cap at 1
    adjusted_p_values = np.minimum(adjusted_p_values, 1.0)

    # Return to original order
    original_order = np.zeros(n, dtype=int)
    original_order[sorted_indices] = np.arange(n)
    adjusted_p_values = adjusted_p_values[original_order]

    return adjusted_p_values


def perform_statistical_analysis(data, metric_type, output_dir, logger):
    """
    Perform statistical analysis comparing phenotypes.

    Parameters:
    -----------
    data : dict
        Dictionary with aggregated data
    metric_type : str
        Type of metric ('alff' or 'reho')
    output_dir : str
        Directory to save results
    logger : Logger
        Logger object

    Returns:
    --------
    dict : Dictionary with statistical results
    """
    try:
        logger.info(f"Performing statistical analysis for {metric_type}")

        # Initialize results structure
        results = {
            'region_level': pd.DataFrame(),
            'network_level': pd.DataFrame()
        }

        # Region-level analysis (comparing phenotypes for each region)
        all_data = data['all_data']

        # Check if data exists
        if all_data.empty:
            logger.warning(f"No {metric_type} data available for statistical analysis")
            return

        # Focus on ADHD vs Control comparison (phenotypes 3 vs 1)
        adhd_data = all_data[all_data['Phenotype'] == 3]
        control_data = all_data[all_data['Phenotype'] == 1]

        # Skip if we don't have both phenotypes
        if adhd_data.empty or control_data.empty:
            logger.warning(f"Missing data for either ADHD or Control in {metric_type}")
            return

        logger.info(
            f"Comparing ADHD (n={adhd_data['Subject'].nunique()}) vs Control (n={control_data['Subject'].nunique()})")

        # Region-level analysis
        region_results = []

        # Get unique regions
        regions = all_data['Region'].unique()

        for region in regions:
            adhd_region = adhd_data[adhd_data['Region'] == region]['Value']
            control_region = control_data[control_data['Region'] == region]['Value']

            # Skip if insufficient data
            if len(adhd_region) < 3 or len(control_region) < 3:
                continue

            # Perform t-test
            t_stat, p_value = stats.ttest_ind(adhd_region, control_region, equal_var=False)

            # Calculate Cohen's d effect size
            adhd_mean = adhd_region.mean()
            control_mean = control_region.mean()
            pooled_std = np.sqrt((adhd_region.std() ** 2 + control_region.std() ** 2) / 2)

            if pooled_std == 0:
                effect_size = 0
            else:
                effect_size = (adhd_mean - control_mean) / pooled_std

            # Get network for this region
            network = all_data[all_data['Region'] == region]['Network'].iloc[0]

            # Store results
            region_results.append({
                'Region': region,
                'Network': network,
                'ADHD_Mean': adhd_mean,
                'Control_Mean': control_mean,
                'ADHD_Std': adhd_region.std(),
                'Control_Std': control_region.std(),
                'T_Statistic': t_stat,
                'P_Value': p_value,
                'Effect_Size': effect_size,
                'ADHD_N': len(adhd_region),
                'Control_N': len(control_region)
            })

        # Convert to DataFrame
        region_df = pd.DataFrame(region_results)

        # Add multiple testing correction
        if not region_df.empty:
            # Use our custom FDR correction function instead of stats.false_discovery_rate_control
            region_df['P_Adjusted'] = fdr_correction(region_df['P_Value'].values, alpha=0.05, method='indep')
            region_df['Significant'] = region_df['P_Adjusted'] < 0.05

            # Sort by p-value
            region_df = region_df.sort_values('P_Value')

            # Save results
            region_output_path = os.path.join(output_dir, f'{metric_type}_region_statistics.csv')
            region_df.to_csv(region_output_path, index=False)
            logger.info(f"Saved region-level statistics to {region_output_path}")

            # Log significant findings
            sig_regions = region_df[region_df['Significant']]
            logger.info(f"Found {len(sig_regions)} significant regions after correction")

            results['region_level'] = region_df

        # Network-level analysis
        network_results = []

        # Get unique networks
        networks = all_data['Network'].unique()

        for network in networks:
            adhd_network = adhd_data[adhd_data['Network'] == network]['Value']
            control_network = control_data[control_data['Network'] == network]['Value']

            # Skip if insufficient data
            if len(adhd_network) < 3 or len(control_network) < 3:
                continue

            # Perform t-test
            t_stat, p_value = stats.ttest_ind(adhd_network, control_network, equal_var=False)

            # Calculate Cohen's d effect size
            adhd_mean = adhd_network.mean()
            control_mean = control_network.mean()
            pooled_std = np.sqrt((adhd_network.std() ** 2 + control_network.std() ** 2) / 2)

            if pooled_std == 0:
                effect_size = 0
            else:
                effect_size = (adhd_mean - control_mean) / pooled_std

            # Store results
            network_results.append({
                'Network': network,
                'ADHD_Mean': adhd_mean,
                'Control_Mean': control_mean,
                'ADHD_Std': adhd_network.std(),
                'Control_Std': control_network.std(),
                'T_Statistic': t_stat,
                'P_Value': p_value,
                'Effect_Size': effect_size,
                'ADHD_N': len(adhd_network),
                'Control_N': len(control_network)
            })

        # Convert to DataFrame
        network_df = pd.DataFrame(network_results)

        # Add multiple testing correction
        if not network_df.empty:
            # Use our custom FDR correction function instead of stats.false_discovery_rate_control
            network_df['P_Adjusted'] = fdr_correction(network_df['P_Value'].values, alpha=0.05, method='indep')
            network_df['Significant'] = network_df['P_Adjusted'] < 0.05

            # Sort by p-value
            network_df = network_df.sort_values('P_Value')

            # Save results
            network_output_path = os.path.join(output_dir, f'{metric_type}_network_statistics.csv')
            network_df.to_csv(network_output_path, index=False)
            logger.info(f"Saved network-level statistics to {network_output_path}")

            # Log significant findings
            sig_networks = network_df[network_df['Significant']]
            logger.info(f"Found {len(sig_networks)} significant networks after correction")

            results['network_level'] = network_df

        return results

    except Exception as e:
        logger.error(f"Error performing statistical analysis: {e}")
        logger.error(traceback.format_exc())
        return None


def create_statistical_plots(stats_results, metric_type, output_dir, logger):
    """
    Create plots visualizing statistical results with enhanced readability.

    Parameters:
    -----------
    stats_results : dict
        Dictionary with statistical results
    metric_type : str
        Type of metric ('alff' or 'reho')
    output_dir : str
        Directory to save visualizations
    logger : Logger
        Logger object

    Returns:
    --------
    None
    """
    try:
        if stats_results is None:
            logger.warning(f"No statistical results available for {metric_type}")
            return

        logger.info(f"Creating statistical plots for {metric_type}")

        # Plot region-level results
        region_df = stats_results.get('region_level')
        if region_df is not None and not region_df.empty:
            # 1. Enhanced Volcano plot with better label placement
            # Create a much larger figure
            plt.figure(figsize=(24, 16))

            # Apply jitter to significant points to reduce overlap
            np.random.seed(42)  # For reproducibility

            # Prepare data
            x = region_df['Effect_Size'].values
            y = -np.log10(region_df['P_Value']).values
            colors = np.array(['gray'] * len(region_df))

            # Identify significant regions
            sig_positive = (region_df['Significant']) & (region_df['Effect_Size'] > 0)
            sig_negative = (region_df['Significant']) & (region_df['Effect_Size'] < 0)

            # Set colors
            colors[sig_positive] = 'red'
            colors[sig_negative] = 'blue'

            # Set marker sizes (larger for significant, smaller for non-significant)
            sizes = np.ones(len(region_df)) * 20  # Default size
            sizes[sig_positive | sig_negative] = 80  # Larger for significant

            # Apply small jitter to significant points to reduce label overlap
            jitter_scale = 0.005  # Small jitter
            x_jitter = x.copy()
            y_jitter = y.copy()

            # Only jitter significant points
            sig_mask = sig_positive | sig_negative
            x_jitter[sig_mask] += np.random.normal(0, jitter_scale, sum(sig_mask))
            y_jitter[sig_mask] += np.random.normal(0, jitter_scale * 5, sum(sig_mask))  # More vertical jitter

            # Create scatter plot
            plt.scatter(
                x_jitter,
                y_jitter,
                c=colors,
                s=sizes,
                alpha=0.7
            )

            # Add labels for top significant regions
            # Get top N most significant positive and negative regions
            top_n = 25  # Label top 25 in each direction

            # Get indices of top significant regions (by p-value)
            if sum(sig_positive) > 0:
                # Filter significant positive regions
                pos_regions = region_df[sig_positive].copy()
                # Sort by p-value
                pos_regions = pos_regions.sort_values('P_Value')
                # Take top N
                top_pos = pos_regions.head(min(top_n, len(pos_regions)))
            else:
                top_pos = pd.DataFrame(columns=region_df.columns)

            if sum(sig_negative) > 0:
                # Filter significant negative regions
                neg_regions = region_df[sig_negative].copy()
                # Sort by p-value
                neg_regions = neg_regions.sort_values('P_Value')
                # Take top N
                top_neg = neg_regions.head(min(top_n, len(neg_regions)))
            else:
                top_neg = pd.DataFrame(columns=region_df.columns)

            # Function to add labels with offsets to avoid overlap
            def add_labels(regions_df, x_offset=0.01, y_offset=0.2):
                for _, row in regions_df.iterrows():
                    region = row['Region']
                    effect = row['Effect_Size']
                    pval = row['P_Value']
                    x_pos = effect
                    y_pos = -np.log10(pval)

                    # Add the label with offset
                    plt.annotate(
                        region,
                        xy=(x_pos, y_pos),
                        xytext=(x_pos + (x_offset if effect > 0 else -x_offset),
                                y_pos + y_offset),
                        fontsize=9,
                        arrowprops=dict(arrowstyle='->', color='black', alpha=0.5, linewidth=0.5)
                    )

            # Add labels for positive and negative effects
            add_labels(top_pos, x_offset=0.01, y_offset=0.2)
            add_labels(top_neg, x_offset=-0.01, y_offset=0.2)

            # Add significance threshold line
            plt.axhline(-np.log10(0.05), linestyle='--', color='gray', alpha=0.7)

            plt.title(f'{metric_type.upper()} Region-Level Volcano Plot (ADHD vs Control)', fontsize=18)
            plt.xlabel('Effect Size (Cohen\'s d)', fontsize=14)
            plt.ylabel('-log10(p-value)', fontsize=14)

            # Add a legend
            plt.legend([
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10)
            ], ['Higher in ADHD (sig)', 'Lower in ADHD (sig)', 'Not significant'],
                loc='upper right', fontsize=12)

            # Draw vertical line at zero
            plt.axvline(0, color='black', linestyle='-', alpha=0.3)

            plt.tight_layout()

            # Save the figure
            output_path = os.path.join(output_dir, f'{metric_type}_region_volcano_plot.png')
            plt.savefig(output_path, dpi=300)
            plt.close()
            logger.info(f"Saved enhanced region volcano plot to {output_path}")

            # 2. Create a high-resolution version for very detailed viewing
            plt.figure(figsize=(36, 24))  # Even larger for detailed inspection

            # Create scatter plot with enhanced aesthetics
            plt.scatter(
                x_jitter,
                y_jitter,
                c=colors,
                s=sizes * 1.5,  # Slightly larger markers
                alpha=0.7,
                edgecolors='none'  # No edges for cleaner look
            )

            # Function to add labels with varied offsets to minimize overlap
            def add_detailed_labels(regions_df, direction='positive'):
                # Calculate base offsets based on direction
                if direction == 'positive':
                    base_x_offset = 0.01
                    angle = 45  # Angled text for positive side
                else:
                    base_x_offset = -0.01
                    angle = -45  # Angled text for negative side

                # Check if P_Value column exists
                if 'P_Value' not in regions_df.columns:
                    logger.warning(
                        f"P_Value column not found in DataFrame for {direction} labels. Available columns: {regions_df.columns.tolist()}")
                    # Use original DataFrame without sorting if P_Value column is missing
                    sorted_df = regions_df
                else:
                    # Sort by p-value for staggered label placement
                    sorted_df = regions_df.sort_values('P_Value')

                # Add labels with staggered offsets
                for i, (_, row) in enumerate(sorted_df.iterrows()):
                    region = row['Region']
                    effect = row['Effect_Size']
                    pval = row['P_Value']
                    x_pos = effect
                    y_pos = -np.log10(pval)

                    # Stagger vertical offsets
                    y_offset = 0.2 + (i % 3) * 0.1  # Alternate offsets

                    # Adjust horizontal offset based on positioning
                    x_offset = base_x_offset * (1 + (i % 2) * 0.5)

                    # Add the label with offset
                    plt.annotate(
                        region,
                        xy=(x_pos, y_pos),
                        xytext=(x_pos + x_offset, y_pos + y_offset),
                        fontsize=8,
                        rotation=angle,
                        arrowprops=dict(arrowstyle='->', color='black', alpha=0.4, linewidth=0.5)
                    )

            # Add labels for positive and negative effects with better positioning
            add_detailed_labels(top_pos, direction='positive')
            add_detailed_labels(top_neg, direction='negative')

            # Add significance threshold line
            plt.axhline(-np.log10(0.05), linestyle='--', color='gray', alpha=0.7)

            plt.title(f'{metric_type.upper()} Region-Level Volcano Plot (ADHD vs Control) - Detailed View', fontsize=20)
            plt.xlabel('Effect Size (Cohen\'s d)', fontsize=16)
            plt.ylabel('-log10(p-value)', fontsize=16)

            # Add a legend
            plt.legend([
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=12),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8)
            ], ['Higher in ADHD (sig)', 'Lower in ADHD (sig)', 'Not significant'],
                loc='upper right', fontsize=14)

            # Draw vertical line at zero
            plt.axvline(0, color='black', linestyle='-', alpha=0.3)

            # Add minor gridlines for better readability
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.tight_layout()

            # Save the figure at higher resolution
            output_path = os.path.join(output_dir, f'{metric_type}_region_volcano_plot_detailed.png')
            plt.savefig(output_path, dpi=300)
            plt.close()
            logger.info(f"Saved detailed region volcano plot to {output_path}")

            # 3. Create separate plots for positive and negative effects (to reduce overlap)
            for effect_type, effect_name, color in [('positive', 'Higher', 'red'), ('negative', 'Lower', 'blue')]:
                plt.figure(figsize=(20, 14))

                # Filter data based on effect direction
                if effect_type == 'positive':
                    mask = region_df['Effect_Size'] > 0
                else:
                    mask = region_df['Effect_Size'] < 0

                sub_df = region_df[mask]

                # Skip if no data
                if len(sub_df) == 0:
                    plt.close()
                    continue

                # Prepare data for this plot
                x = sub_df['Effect_Size'].values
                y = -np.log10(sub_df['P_Value']).values
                is_sig = sub_df['Significant'].values

                # Define marker sizes and colors
                sizes = np.ones(len(sub_df)) * 30  # Default size
                sizes[is_sig] = 100  # Larger for significant

                colors_array = np.array(['lightgray'] * len(sub_df))
                colors_array[is_sig] = color

                # Apply jitter
                jitter_scale = 0.003  # Small jitter
                x_jitter = x.copy()
                y_jitter = y.copy()

                # Jitter only significant points for clarity
                x_jitter[is_sig] += np.random.normal(0, jitter_scale, sum(is_sig))
                y_jitter[is_sig] += np.random.normal(0, jitter_scale * 3, sum(is_sig))

                # Create scatter plot
                plt.scatter(
                    x_jitter,
                    y_jitter,
                    c=colors_array,
                    s=sizes,
                    alpha=0.8
                )

                # Label significant points
                if sum(is_sig) > 0:
                    sig_df = sub_df[is_sig].copy()
                    # Sort by p-value
                    sig_df = sig_df.sort_values('P_Value')

                    # Label top regions using staggered approach
                    max_labels = min(40, len(sig_df))  # Label up to 40 significant regions

                    for i, (_, row) in enumerate(sig_df.head(max_labels).iterrows()):
                        region = row['Region']
                        effect = row['Effect_Size']
                        pval = row['P_Value']
                        x_pos = effect
                        y_pos = -np.log10(pval)

                        # Use different offsets to minimize overlap
                        angle = 0  # Horizontal text
                        if effect_type == 'positive':
                            x_offset = 0.01 * (1 + (i % 3) * 0.5)
                            y_offset = 0.15 + (i % 4) * 0.1
                        else:
                            x_offset = -0.01 * (1 + (i % 3) * 0.5)
                            y_offset = 0.15 + (i % 4) * 0.1

                        # Add label with arrow
                        plt.annotate(
                            region,
                            xy=(x_pos, y_pos),
                            xytext=(x_pos + x_offset, y_pos + y_offset),
                            fontsize=9,
                            arrowprops=dict(arrowstyle='->', color='black', alpha=0.5, linewidth=0.5)
                        )

                # Add significance threshold line
                plt.axhline(-np.log10(0.05), linestyle='--', color='gray', alpha=0.7)

                plt.title(f'{metric_type.upper()} Regions {effect_name} in ADHD vs Control', fontsize=18)
                plt.xlabel('Effect Size (Cohen\'s d)', fontsize=14)
                plt.ylabel('-log10(p-value)', fontsize=14)

                # Add legend
                plt.legend([
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=12),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=8)
                ], [f'Significant ({effect_name} in ADHD)', 'Not significant'],
                    loc='upper right', fontsize=12)

                # Add minor gridlines
                plt.grid(True, linestyle='--', alpha=0.3)

                plt.tight_layout()

                # Save the figure
                output_path = os.path.join(output_dir, f'{metric_type}_region_volcano_{effect_type}.png')
                plt.savefig(output_path, dpi=300)
                plt.close()
                logger.info(f"Saved {effect_type} effect volcano plot to {output_path}")

            # 4. Network effect size plot (with improved aesthetics)
            # Continue with the network-level effect size plot as before, with slight enhancements
            plt.figure(figsize=(12, 10))

            # Define network colors
            network_colors = {
                'Visual1': '#000023',
                'Visual2': '#1E90FF',
                'Somatomotor': '#32CD32',
                'Cingulo-Opercular': '#FFD700',
                'Dorsal_Attention': '#12670f',
                'Language': '#FF69B4',
                'Frontoparietal': '#FF6347',
                'Auditory': '#00CED1',
                'Posterior_Multimodal': '#9400D3',
                'Default': '#800000',
                'Orbito-Affective': '#FF8C00',
                'Ventral_Multimodal': '#8B4513',
                'Subcortex': '#808080',
                'Unknown': '#CCCCCC'
            }

            # Sort by network and effect size
            sorted_df = region_df.sort_values(['Network', 'Effect_Size'])

            # Create positions for bars
            positions = np.arange(len(sorted_df))

            # Plot bars with enhanced aesthetics
            bars = plt.bar(
                positions,
                sorted_df['Effect_Size'],
                color=[network_colors.get(net, '#CCCCCC') for net in sorted_df['Network']],
                alpha=0.8,
                width=0.8,  # Slightly thinner bars
                edgecolor='none'  # No edges
            )

            # Add a thin gray line at zero
            plt.axhline(0, color='black', linestyle='-', alpha=0.4, linewidth=0.8)

            # Highlight significant regions
            for i, is_sig in enumerate(sorted_df['Significant']):
                if is_sig:
                    bars[i].set_edgecolor('black')
                    bars[i].set_linewidth(1.2)

            plt.title(f'{metric_type.upper()} Region-Level Effect Sizes by Network (ADHD vs Control)', fontsize=16)
            plt.xlabel('Region', fontsize=13)
            plt.ylabel('Effect Size (Cohen\'s d)', fontsize=13)
            plt.xticks([])  # Hide x-tick labels due to large number

            # Add a legend for networks
            handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.8) for network, color in network_colors.items()
                       if network in sorted_df['Network'].values]
            labels = [network for network in network_colors.keys() if network in sorted_df['Network'].values]

            plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

            plt.tight_layout()

            # Save the figure
            output_path = os.path.join(output_dir, f'{metric_type}_region_effect_size_plot.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved enhanced region effect size plot to {output_path}")

        # Plot network-level results with improved aesthetics
        network_df = stats_results.get('network_level')
        if network_df is not None and not network_df.empty:
            # Network effect size plot
            plt.figure(figsize=(14, 8))

            # Define network colors
            network_colors = {
                'Visual1': '#000023',
                'Visual2': '#1E90FF',
                'Somatomotor': '#32CD32',
                'Cingulo-Opercular': '#FFD700',
                'Dorsal_Attention': '#12670f',
                'Language': '#FF69B4',
                'Frontoparietal': '#FF6347',
                'Auditory': '#00CED1',
                'Posterior_Multimodal': '#9400D3',
                'Default': '#800000',
                'Orbito-Affective': '#FF8C00',
                'Ventral_Multimodal': '#8B4513',
                'Subcortex': '#808080',
                'Unknown': '#CCCCCC'
            }

            # Sort by effect size
            sorted_df = network_df.sort_values('Effect_Size')

            # Plot bars with enhanced aesthetics
            bars = plt.bar(
                sorted_df['Network'],
                sorted_df['Effect_Size'],
                color=[network_colors.get(net, '#CCCCCC') for net in sorted_df['Network']],
                alpha=0.8,
                width=0.7  # Slightly thinner bars
            )

            # Add a thin gray line at zero
            plt.axhline(0, color='black', linestyle='-', alpha=0.4, linewidth=0.8)

            # Highlight significant networks
            for i, (_, row) in enumerate(sorted_df.iterrows()):
                if row['Significant']:
                    bars[i].set_edgecolor('black')
                    bars[i].set_linewidth(1.5)

                    # Add a star above significant bars
                    height = row['Effect_Size']
                    plt.text(i, height + (0.02 if height > 0 else -0.02),
                             '*', ha='center', fontsize=16, color='black')

            # Add error bars
            plt.errorbar(
                x=np.arange(len(sorted_df)),
                y=sorted_df['Effect_Size'],
                yerr=sorted_df[['ADHD_Std', 'Control_Std']].mean(axis=1) / np.sqrt(
                    sorted_df[['ADHD_N', 'Control_N']].mean(axis=1)),
                fmt='none',
                ecolor='black',
                capsize=5,
                alpha=0.7,
                linewidth=1.2
            )

            plt.title(f'{metric_type.upper()} Network-Level Effect Sizes (ADHD vs Control)', fontsize=16)
            plt.xlabel('Network', fontsize=14)
            plt.ylabel('Effect Size (Cohen\'s d)', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.yticks(fontsize=11)

            # Add gridlines
            plt.grid(axis='y', linestyle='--', alpha=0.3)

            plt.tight_layout()

            # Save the figure
            output_path = os.path.join(output_dir, f'{metric_type}_network_effect_size_plot.png')
            plt.savefig(output_path, dpi=300)
            plt.close()
            logger.info(f"Saved enhanced network effect size plot to {output_path}")

    except Exception as e:
        logger.error(f"Error creating statistical plots: {e}")
        logger.error(traceback.format_exc())

def process_all_subjects(subject_list, exclude_list, task_label, data_dir1, data_dir2, output_dir, cole_mapping,
                         phenotype_map, raw_phenotype_map, logger):
    """
    Process all subjects in parallel and create group-level visualizations.

    Parameters:
    -----------
    subject_list : list
        List of subject IDs to process
    exclude_list : list
        List of subject IDs to exclude
    task_label : str
        Task label
    data_dir1 : str
        Path to uncensored XCP-D output directory (for ALFF)
    data_dir2 : str
        Path to censored XCP-D output directory (for ReHo)
    output_dir : str
        Directory to save visualizations
    cole_mapping : pandas.DataFrame
        Cole network mapping DataFrame
    phenotype_map : dict
        Dictionary mapping subject IDs to phenotypes
    raw_phenotype_map : dict
        Dictionary with raw phenotype mappings for debugging
    logger : Logger
        Logger object
    """
    # Filter out completely excluded subjects
    if exclude_list:
        subject_list = [sub for sub in subject_list if sub not in exclude_list]
        logger.info(f"Excluded {len(exclude_list)} subjects, processing remaining {len(subject_list)} subjects")

    # Initialize list for all subjects
    all_subject_data = []

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
                data_dir1=data_dir1,
                data_dir2=data_dir2,
                cole_mapping=cole_mapping,
                phenotype_map=phenotype_map,
                raw_phenotype_map=raw_phenotype_map,
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
                    if result and (result['alff'] is not None or result['reho'] is not None):
                        # Successfully processed
                        all_subject_data.append(result)
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

        # Create output directories
        alff_dir = os.path.join(output_dir, "alff")
        reho_dir = os.path.join(output_dir, "reho")
        os.makedirs(alff_dir, exist_ok=True)
        os.makedirs(reho_dir, exist_ok=True)

        # Aggregate data by phenotype
        aggregated = aggregate_data_by_phenotype(all_subject_data, logger)

        # Define phenotype order for visualization
        phenotype_order = [1, 2, 3, 4, None]  # Control, Subthreshold, ADHD, Not Clean Control, Unknown

        # Create visualizations for ALFF
        if not aggregated['alff']['all_data'].empty:
            # Region-level scatter plot
            create_region_scatter_plot(
                aggregated['alff']['all_data'],
                phenotype_order,
                'alff',
                alff_dir,
                logger
            )

            # Network-level bar plot
            create_network_bar_plot(
                aggregated['alff']['by_network'],
                phenotype_order,
                'alff',
                alff_dir,
                logger
            )

            # Statistical analysis
            alff_stats = perform_statistical_analysis(
                aggregated['alff'],
                'alff',
                alff_dir,
                logger
            )

            # Create statistical plots
            create_statistical_plots(
                alff_stats,
                'alff',
                alff_dir,
                logger
            )

        # Create visualizations for ReHo
        if not aggregated['reho']['all_data'].empty:
            # Region-level scatter plot
            create_region_scatter_plot(
                aggregated['reho']['all_data'],
                phenotype_order,
                'reho',
                reho_dir,
                logger
            )

            # Network-level bar plot
            create_network_bar_plot(
                aggregated['reho']['by_network'],
                phenotype_order,
                'reho',
                reho_dir,
                logger
            )

            # Statistical analysis
            reho_stats = perform_statistical_analysis(
                aggregated['reho'],
                'reho',
                reho_dir,
                logger
            )

            # Create statistical plots
            create_statistical_plots(
                reho_stats,
                'reho',
                reho_dir,
                logger
            )

    # Create aggregated CSVs for ML
    alff_csv = create_aggregated_csv(all_subject_data, 'alff', output_dir, logger)
    reho_csv = create_aggregated_csv(all_subject_data, 'reho', output_dir, logger)

    if alff_csv:
        logger.info(f"ML-ready ALFF data available at: {alff_csv}")
    if reho_csv:
        logger.info(f"ML-ready ReHo data available at: {reho_csv}")

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
    logger.info("Starting Oregon ALFF and ReHo Analysis Script")

    try:
        # Log system information
        logger.info(
            f"System information: {os.cpu_count()} CPUs, {psutil.virtual_memory().total / (1024 ** 3):.1f} GB RAM")
        logger.info(f"Using up to {MAX_WORKERS} parallel workers")

        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)

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
            # Find all subjects in the data directories (search both)
            uncensored_subjects = find_all_subjects(DATA_DIR1, logger)
            censored_subjects = find_all_subjects(DATA_DIR2, logger)

            # Combine subjects from both directories
            subject_list = list(set(uncensored_subjects + censored_subjects))
            logger.info(f"Found {len(subject_list)} unique subjects across both directories")

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
        exclude_list = None
        if SUBJECT_EXCLUDE is not None:
            exclude_list = [clean_subject_id(sub) for sub in SUBJECT_EXCLUDE]
            exclude_list = list(set(exclude_list))
            logger.info(f"Found {len(exclude_list)} subjects to completely exclude")

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
            exclude_list=exclude_list,
            task_label=TASK,
            data_dir1=DATA_DIR1,
            data_dir2=DATA_DIR2,
            output_dir=OUTPUT_DIR,
            cole_mapping=cole_mapping,
            phenotype_map=phenotype_map,
            raw_phenotype_map=raw_phenotype_map,
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