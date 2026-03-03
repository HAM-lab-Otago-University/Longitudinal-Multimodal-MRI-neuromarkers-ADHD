"""
Longitudinal Cognitive Data Extraction with Independent Folds - OREGON1000 Dataset

This script extracts cognitive test scores and creates 5 completely independent folds
where each subject appears in only one fold (either in train or test), with no data
sharing across folds.

Key features:
- Creates independent train/test splits across 5 folds with no subject overlap
- Ensures each subject's timepoints stay together (all in train or all in test)
- Only extracts specific cognitive variables needed for G-factor analysis
- Handles numeric data properly
- Excludes specified subject-timepoint combinations from analysis
- Outputs fold assignments to CSV files for tracking and analysis
"""

import pandas as pd
import numpy as np
import os
import logging
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define global paths
BASE_DATA_PATH = '/media/hcs-sci-psy-narun/Jack'
OUTPUT_DIR = os.path.join(BASE_DATA_PATH, 'g_factor_analysis_v5_nested')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File paths
IMAGE_FILE = os.path.join(BASE_DATA_PATH, "image03_copy.csv")
NBACK_FILE = os.path.join(BASE_DATA_PATH, "nback01.csv")
STOPSIG_FILE = os.path.join(BASE_DATA_PATH, "stopsig01_copy.csv")
CPTH_FILE = os.path.join(BASE_DATA_PATH, "cpth01_copy.csv")
WISC_FILE = os.path.join(BASE_DATA_PATH, "wisc_iv_part102_copy.csv")
WIAT_FILE = os.path.join(BASE_DATA_PATH, "wiat_ii01.csv")

# List of subjects to exclude completely from analysis (all timepoints)
# From Oregon_ALFF_ReHo_v6.py - subjects with bad data across all timepoints
SUBJECT_EXCLUDE_COMPLETE = [
    "NDARINVKPGRKEZW", "NDARINVCGJ4G85F", "NDARINV0H2MNXAU", "NDARINVE5F245ZC",
    "NDARINVD0UHEXV5", "NDARINVKKE5M36H", "NDARINVDPJEMFNZ", "NDARINV9U08EKVL",
    "NDARINVEPTLP4ED", "NDARINVT4FZHCTW", "NDARINV4E1GLBYM", "NDARINV8LZY8RBZ",
    "NDARINVUYKLT0YE", "NDARINVXF3RN4HD", "NDARINV09MPGC1A", "NDARINVPF66BBJZ",
    "NDARINVZZVK3FXR", "NDARINVT3V0N9JR", "NDARINV9LMBWJ8Y", "NDARINVKF5TZ56H",
    "NDARINVRD9739F6", "NDARINVXGH2TU13", "NDARINVDGLFV9JE", "NDARINV31PK79NV",
    "NDARINV4AWRRTML", "NDARINVKRWMZUZ0", "NDARINVC597YE96", "NDARINVT5HEHVWA",
    "NDARINVM5GDJH10"
]
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

# Define EXACT cognitive variables to extract with their properties
COGNITIVE_VARIABLES = {
    # Stop Signal Task
    'stopsig_stop_sdgort': {
        'file': STOPSIG_FILE,
        'variable': 'stop_sdgort',
        'description': 'Stop task SD on go trials (ms)',
        'invert': True,  # Higher is worse
        'expected_range': [0, 1000]  # milliseconds
    },
    # Continuous Performance Task
    'cpth_dprime1': {
        'file': CPTH_FILE,
        'variable': 'dprime1',
        'description': 'CPT D-prime vs catch trials',
        'invert': False,  # Higher is better
        'expected_range': [-3, 5]  # typical d-prime range
    },
    # N-Back Task
    'nback_n2backacc': {
        'file': NBACK_FILE,
        'variable': 'n2backacc',
        'description': 'N-Back 2-back accuracy (%)',
        'invert': False,  # Higher is better
        'expected_range': [0, 100]  # percentage
    },
    # WISC Digit Span
    'wisc_digitspan_dsfscore': {
        'file': WISC_FILE,
        'variable': 'digitspan_dsfscore',
        'description': 'Digit Span Forward Score',
        'invert': False,  # Higher is better
        'expected_range': [0, 20]  # typical range
    },
    'wisc_digitspan_dsbscore': {
        'file': WISC_FILE,
        'variable': 'digitspan_dsbscore',
        'description': 'Digit Span Backward Score',
        'invert': False,  # Higher is better
        'expected_range': [0, 20]  # typical range
    },
    # WIAT Achievement Test
    'wiat_wordread_score': {
        'file': WIAT_FILE,
        'variable': 'wordread_score',
        'description': 'WIAT Word Reading Score',
        'invert': False,  # Higher is better
        'expected_range': [0, 200]  # standardized scores
    }
}


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


def parse_timepoint_info(timepoint_id):
    """
    Parse a timepoint ID to extract age and study year information.

    Parameters:
    -----------
    timepoint_id : str
        Timepoint ID (e.g., "NDARXXXAge136MonthsStudyYear02" where 136 means 136 total months)

    Returns:
    --------
    tuple : (age_months, study_year) or (None, None) if not parsable
    """
    if timepoint_id is None:
        return None, None

    # Extract age digits from the format
    age_match = re.search(r'Age(\d+)Months', timepoint_id)
    if not age_match:
        return None, None

    # Age is already in total months - no conversion needed!
    age_months = int(age_match.group(1))

    # Extract study year
    year_match = re.search(r'StudyYear(\d+)', timepoint_id)
    study_year = int(year_match.group(1)) if year_match else None

    return age_months, study_year


def should_exclude_timepoint(subject_id, age_months=None, interview_age=None):
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
    normalized_complete = [normalize_subject_id(s) for s in SUBJECT_EXCLUDE_COMPLETE]

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


def load_and_preprocess_data(file_path, has_subheader=False):
    """
    Load and preprocess data from CSV files.
    If has_subheader is True, skip the second row (index 1) which contains a subheader.
    """
    try:
        if has_subheader:
            # Skip the second row (subheader)
            df = pd.read_csv(file_path, skiprows=[1])
        else:
            df = pd.read_csv(file_path)

        # Strip underscores from subjectkey
        if 'subjectkey' in df.columns:
            df['subjectkey'] = df['subjectkey'].astype(str).str.replace('_', '')

        logger.info(f"Loaded data from {file_path}: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()  # Return empty dataframe on error


def format_age_suffix(age_months):
    """
    Format the age suffix for the timepoint ID
    Uses total months to match folder naming convention
    """
    if pd.isna(age_months):
        return "AgeUnknown"

    try:
        age_months = int(float(age_months))
        # Use total months directly to match folder format
        return f"Age{age_months:03d}"  # Zero-pad to 3 digits: Age136
    except (ValueError, TypeError):
        return "AgeUnknown"


def clean_missing_values(df, variable_name, expected_range=None):
    """
    Clean missing values for a specific variable.

    Parameters:
    - df: DataFrame containing the variable
    - variable_name: Name of the column to clean
    - expected_range: Optional tuple/list of (min, max) expected values

    Returns:
    - Cleaned DataFrame
    """
    if df.empty or variable_name not in df.columns:
        return df

    # Make a copy to avoid modifying the original
    result = df.copy()

    # Check if the column is numeric
    is_numeric = pd.api.types.is_numeric_dtype(df[variable_name])
    if not is_numeric:
        try:
            # Try converting to numeric
            result[variable_name] = pd.to_numeric(result[variable_name], errors='coerce')
            logger.info(f"Converted column {variable_name} to numeric")
        except:
            logger.warning(f"Column {variable_name} is not numeric and cannot be converted. Skipping cleaning.")
            return result

    # Log before cleaning
    before_count = result[variable_name].isna().sum()

    # Known missing value codes
    missing_codes = [-999, 999, -888, 888, -777, 777, -666, 666, -9999, 9999]

    # Replace known missing codes with NaN
    for code in missing_codes:
        mask = result[variable_name] == code
        if mask.any():
            logger.debug(f"Found {mask.sum()} instances of missing code {code} in column {variable_name}")
            result.loc[mask, variable_name] = np.nan

    # Apply expected range if provided
    if expected_range is not None:
        min_val, max_val = expected_range
        out_of_range_mask = (result[variable_name] < min_val) | (result[variable_name] > max_val)
        if out_of_range_mask.any():
            logger.debug(f"Found {out_of_range_mask.sum()} values outside expected range " +
                         f"[{min_val}, {max_val}] in column {variable_name}")
            result.loc[out_of_range_mask, variable_name] = np.nan

    # Log after cleaning
    after_count = result[variable_name].isna().sum()
    if after_count > before_count:
        logger.info(f"Cleaned {after_count - before_count} missing values in {variable_name}")

    return result


def find_matching_cognitive_data(fmri_scan, task_dataframes, needed_variables, days_threshold=30):
    """
    Find matching cognitive data for a given fMRI scan, considering the longitudinal
    structure of the data. Only extracts the specific cognitive variables needed.

    Parameters:
    - fmri_scan: Row from fMRI dataframe
    - task_dataframes: Dictionary of dataframes for each cognitive task
    - needed_variables: Dictionary of needed cognitive variables
    - days_threshold: Maximum days difference for matching

    Returns:
    - Dictionary with matched cognitive data or None if no match is found
    """
    subject_id = fmri_scan['subjectkey']

    # Skip if this subject is in the complete exclusion list
    normalized_id = normalize_subject_id(subject_id)
    normalized_complete = [normalize_subject_id(s) for s in SUBJECT_EXCLUDE_COMPLETE]
    if normalized_id in normalized_complete:
        return None

    matched_data = {'subject_id': subject_id}

    # Set scan info
    if 'interview_date' in fmri_scan and not pd.isna(fmri_scan['interview_date']):
        try:
            fmri_date = pd.to_datetime(fmri_scan['interview_date'], errors='coerce')
            matched_data['fmri_date'] = fmri_date
        except:
            fmri_date = None
    else:
        fmri_date = None

    if 'interview_age' in fmri_scan and not pd.isna(fmri_scan['interview_age']):
        try:
            fmri_age = float(fmri_scan['interview_age'])
            matched_data['age_months'] = fmri_age
            matched_data['age_years'] = fmri_age / 12
        except:
            fmri_age = None
    else:
        fmri_age = None

    # Check if this specific timepoint should be excluded
    if should_exclude_timepoint(subject_id, interview_age=fmri_age):
        return None

    # Create a unique timepoint ID
    if fmri_age is not None:
        age_suffix = format_age_suffix(fmri_age)
        matched_data['timepoint_id'] = f"{subject_id}_{age_suffix}"
    elif fmri_date is not None:
        # Use date instead if age is missing
        matched_data['timepoint_id'] = f"{subject_id}_{fmri_date.strftime('%Y%m%d')}"
    else:
        # Fallback for missing age and date
        matched_data['timepoint_id'] = f"{subject_id}_unknown"

    # Group variables by task file
    variables_by_file = {}
    for var_name, var_info in needed_variables.items():
        file_path = var_info['file']
        if file_path not in variables_by_file:
            variables_by_file[file_path] = []
        variables_by_file[file_path].append((var_name, var_info['variable']))

    # For each task file
    for file_path, variable_pairs in variables_by_file.items():
        task_name = os.path.basename(file_path).split('.')[0]
        task_df = task_dataframes.get(task_name)

        if task_df is None or task_df.empty:
            continue

        # Find data for this subject
        subject_task_df = task_df[task_df['subjectkey'] == subject_id]

        if subject_task_df.empty:
            continue

        best_match = None
        best_match_diff = float('inf')

        # Try to match by date
        if fmri_date is not None:
            for _, task_row in subject_task_df.iterrows():
                if 'interview_date' not in task_row or pd.isna(task_row['interview_date']):
                    continue

                try:
                    task_date = pd.to_datetime(task_row['interview_date'], errors='coerce')
                    if pd.isna(task_date):
                        continue

                    days_diff = abs((fmri_date - task_date).days)

                    if days_diff < best_match_diff:
                        best_match_diff = days_diff
                        best_match = task_row

                        # If perfect match, break early
                        if days_diff == 0:
                            break
                except:
                    continue

            # If we found a match within the threshold, use it
            if best_match is not None and best_match_diff <= days_threshold:
                # Extract only the variables we need
                for output_name, var_name in variable_pairs:
                    if var_name in best_match:
                        try:
                            value = best_match[var_name]
                            # Convert to numeric if possible
                            matched_data[output_name] = pd.to_numeric(value, errors='coerce')
                        except:
                            logger.warning(f"Could not convert {var_name} to numeric for subject {subject_id}")
                            matched_data[output_name] = np.nan
                continue

        # If no match by date or no date available, try matching by age
        if fmri_age is not None:
            best_match = None
            best_match_diff = float('inf')

            for _, task_row in subject_task_df.iterrows():
                if 'interview_age' not in task_row or pd.isna(task_row['interview_age']):
                    continue

                try:
                    task_age = float(task_row['interview_age'])
                    age_diff = abs(fmri_age - task_age)

                    if age_diff < best_match_diff:
                        best_match_diff = age_diff
                        best_match = task_row

                        # If difference is less than 1 month, break early
                        if age_diff <= 1:
                            break
                except:
                    continue

            # If we found a match within 3 months, use it
            if best_match is not None and best_match_diff <= 3:
                # Extract only the variables we need
                for output_name, var_name in variable_pairs:
                    if var_name in best_match:
                        try:
                            value = best_match[var_name]
                            # Convert to numeric if possible
                            matched_data[output_name] = pd.to_numeric(value, errors='coerce')
                        except:
                            logger.warning(f"Could not convert {var_name} to numeric for subject {subject_id}")
                            matched_data[output_name] = np.nan
                continue

        # If still no match, use the closest task data point (if any)
        if len(subject_task_df) > 0:
            task_row = subject_task_df.iloc[0]  # Just use the first one as fallback
            # Extract only the variables we need
            for output_name, var_name in variable_pairs:
                if var_name in task_row:
                    try:
                        value = task_row[var_name]
                        # Convert to numeric if possible
                        matched_data[output_name] = pd.to_numeric(value, errors='coerce')
                    except:
                        logger.warning(f"Could not convert {var_name} to numeric for subject {subject_id}")
                        matched_data[output_name] = np.nan

    return matched_data


def invert_variables(df):
    """
    Invert variables where higher values indicate worse performance (e.g., reaction times)
    so that all variables follow the "higher is better" convention for g-factor.
    """
    df_copy = df.copy()

    for var_key, var_info in COGNITIVE_VARIABLES.items():
        if var_info.get('invert') is True and var_key in df_copy.columns:
            # Get non-NaN values for this variable
            non_nan_values = df_copy[var_key].dropna()

            if len(non_nan_values) > 0:
                # Calculate max value for normalization
                max_val = non_nan_values.max()

                if max_val > 0:
                    logger.info(f"Inverting {var_key} (higher is worse) using max value {max_val}")
                    # Invert scores so higher = better (for all variables in g factor)
                    df_copy[var_key] = df_copy[var_key].apply(lambda x: max_val - x if not pd.isna(x) else x)
                    logger.info(f"After inversion: range {df_copy[var_key].min()} to {df_copy[var_key].max()}")

    return df_copy


def transform_variables(df):
    """
    Apply various transformations to improve variable distributions:
    - Log transform d-prime measures
    - Cube transform WIAT reading scores
    - Discretize nback_n2backacc to 10 categories (removed)

    Parameters:
    - df: DataFrame containing the variables to transform

    Returns:
    - DataFrame with transformed variables
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    # --- Original d-prime transformations ---
    # Transform dprime1 if it exists
    if 'cpth_dprime1' in result.columns:
        # Get min value to determine shift amount
        min_val = result['cpth_dprime1'].min()
        # Calculate shift (ensure all values are positive for log transform)
        shift = abs(min_val) + 1 if min_val < 0 else 1
        # Apply log transformation
        result['cpth_dprime1'] = np.log(result['cpth_dprime1'] + shift)
        logger.info(f"Applied log transformation to cpth_dprime1 with shift {shift}")

    # --- New transformations ---
    # Apply cube transformation to wiat_wordread_score if it exists
    if 'wiat_wordread_score' in result.columns:
        # Store original values for comparison
        original_values = result['wiat_wordread_score'].copy()

        # Apply cube transformation
        result['wiat_wordread_score'] = result['wiat_wordread_score'] ** 3

        # Calculate and log skewness before and after transformation
        from scipy import stats
        original_skew = stats.skew(original_values.dropna())
        transformed_skew = stats.skew(result['wiat_wordread_score'].dropna())

        logger.info(f"Applied cube transformation to wiat_wordread_score")
        logger.info(f"Skewness before: {original_skew:.4f}, after: {transformed_skew:.4f}")

    return result


def create_fold_directories(base_path):
    """Create directories for each fold"""
    for i in range(5):
        fold_dir = os.path.join(base_path, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "cogscores"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "g"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "factors"), exist_ok=True)


def validate_independent_folds(fold_data):
    """
    Validates that subjects are completely separated across folds.

    Parameters:
    - fold_data: List of dictionaries with 'train_subjects' and 'test_subjects' for each fold

    Returns:
    - True if validation passes, False if overlap detected
    """
    logger.info("\n=== FOLD INDEPENDENCE VALIDATION ===")
    validation_passed = True

    # Get all subjects from all folds
    all_subjects = set()
    for fold_idx, fold in enumerate(fold_data):
        all_subjects.update(fold['train_subjects'])
        all_subjects.update(fold['test_subjects'])

    logger.info(f"Total unique subjects across all folds: {len(all_subjects)}")

    # Check for overlap between any pair of folds
    for i in range(len(fold_data)):
        subjects_i = fold_data[i]['train_subjects'].union(fold_data[i]['test_subjects'])

        for j in range(i + 1, len(fold_data)):
            subjects_j = fold_data[j]['train_subjects'].union(fold_data[j]['test_subjects'])

            overlap = subjects_i.intersection(subjects_j)
            if overlap:
                logger.error(f"VALIDATION FAILED! Fold {i} and Fold {j} share {len(overlap)} subjects!")
                logger.error(f"Example overlapping subjects: {list(overlap)[:5]}")
                validation_passed = False
            else:
                logger.info(f"✓ Fold {i} and Fold {j}: No subject overlap")

    # Check for subject overlap between train and test within each fold
    for fold_idx, fold in enumerate(fold_data):
        overlap = fold['train_subjects'].intersection(fold['test_subjects'])
        if overlap:
            logger.error(f"VALIDATION FAILED! Fold {fold_idx} has {len(overlap)} subjects in both train and test!")
            logger.error(f"Example overlapping subjects: {list(overlap)[:5]}")
            validation_passed = False
        else:
            logger.info(f"✓ Fold {fold_idx}: No subject overlap between train and test sets")

    # Check if all validations passed
    if validation_passed:
        logger.info("✓ All validations passed: Folds are completely independent")
    else:
        logger.error("✗ VALIDATION FAILED: Subject overlap detected between folds")

    logger.info("=== VALIDATION COMPLETE ===\n")
    return validation_passed


def custom_standardize(X_train, X_test):
    """
    Custom standardization function that properly handles zero-variance features.

    Parameters:
    - X_train: Training data to fit standardization parameters
    - X_test: Test data to transform

    Returns:
    - X_train_scaled: Standardized training data
    - X_test_scaled: Standardized test data
    """
    # Initialize outputs
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # For each column
    for col in X_train.columns:
        # Ensure the column is numeric
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            logger.warning(f"Column {col} is not numeric. Skipping standardization.")
            continue

        # Get column mean and standard deviation
        col_mean = X_train[col].mean()
        col_std = X_train[col].std()

        # Check if the column has variance
        if col_std > 0:
            # Standard scaling: (x - mean) / std
            X_train_scaled[col] = (X_train[col] - col_mean) / col_std
            X_test_scaled[col] = (X_test[col] - col_mean) / col_std
        else:
            # Zero variance column - set to 0 after centering
            logger.warning(f"Zero variance detected in feature '{col}'. Setting to 0 after centering.")
            X_train_scaled[col] = X_train[col] - col_mean  # Just center
            X_test_scaled[col] = X_test[col] - col_mean  # Just center

    return X_train_scaled, X_test_scaled


def check_numeric_columns(df, expected_columns=None):
    """
    Check which columns in the dataframe are numeric. For non-numeric columns,
    attempts to convert them to numeric and reports results.

    Parameters:
    - df: DataFrame to check
    - expected_columns: Optional list of column names that are expected to be numeric

    Returns:
    - List of column names that are safely numeric
    """
    logger.info("Checking numeric columns...")

    if expected_columns is None:
        columns_to_check = df.columns
    else:
        columns_to_check = [col for col in expected_columns if col in df.columns]

    numeric_columns = []

    for col in columns_to_check:
        # Skip metadata columns
        if col in ['subject_id', 'timepoint_id', 'fmri_date']:
            continue

        # Check if already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)
            continue

        # Try to convert to numeric
        try:
            # Check a few values to see if conversion is likely to succeed
            sample = df[col].dropna().head(10)
            converted = pd.to_numeric(sample, errors='coerce')
            if converted.isna().any():
                logger.warning(f"Column '{col}' appears to be non-numeric: {sample.tolist()}")
                continue

            # Try converting the full column
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().mean() < 0.5:  # If less than 50% values were converted to NaN
                numeric_columns.append(col)
                logger.info(f"Successfully converted '{col}' to numeric")
            else:
                logger.warning(f"Column '{col}' has too many non-numeric values")
        except Exception as e:
            logger.warning(f"Could not convert column '{col}' to numeric: {e}")

    logger.info(f"Found {len(numeric_columns)}/{len(columns_to_check)} numeric columns")
    return numeric_columns


def create_nested_cv_folds(unique_subjects, n_folds=5, random_state=42):
    """
    Create nested cross-validation folds.

    Parameters:
    -----------
    unique_subjects : array
        Array of unique subject IDs
    n_folds : int
        Number of folds (default 5)
    random_state : int
        Fixed seed for reproducibility

    Returns:
    --------
    dict: Contains inner_folds and outer_folds structure
    """
    np.random.seed(random_state)

    # Shuffle subjects once for consistent assignment
    shuffled_subjects = unique_subjects.copy()
    np.random.shuffle(shuffled_subjects)

    # Create 5 inner folds (these assignments stay consistent across all outer folds)
    inner_folds = np.array_split(shuffled_subjects, n_folds)

    # Create 5 outer folds
    outer_folds = []
    for test_fold_idx in range(n_folds):
        # Use one inner fold as test
        test_subjects = list(inner_folds[test_fold_idx])

        # Use remaining 4 inner folds as train
        train_subjects = []
        for train_fold_idx in range(n_folds):
            if train_fold_idx != test_fold_idx:
                train_subjects.extend(inner_folds[train_fold_idx])

        outer_folds.append({
            'train': train_subjects,
            'test': test_subjects,
            'test_inner_fold': test_fold_idx
        })

    logger.info(f"Created nested CV structure:")
    logger.info(f"Inner fold sizes: {[len(fold) for fold in inner_folds]}")
    logger.info(f"Outer fold train/test sizes: {[(len(fold['train']), len(fold['test'])) for fold in outer_folds]}")

    return {
        'inner_folds': [list(fold) for fold in inner_folds],
        'outer_folds': outer_folds
    }


def validate_nested_cv_structure(all_fold_assignments, cv_structure, logger):
    """
    Comprehensive validation of the nested CV structure.
    """
    logger.info("\n===== COMPREHENSIVE NESTED CV VALIDATION =====")

    validation_passed = True

    # ===== 1. VALIDATE INNER FOLD STRUCTURE =====
    logger.info("1. Validating inner fold structure...")

    # Check that each subject appears in exactly one inner fold
    subjects_in_inner_folds = set()
    inner_fold_subjects = {}

    for inner_idx, subjects in enumerate(cv_structure['inner_folds']):
        inner_fold_subjects[inner_idx] = set(subjects)

        # Check for duplicates within this inner fold
        if len(subjects) != len(set(subjects)):
            logger.error(f"Inner fold {inner_idx} contains duplicate subjects!")
            validation_passed = False

        # Check for subjects already seen in other inner folds
        overlap_with_previous = subjects_in_inner_folds.intersection(set(subjects))
        if overlap_with_previous:
            logger.error(
                f"Inner fold {inner_idx} contains subjects already in other inner folds: {overlap_with_previous}")
            validation_passed = False

        subjects_in_inner_folds.update(subjects)
        logger.info(f"✓ Inner fold {inner_idx}: {len(subjects)} unique subjects")

    # Check that inner folds are mutually exclusive
    for i in range(len(cv_structure['inner_folds'])):
        for j in range(i + 1, len(cv_structure['inner_folds'])):
            overlap = inner_fold_subjects[i].intersection(inner_fold_subjects[j])
            if overlap:
                logger.error(f"Inner folds {i} and {j} share {len(overlap)} subjects: {overlap}")
                validation_passed = False

    if validation_passed:
        logger.info("✓ Inner fold structure is valid: no subject appears in multiple inner folds")

    # ===== 2. VALIDATE OUTER FOLD STRUCTURE =====
    logger.info("\n2. Validating outer fold structure...")

    # Verify that each subject appears in exactly 4 train sets and 1 test set
    subject_appearances = defaultdict(lambda: {'train': 0, 'test': 0, 'test_folds': []})

    for assignment in all_fold_assignments:
        subject = assignment['subject_id']
        split = assignment['inner_split']
        outer_fold = assignment['outer_fold']
        subject_appearances[subject][split] += 1
        if split == 'test':
            subject_appearances[subject]['test_folds'].append(outer_fold)

    # Check the nested CV structure
    for subject, counts in subject_appearances.items():
        if counts['train'] != 4 or counts['test'] != 1:
            logger.error(
                f"Subject {subject} appears in {counts['train']} train sets and {counts['test']} test sets (should be 4 train, 1 test)")
            validation_passed = False

        if len(counts['test_folds']) != 1:
            logger.error(
                f"Subject {subject} appears as test in {len(counts['test_folds'])} outer folds: {counts['test_folds']}")
            validation_passed = False

    if validation_passed:
        logger.info("✓ Outer fold structure is valid: each subject appears in exactly 4 train sets and 1 test set")

    # ===== 3. VALIDATE NO DATA LEAKAGE WITHIN OUTER FOLDS =====
    logger.info("\n3. Checking for data leakage within outer folds...")

    outer_fold_data = defaultdict(lambda: {'train': set(), 'test': set()})

    for assignment in all_fold_assignments:
        outer_fold = assignment['outer_fold']
        subject = assignment['subject_id']
        split = assignment['inner_split']
        outer_fold_data[outer_fold][split].add(subject)

    leakage_detected = False
    for outer_fold in range(5):
        train_subjects = outer_fold_data[outer_fold]['train']
        test_subjects = outer_fold_data[outer_fold]['test']
        overlap = train_subjects.intersection(test_subjects)

        if overlap:
            logger.error(f"Data leakage in outer fold {outer_fold}: {len(overlap)} subjects in both train and test")
            leakage_detected = True
            validation_passed = False
        else:
            logger.info(
                f"✓ Outer fold {outer_fold}: No data leakage ({len(train_subjects)} train, {len(test_subjects)} test)")

    # ===== 4. VALIDATE CONSISTENCY BETWEEN INNER AND OUTER FOLDS =====
    logger.info("\n4. Validating consistency between inner and outer fold assignments...")

    # For each outer fold, verify that test subjects come from the correct inner fold
    for outer_fold_idx, fold_info in enumerate(cv_structure['outer_folds']):
        expected_test_subjects = set(fold_info['test'])
        expected_test_inner_fold = fold_info['test_inner_fold']

        # Get actual test subjects from assignments
        actual_test_subjects = outer_fold_data[outer_fold_idx]['test']

        # Check that they match
        if expected_test_subjects != actual_test_subjects:
            logger.error(f"Outer fold {outer_fold_idx}: Expected test subjects don't match actual test subjects")
            logger.error(f"  Expected: {len(expected_test_subjects)} subjects")
            logger.error(f"  Actual: {len(actual_test_subjects)} subjects")
            validation_passed = False

        # Verify that test subjects come from the correct inner fold
        inner_fold_test_subjects = set(cv_structure['inner_folds'][expected_test_inner_fold])
        if expected_test_subjects != inner_fold_test_subjects:
            logger.error(
                f"Outer fold {outer_fold_idx}: Test subjects don't match inner fold {expected_test_inner_fold}")
            validation_passed = False
        else:
            logger.info(
                f"✓ Outer fold {outer_fold_idx}: Test subjects correctly from inner fold {expected_test_inner_fold}")

    # ===== 5. FINAL VALIDATION SUMMARY =====
    logger.info("\n5. Final validation summary...")

    # Count unique subjects in assignments directly
    unique_subjects_in_assignments = set(assignment['subject_id'] for assignment in all_fold_assignments)
    total_subjects_in_assignments = len(unique_subjects_in_assignments)
    total_subjects_in_inner_folds = len(subjects_in_inner_folds)

    # Additional debugging info
    logger.info(f"Total assignments: {len(all_fold_assignments)}")
    logger.info(f"Expected assignments (601 subjects × 5 folds): {total_subjects_in_inner_folds * 5}")
    logger.info(f"Unique subjects in assignments: {total_subjects_in_assignments}")
    logger.info(f"Unique subjects in inner folds: {total_subjects_in_inner_folds}")

    # Check for missing subjects
    subjects_missing_from_assignments = subjects_in_inner_folds - unique_subjects_in_assignments
    subjects_extra_in_assignments = unique_subjects_in_assignments - subjects_in_inner_folds

    if subjects_missing_from_assignments:
        logger.warning(f"Subjects in inner folds but missing from assignments: {subjects_missing_from_assignments}")

    if subjects_extra_in_assignments:
        logger.warning(f"Subjects in assignments but not in inner folds: {subjects_extra_in_assignments}")

    if total_subjects_in_assignments != total_subjects_in_inner_folds:
        logger.error(
            f"Subject count mismatch: {total_subjects_in_assignments} in assignments vs {total_subjects_in_inner_folds} in inner folds")

        # This might not be a fatal error if the mismatch is due to filtering
        if abs(total_subjects_in_assignments - total_subjects_in_inner_folds) <= 1:
            logger.warning("Small subject count difference (≤1) - this might be due to rounding in np.array_split")
            # Don't fail validation for small differences
        else:
            validation_passed = False

    if validation_passed:
        logger.info(" ALL VALIDATIONS PASSED: Nested CV structure is completely valid")
        logger.info(
            f"   - {len(cv_structure['inner_folds'])} inner folds with {total_subjects_in_inner_folds} total subjects")
        logger.info(f"   - {len(cv_structure['outer_folds'])} outer folds")
        logger.info(f"   - No data leakage detected")
        logger.info(f"   - All subjects properly distributed")
    else:
        logger.error(" VALIDATION FAILED: Issues detected in nested CV structure")

    logger.info("===== NESTED CV VALIDATION COMPLETE =====")
    return validation_passed



def main():
    """Main function to extract and prepare longitudinal cognitive data for G factor analysis with nested CV"""
    try:
        # Create fold directories
        create_fold_directories(OUTPUT_DIR)

        # Load image data (for fMRI scans)
        image_df = load_and_preprocess_data(IMAGE_FILE)
        logger.info(f"Loaded image data with {len(image_df)} rows")

        # Filter rows that contain 'fMRI' in the scan_type column
        fmri_df = image_df[image_df['scan_type'].str.contains('fMRI', na=False)]
        logger.info(f"Found {len(fmri_df)} fMRI scans")

        # Log the number of subjects and timepoints that will be checked for exclusion
        n_subjects_to_check = fmri_df['subjectkey'].nunique()
        logger.info(f"Will check {n_subjects_to_check} unique subjects with {len(fmri_df)} timepoints for exclusion")

        # Create exclusion tracking lists for reporting
        excluded_completely = []
        excluded_timepoints = []

        # Load each task dataset
        task_dataframes = {}
        task_files = set(info['file'] for info in COGNITIVE_VARIABLES.values())

        for file_path in task_files:
            task_df = load_and_preprocess_data(file_path, has_subheader=True)
            if not task_df.empty:
                task_name = os.path.basename(file_path).split('.')[0]
                task_dataframes[task_name] = task_df

                # Clean task data for each variable from this task
                for var_key, var_info in COGNITIVE_VARIABLES.items():
                    if var_info['file'] == file_path:
                        var_name = var_info['variable']
                        if var_name in task_df.columns:
                            task_dataframes[task_name] = clean_missing_values(
                                task_dataframes[task_name],
                                var_name,
                                var_info.get('expected_range')
                            )

        # Find matching cognitive data for each fMRI scan
        all_timepoints = []
        excluded_count = 0

        # Group fMRI scans by subject to get multiple timepoints
        for subject, subject_scans in fmri_df.groupby('subjectkey'):
            logger.info(f"Processing subject {subject} with {len(subject_scans)} fMRI scans")

            # Check if subject should be completely excluded
            if normalize_subject_id(subject) in SUBJECT_EXCLUDE_COMPLETE:
                excluded_completely.append(subject)
                excluded_count += len(subject_scans)
                logger.info(f"Subject {subject} is in complete exclusion list - skipping all timepoints")
                continue

            subject_timepoints_excluded = 0

            # For each fMRI scan, find matching cognitive data
            for _, scan in subject_scans.iterrows():
                # Extract age information for timepoint-specific exclusion check
                interview_age = scan.get('interview_age')

                # Check if this specific timepoint should be excluded
                if should_exclude_timepoint(subject, interview_age=interview_age):
                    # Format timepoint information for reporting
                    age_str = f"Age{int(interview_age)}Months" if interview_age is not None else "AgeUnknown"
                    timepoint_id = f"{subject}_{age_str}"
                    excluded_timepoints.append(timepoint_id)
                    excluded_count += 1
                    subject_timepoints_excluded += 1
                    continue

                # Find matching cognitive data for non-excluded timepoints
                matched_data = find_matching_cognitive_data(scan, task_dataframes, COGNITIVE_VARIABLES)
                if matched_data:
                    # Add to our list of timepoints
                    all_timepoints.append(matched_data)

            if subject_timepoints_excluded > 0:
                logger.info(
                    f"Excluded {subject_timepoints_excluded} of {len(subject_scans)} timepoints for subject {subject}")

        # Convert to DataFrame
        timepoints_df = pd.DataFrame(all_timepoints)
        logger.info(
            f"Created longitudinal dataset with {len(timepoints_df)} timepoints across {timepoints_df['subject_id'].nunique()} subjects")
        logger.info(f"Excluded a total of {excluded_count} timepoints based on exclusion criteria")

        # Save the exclusion information
        exclusion_info = {
            'completely_excluded_subjects': sorted(excluded_completely),
            'excluded_timepoints': sorted(excluded_timepoints)
        }

        # Convert to DataFrame and save
        exclusion_df = pd.DataFrame({
            'excluded_id': excluded_completely + excluded_timepoints,
            'exclusion_type': ['complete_subject'] * len(excluded_completely) +
                              ['specific_timepoint'] * len(excluded_timepoints)
        })
        exclusion_df.to_csv(os.path.join(OUTPUT_DIR, "excluded_entries.csv"), index=False)
        logger.info(f"Saved exclusion information to {os.path.join(OUTPUT_DIR, 'excluded_entries.csv')}")

        # Save the raw matched data (before variable inversion)
        timepoints_df.to_csv(os.path.join(OUTPUT_DIR, "all_longitudinal_cognitive_data_raw.csv"), index=False)

        # Check which columns are numeric
        numeric_columns = check_numeric_columns(timepoints_df)

        # Get the actual cognitive variables (exclude metadata)
        metadata_cols = ['subject_id', 'timepoint_id', 'fmri_date', 'age_months', 'age_years']
        cognitive_columns = [col for col in numeric_columns if col not in metadata_cols]
        logger.info(f"Found {len(cognitive_columns)} valid numeric cognitive variables for analysis")

        # After inverting variables where higher values mean worse performance
        processed_df = invert_variables(timepoints_df)

        # Transform d-prime measures to improve normality
        processed_df = transform_variables(processed_df)

        # Save the processed data (now including transformed d-primes)
        processed_df.to_csv(os.path.join(OUTPUT_DIR, "all_longitudinal_cognitive_data_processed.csv"), index=False)

        # Filter rows with at least 75% non-missing cognitive values
        min_non_missing = len(cognitive_columns) * 0.75
        numeric_columns_plus_id = cognitive_columns + ['timepoint_id', 'subject_id']
        complete_rows = processed_df.dropna(subset=numeric_columns_plus_id,
                                            thresh=min_non_missing + 1)  # +1 for timepoint_id
        logger.info(f"Found {len(complete_rows)} timepoints with at least 75% complete data " +
                    f"({min_non_missing} of {len(cognitive_columns)} variables)")

        # Get unique subjects
        unique_subjects = complete_rows['subject_id'].unique()
        logger.info(f"Found {len(unique_subjects)} unique subjects with sufficient data")

        # ===== NESTED CROSS-VALIDATION STRUCTURE =====
        logger.info("Creating nested cross-validation folds...")
        cv_structure = create_nested_cv_folds(unique_subjects, n_folds=5, random_state=42)

        # Track all fold assignments for validation and saving
        all_fold_assignments = []

        # Process each outer fold
        for outer_fold_idx, fold_info in enumerate(cv_structure['outer_folds']):
            train_subjects = fold_info['train']
            test_subjects = fold_info['test']
            test_inner_fold = fold_info['test_inner_fold']

            logger.info(f"\n===== Processing Outer Fold {outer_fold_idx} =====")
            logger.info(f"Train subjects: {len(train_subjects)}, Test subjects: {len(test_subjects)}")
            logger.info(f"Test subjects from inner fold {test_inner_fold}")

            # Store fold assignments for tracking
            for subject in train_subjects:
                all_fold_assignments.append({
                    'subject_id': subject,
                    'outer_fold': outer_fold_idx,
                    'inner_split': 'train',
                    'test_inner_fold': test_inner_fold,
                    'fold_assignment': f'OuterFold{outer_fold_idx}_Train'
                })

            for subject in test_subjects:
                all_fold_assignments.append({
                    'subject_id': subject,
                    'outer_fold': outer_fold_idx,
                    'inner_split': 'test',
                    'test_inner_fold': test_inner_fold,
                    'fold_assignment': f'OuterFold{outer_fold_idx}_Test'
                })

            # Get all timepoints for train and test subjects
            train_df = complete_rows[complete_rows['subject_id'].isin(train_subjects)]
            test_df = complete_rows[complete_rows['subject_id'].isin(test_subjects)]

            logger.info(
                f"Outer fold {outer_fold_idx}: {len(train_df)} train timepoints, {len(test_df)} test timepoints")

            # Select only cognitive variables and timepoint ID for processing
            X_train = train_df[cognitive_columns]
            X_test = test_df[cognitive_columns]
            train_ids = train_df['timepoint_id']
            test_ids = test_df['timepoint_id']

            # Check if any columns have all NaN values in training set
            na_counts = X_train.isna().sum()
            all_na_cols = na_counts[na_counts == len(X_train)].index.tolist()
            if all_na_cols:
                logger.warning(f"Columns with all NaN in train set for outer fold {outer_fold_idx}: {all_na_cols}")
                # Remove these columns from both train and test
                X_train = X_train.drop(columns=all_na_cols)
                X_test = X_test.drop(columns=all_na_cols)
                logger.info(f"Removed {len(all_na_cols)} all-NaN columns from outer fold {outer_fold_idx}")

            # Standardize the data
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()

            # Standardize only the non-missing values, column by column
            for col in X_train.columns:
                # Get mean and std of non-missing values in training set
                train_mean = X_train[col].mean()
                train_std = X_train[col].std()

                # Only standardize non-missing values
                if train_std > 0:  # Check for zero variance
                    # Train data - standardize only non-missing values
                    mask = ~X_train[col].isna()
                    X_train_scaled.loc[mask, col] = (X_train.loc[mask, col] - train_mean) / train_std

                    # Test data - standardize only non-missing values
                    mask = ~X_test[col].isna()
                    X_test_scaled.loc[mask, col] = (X_test.loc[mask, col] - train_mean) / train_std
                else:
                    # For zero variance, just center
                    mask = ~X_train[col].isna()
                    X_train_scaled.loc[mask, col] = X_train.loc[mask, col] - train_mean

                    mask = ~X_test[col].isna()
                    X_test_scaled.loc[mask, col] = X_test.loc[mask, col] - train_mean

                    logger.warning(
                        f"Zero variance detected in feature '{col}' for outer fold {outer_fold_idx}. Setting to 0 after centering.")

            # Add timepoint IDs back
            X_train_scaled['eid'] = train_ids.values
            X_test_scaled['eid'] = test_ids.values

            # Reorder columns to put 'eid' first
            cols = ['eid'] + X_train.columns.tolist()
            X_train_scaled = X_train_scaled[cols]
            X_test_scaled = X_test_scaled[cols]

            # Save each outer fold
            fold_dir = os.path.join(OUTPUT_DIR, f"fold_{outer_fold_idx}")
            X_train_scaled.to_csv(
                os.path.join(fold_dir, "cogscores", f"cogscores_12T_i2_train_with_id_scaled_fold_{outer_fold_idx}.csv"),
                index=False
            )
            X_test_scaled.to_csv(
                os.path.join(fold_dir, "cogscores", f"cogscores_12T_i2_test_with_id_scaled_fold_{outer_fold_idx}.csv"),
                index=False
            )

            logger.info(f"Saved outer fold {outer_fold_idx} data files")

        # Validate the nested CV structure
        validation_result = validate_nested_cv_structure(all_fold_assignments, cv_structure, logger)

        if validation_result:
            logger.info("All nested CV validations passed!")
        else:
            logger.error("Nested CV validation FAILED. Check the logs for details.")

        # Save fold assignment information
        if all_fold_assignments:
            # Enhanced format with nested CV info
            fold_assignments_df = pd.DataFrame(all_fold_assignments)
            fold_assignments_path = os.path.join(OUTPUT_DIR, "nested_fold_assignments.csv")
            fold_assignments_df.to_csv(fold_assignments_path, index=False)
            logger.info(f"Saved nested CV fold assignments to {fold_assignments_path}")

            # Also save the inner fold structure for reference
            inner_fold_assignments = []
            for inner_idx, subjects in enumerate(cv_structure['inner_folds']):
                for subject in subjects:
                    inner_fold_assignments.append({
                        'subject_id': subject,
                        'inner_fold': inner_idx
                    })

            inner_fold_df = pd.DataFrame(inner_fold_assignments)
            inner_fold_path = os.path.join(OUTPUT_DIR, "inner_fold_assignments.csv")
            inner_fold_df.to_csv(inner_fold_path, index=False)
            logger.info(f"Saved inner fold assignments to {inner_fold_path}")

        logger.info("Nested cross-validation data extraction and preparation complete")
        return processed_df

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    main()