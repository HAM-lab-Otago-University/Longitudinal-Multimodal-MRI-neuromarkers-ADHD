#!/usr/bin/env python3
"""
Brain Correlation Visualization with Stacked Predictions
=======================================================

Correlates raw brain imaging data (ALFF, ReHo, FC) with predicted g-factor
from stacked model and creates brain visualizations.

- ALFF/ReHo: Uses HCP MMP parcellation with surfplot brain visualizations
- FC: Uses Cole network organization with correlation heatmaps

Adapted from adaptations upon adaptions upon adaptions from Farzane, Ksenia & Alina scripts,
plus my own peculiarities
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base paths
BASE_PATH = "/media/hcs-sci-psy-narun/Jack/First_level_model_aligned"
PRED_PATH = "/media/hcs-sci-psy-narun/Jack/First_level_model_aligned/Stacked_model_ridge_v4_EXTREME_conserve_a2000_keeper"
OUTPUT_DIR = "/media/hcs-sci-psy-narun/Jack/HCP_Brain_viz_cor_func_NEW_v2"

# Data paths
ALFF_BASE = f"{BASE_PATH}/functional/alff"
REHO_BASE = f"{BASE_PATH}/functional/reho"
FC_BASE = f"{BASE_PATH}/functional/functional_connectivity"

# Cole network mapping file
COLE_MAPPING_FILE = "/media/hcs-sci-psy-narun/Jack/groi_colenetwork_mapping.csv"

# Cole-Anticevic Network mapping (12 networks + subcortical)
NETWORK_NAMES = {
    1: "Visual1", 2: "Visual2", 3: "Somatomotor", 4: "Cingulo-Opercular",
    5: "Dorsal_Attention", 6: "Language", 7: "Frontoparietal", 8: "Auditory",
    9: "Default", 10: "Posterior_Multimodal", 11: "Ventral_Multimodal",
    12: "Orbito-Affective", 13: "Subcortical"
}

# Network colors for visualization
NETWORK_COLORS = {
    1: '#781286',  # Visual1 - Purple
    2: '#4682B4',  # Visual2 - Steel Blue
    3: '#00CED1',  # Somatomotor - Dark Turquoise
    4: '#228B22',  # Cingulo-Opercular - Forest Green
    5: '#00FF00',  # Dorsal_Attention - Lime
    6: '#FFD700',  # Language - Gold
    7: '#FF8C00',  # Frontoparietal - Dark Orange
    8: '#FF1493',  # Auditory - Deep Pink
    9: '#FF0000',  # Default - Red
    10: '#8B0000',  # Posterior_Multimodal - Dark Red
    11: '#800080',  # Ventral_Multimodal - Purple
    12: '#000000',  # Orbito-Affective - Black
    13: '#CCCCCC'  # Subcortical - Grey
}

# FC region names (377 total) - from v8
FC_REGION_NAMES = [
    'R_V1', 'R_ProS', 'R_DVT', 'L_V1', 'L_ProS', 'L_DVT', 'R_MST', 'R_V6', 'R_V2', 'R_V3', 'R_V4', 'R_V8', 'R_V3A',
    'R_V7', 'R_IPS1', 'R_FFC', 'R_V3B', 'R_LO1', 'R_LO2', 'R_PIT', 'R_MT', 'R_LIPv', 'R_VIP', 'R_PH', 'R_V6A', 'R_VMV1',
    'R_VMV3', 'R_V4t', 'R_FST', 'R_V3CD', 'R_LO3', 'R_VMV2', 'R_VVC', 'L_MST', 'L_V6', 'L_V2', 'L_V3', 'L_V4', 'L_V8',
    'L_V3A', 'L_V7', 'L_IPS1', 'L_FFC', 'L_V3B', 'L_LO1', 'L_LO2', 'L_PIT', 'L_MT', 'L_LIPv', 'L_VIP', 'L_PH', 'L_V6A',
    'L_VMV1', 'L_VMV3', 'L_V4t', 'L_FST', 'L_V3CD', 'L_LO3', 'L_VMV2', 'L_VVC', 'R_4', 'R_3b', 'R_5m', 'R_5L', 'R_24dd',
    'R_24dv', 'R_7AL', 'R_7PC', 'R_1', 'R_2', 'R_3a', 'R_6d', 'R_6mp', 'R_6v', 'R_OP4', 'R_OP1', 'R_OP2-3', 'R_RI',
    'R_FOP2', 'R_Ig', 'L_4', 'L_3b', 'L_5m', 'L_5L', 'L_24dd', 'L_24dv', 'L_7AL', 'L_7PC', 'L_1', 'L_2', 'L_3a', 'L_6d',
    'L_6mp', 'L_6v', 'L_OP4', 'L_OP1', 'L_OP2-3', 'L_FOP2', 'L_Ig', 'R_FEF', 'R_PEF', 'R_PSL', 'R_5mv', 'R_23c',
    'R_SCEF', 'R_6ma', 'R_7Am', 'R_p24pr', 'R_a24pr', 'R_p32pr', 'R_6r', 'R_IFSa', 'R_46', 'R_9-46d', 'R_43', 'R_PFcm',
    'R_PoI2', 'R_FOP4', 'R_MI', 'R_FOP1', 'R_FOP3', 'R_PFop', 'R_PF', 'R_PoI1', 'R_FOP5', 'R_PI', 'R_a32pr', 'R_p24',
    'L_FEF', 'L_5mv', 'L_23c', 'L_SCEF', 'L_6ma', 'L_7Am', 'L_p24pr', 'L_33pr', 'L_a24pr', 'L_p32pr', 'L_6r', 'L_46',
    'L_9-46d', 'L_43', 'L_PFcm', 'L_PoI2', 'L_FOP4', 'L_MI', 'L_FOP1', 'L_FOP3', 'L_PFop', 'L_PF', 'L_PoI1', 'L_FOP5',
    'L_PI', 'L_a32pr', 'L_p24', 'R_7PL', 'R_MIP', 'R_LIPd', 'R_6a', 'R_PFt', 'R_AIP', 'R_PHA3', 'R_TE2p', 'R_PHT',
    'R_PGp', 'R_IP0', 'L_PEF', 'L_7PL', 'L_MIP', 'L_LIPd', 'L_6a', 'L_PFt', 'L_AIP', 'L_PHA3', 'L_TE2p', 'L_PHT',
    'L_PGp', 'L_IP0', 'R_55b', 'R_SFL', 'R_45', 'R_IFJa', 'R_STGa', 'R_A5', 'R_STSdp', 'R_TPOJ1', 'R_TGv', 'L_55b',
    'L_PSL', 'L_SFL', 'L_STV', 'L_44', 'L_45', 'L_IFJa', 'L_IFSp', 'L_STGa', 'L_A5', 'L_STSda', 'L_STSdp', 'L_TPOJ1',
    'L_TGv', 'R_RSC', 'R_POS2', 'R_7Pm', 'R_33pr', 'R_d32', 'R_8BM', 'R_8C', 'R_44', 'R_a47r', 'R_IFJp', 'R_IFSp',
    'R_p9-46v', 'R_a9-46v', 'R_a10p', 'R_11l', 'R_13l', 'R_OFC', 'R_i6-8', 'R_s6-8', 'R_AVI', 'R_TE1p', 'R_IP2',
    'R_IP1', 'R_PFm', 'R_31a', 'R_p10p', 'R_p47r', 'R_TE1m', 'L_RSC', 'L_POS2', 'L_7Pm', 'L_8BM', 'L_8C', 'L_a47r',
    'L_IFJp', 'L_IFSa', 'L_p9-46v', 'L_a9-46v', 'L_a10p', 'L_11l', 'L_13l', 'L_i6-8', 'L_s6-8', 'L_AVI', 'L_TE1p',
    'L_IP2', 'L_IP1', 'L_PFm', 'L_p10p', 'L_p47r', 'R_A1', 'R_52', 'R_TA2', 'R_PBelt', 'R_MBelt', 'R_LBelt', 'R_A4',
    'L_A1', 'L_52', 'L_RI', 'L_TA2', 'L_PBelt', 'L_MBelt', 'L_LBelt', 'L_A4', 'R_7m', 'R_POS1', 'R_23d', 'R_v23ab',
    'R_d23ab', 'R_31pv', 'R_a24', 'R_p32', 'R_10r', 'R_47m', 'R_8Av', 'R_8Ad', 'R_9m', 'R_8BL', 'R_9p', 'R_10d',
    'R_47l', 'R_9a', 'R_10v', 'R_10pp', 'R_47s', 'R_EC', 'R_PreS', 'R_PHA1', 'R_STSda', 'R_STSvp', 'R_TGd', 'R_TE1a',
    'R_TE2a', 'R_PGi', 'R_PGs', 'R_PHA2', 'R_31pd', 'R_25', 'R_s32', 'R_STSva', 'L_7m', 'L_POS1', 'L_23d', 'L_v23ab',
    'L_d23ab', 'L_31pv', 'L_a24', 'L_d32', 'L_p32', 'L_10r', 'L_47m', 'L_8Av', 'L_8Ad', 'L_9m', 'L_8BL', 'L_9p',
    'L_10d', 'L_47l', 'L_9a', 'L_10v', 'L_10pp', 'L_OFC', 'L_47s', 'L_EC', 'L_PreS', 'L_PHA1', 'L_STSvp', 'L_TGd',
    'L_TE1a', 'L_TE2a', 'L_PGi', 'L_PGs', 'L_PHA2', 'L_31pd', 'L_31a', 'L_25', 'L_s32', 'L_STSva', 'L_TE1m', 'R_PCV',
    'R_STV', 'R_TPOJ2', 'R_TPOJ3', 'L_PCV', 'L_TPOJ2', 'L_TPOJ3', 'R_PeEc', 'R_TF', 'L_PeEc', 'L_TF', 'R_Pir', 'R_AAIC',
    'R_pOFC', 'L_Pir', 'L_AAIC', 'L_pOFC', 'CEREBELLUM_LEFT', 'THALAMUS_LEFT', 'CAUDATE_LEFT', 'PUTAMEN_LEFT',
    'PALLIDUM_LEFT', 'BRAIN_STEM', 'HIPPOCAMPUS_LEFT', 'AMYGDALA_LEFT', 'ACCUMBENS_LEFT', 'DIENCEPHALON_VENTRAL_LEFT',
    'CEREBELLUM_RIGHT', 'THALAMUS_RIGHT', 'CAUDATE_RIGHT', 'PUTAMEN_RIGHT', 'PALLIDUM_RIGHT', 'HIPPOCAMPUS_RIGHT',
    'AMYGDALA_RIGHT', 'ACCUMBENS_RIGHT', 'DIENCEPHALON_VENTRAL_RIGHT'
]

# ALFF/ReHo region names (379 total) - from v13 with L_H/R_H included
ALFF_REHO_REGION_NAMES = [
    'ACCUMBENS_LEFT', 'ACCUMBENS_RIGHT', 'AMYGDALA_LEFT', 'AMYGDALA_RIGHT',
    'BRAIN_STEM', 'CAUDATE_LEFT', 'CAUDATE_RIGHT', 'CEREBELLUM_LEFT',
    'CEREBELLUM_RIGHT', 'DIENCEPHALON_VENTRAL_LEFT', 'DIENCEPHALON_VENTRAL_RIGHT',
    'HIPPOCAMPUS_LEFT', 'HIPPOCAMPUS_RIGHT', 'Left_1', 'Left_10d', 'Left_10pp',
    'Left_10r', 'Left_10v', 'Left_11l', 'Left_13l', 'Left_2', 'Left_23c',
    'Left_23d', 'Left_24dd', 'Left_24dv', 'Left_25', 'Left_31a', 'Left_31pd',
    'Left_31pv', 'Left_33pr', 'Left_3a', 'Left_3b', 'Left_4', 'Left_43',
    'Left_44', 'Left_45', 'Left_46', 'Left_47l', 'Left_47m', 'Left_47s',
    'Left_52', 'Left_55b', 'Left_5L', 'Left_5m', 'Left_5mv', 'Left_6a',
    'Left_6d', 'Left_6ma', 'Left_6mp', 'Left_6r', 'Left_6v', 'Left_7AL',
    'Left_7Am', 'Left_7PC', 'Left_7PL', 'Left_7Pm', 'Left_7m', 'Left_8Ad',
    'Left_8Av', 'Left_8BL', 'Left_8BM', 'Left_8C', 'Left_9-46d', 'Left_9a',
    'Left_9m', 'Left_9p', 'Left_A1', 'Left_A4', 'Left_A5', 'Left_AAIC',
    'Left_AIP', 'Left_AVI', 'Left_DVT', 'Left_EC', 'Left_FEF', 'Left_FFC',
    'Left_FOP1', 'Left_FOP2', 'Left_FOP3', 'Left_FOP4', 'Left_FOP5', 'Left_FST',
    'Left_H',  # CRITICAL: L_H region for HCP MMP
    'Left_IFJa', 'Left_IFJp', 'Left_IFSa', 'Left_IFSp', 'Left_IP0', 'Left_IP1',
    'Left_IP2', 'Left_IPS1', 'Left_Ig', 'Left_LBelt', 'Left_LIPd', 'Left_LIPv',
    'Left_LO1', 'Left_LO2', 'Left_LO3', 'Left_MBelt', 'Left_MI', 'Left_MIP',
    'Left_MST', 'Left_MT', 'Left_OFC', 'Left_OP1', 'Left_OP2-3', 'Left_OP4',
    'Left_PBelt', 'Left_PCV', 'Left_PEF', 'Left_PF', 'Left_PFcm', 'Left_PFm',
    'Left_PFop', 'Left_PFt', 'Left_PGi', 'Left_PGp', 'Left_PGs', 'Left_PH',
    'Left_PHA1', 'Left_PHA2', 'Left_PHA3', 'Left_PHT', 'Left_PI', 'Left_PIT',
    'Left_POS1', 'Left_POS2', 'Left_PSL', 'Left_PeEc', 'Left_Pir', 'Left_PoI1',
    'Left_PoI2', 'Left_PreS', 'Left_ProS', 'Left_RI', 'Left_RSC', 'Left_SCEF',
    'Left_SFL', 'Left_STGa', 'Left_STSda', 'Left_STSdp', 'Left_STSva', 'Left_STSvp',
    'Left_STV', 'Left_TA2', 'Left_TE1a', 'Left_TE1m', 'Left_TE1p', 'Left_TE2a',
    'Left_TE2p', 'Left_TF', 'Left_TGd', 'Left_TGv', 'Left_TPOJ1', 'Left_TPOJ2',
    'Left_TPOJ3', 'Left_V1', 'Left_V2', 'Left_V3', 'Left_V3A', 'Left_V3B',
    'Left_V3CD', 'Left_V4', 'Left_V4t', 'Left_V6', 'Left_V6A', 'Left_V7',
    'Left_V8', 'Left_VIP', 'Left_VMV1', 'Left_VMV2', 'Left_VMV3', 'Left_VVC',
    'Left_a10p', 'Left_a24', 'Left_a24pr', 'Left_a32pr', 'Left_a47r', 'Left_a9-46v',
    'Left_d23ab', 'Left_d32', 'Left_i6-8', 'Left_p10p', 'Left_p24', 'Left_p24pr',
    'Left_p32', 'Left_p32pr', 'Left_p47r', 'Left_p9-46v', 'Left_pOFC', 'Left_s32',
    'Left_s6-8', 'Left_v23ab', 'PALLIDUM_LEFT', 'PALLIDUM_RIGHT', 'PUTAMEN_LEFT',
    'PUTAMEN_RIGHT', 'Right_1', 'Right_10d', 'Right_10pp', 'Right_10r', 'Right_10v',
    'Right_11l', 'Right_13l', 'Right_2', 'Right_23c', 'Right_23d', 'Right_24dd',
    'Right_24dv', 'Right_25', 'Right_31a', 'Right_31pd', 'Right_31pv', 'Right_33pr',
    'Right_3a', 'Right_3b', 'Right_4', 'Right_43', 'Right_44', 'Right_45',
    'Right_46', 'Right_47l', 'Right_47m', 'Right_47s', 'Right_52', 'Right_55b',
    'Right_5L', 'Right_5m', 'Right_5mv', 'Right_6a', 'Right_6d', 'Right_6ma',
    'Right_6mp', 'Right_6r', 'Right_6v', 'Right_7AL', 'Right_7Am', 'Right_7PC',
    'Right_7PL', 'Right_7Pm', 'Right_7m', 'Right_8Ad', 'Right_8Av', 'Right_8BL',
    'Right_8BM', 'Right_8C', 'Right_9-46d', 'Right_9a', 'Right_9m', 'Right_9p',
    'Right_A1', 'Right_A4', 'Right_A5', 'Right_AAIC', 'Right_AIP', 'Right_AVI',
    'Right_DVT', 'Right_EC', 'Right_FEF', 'Right_FFC', 'Right_FOP1', 'Right_FOP2',
    'Right_FOP3', 'Right_FOP4', 'Right_FOP5', 'Right_FST',
    'Right_H',  # CRITICAL: R_H region for HCP MMP
    'Right_IFJa', 'Right_IFJp',
    'Right_IFSa', 'Right_IFSp', 'Right_IP0', 'Right_IP1', 'Right_IP2', 'Right_IPS1',
    'Right_Ig', 'Right_LBelt', 'Right_LIPd', 'Right_LIPv', 'Right_LO1', 'Right_LO2',
    'Right_LO3', 'Right_MBelt', 'Right_MI', 'Right_MIP', 'Right_MST', 'Right_MT',
    'Right_OFC', 'Right_OP1', 'Right_OP2-3', 'Right_OP4', 'Right_PBelt', 'Right_PCV',
    'Right_PEF', 'Right_PF', 'Right_PFcm', 'Right_PFm', 'Right_PFop', 'Right_PFt',
    'Right_PGi', 'Right_PGp', 'Right_PGs', 'Right_PH', 'Right_PHA1', 'Right_PHA2',
    'Right_PHA3', 'Right_PHT', 'Right_PI', 'Right_PIT', 'Right_POS1', 'Right_POS2',
    'Right_PSL', 'Right_PeEc', 'Right_Pir', 'Right_PoI1', 'Right_PoI2', 'Right_PreS',
    'Right_ProS', 'Right_RI', 'Right_RSC', 'Right_SCEF', 'Right_SFL', 'Right_STGa',
    'Right_STSda', 'Right_STSdp', 'Right_STSva', 'Right_STSvp', 'Right_STV', 'Right_TA2',
    'Right_TE1a', 'Right_TE1m', 'Right_TE1p', 'Right_TE2a', 'Right_TE2p', 'Right_TF',
    'Right_TGd', 'Right_TGv', 'Right_TPOJ1', 'Right_TPOJ2', 'Right_TPOJ3', 'Right_V1',
    'Right_V2', 'Right_V3', 'Right_V3A', 'Right_V3B', 'Right_V3CD', 'Right_V4',
    'Right_V4t', 'Right_V6', 'Right_V6A', 'Right_V7', 'Right_V8', 'Right_VIP',
    'Right_VMV1', 'Right_VMV2', 'Right_VMV3', 'Right_VVC', 'Right_a10p', 'Right_a24',
    'Right_a24pr', 'Right_a32pr', 'Right_a47r', 'Right_a9-46v', 'Right_d23ab',
    'Right_d32', 'Right_i6-8', 'Right_p10p', 'Right_p24', 'Right_p24pr', 'Right_p32',
    'Right_p32pr', 'Right_p47r', 'Right_p9-46v', 'Right_pOFC', 'Right_s32', 'Right_s6-8',
    'Right_v23ab', 'THALAMUS_LEFT', 'THALAMUS_RIGHT'
]

print(f"Verification: ALFF/ReHo regions: {len(ALFF_REHO_REGION_NAMES)} (should be 379)")
print(f"Verification: FC regions: {len(FC_REGION_NAMES)} (should be 377)")


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_stacked_predictions():
    """Load stacked ridge predictions"""
    pred_file = f"{PRED_PATH}/stacked_ridge_predictions.csv"

    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")

    df = pd.read_csv(pred_file)
    test_df = df[df['split'] == 'test'].copy()

    print(f"Loaded {len(test_df)} test subjects from stacked predictions")
    print(f"Predicted g range: {test_df['stacked_pred'].min():.3f} to {test_df['stacked_pred'].max():.3f}")

    return test_df


def load_all_alff_reho_data(modality):
    """Load ALFF or ReHo data across all folds"""
    all_subjects = []
    all_brain_data = []

    if modality == 'alff':
        base_path = ALFF_BASE
    elif modality == 'reho':
        base_path = REHO_BASE
    else:
        raise ValueError("Modality must be 'alff' or 'reho'")

    print(f"Loading {modality.upper()} data...")
    for fold in range(5):
        data_path = f"{base_path}/fold_{fold}/test.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            subjects = df['eid'].values
            brain_data = df.drop(['eid', 'subject_id'], axis=1).values
            all_subjects.extend(subjects)
            all_brain_data.extend(brain_data)
            print(f"  Loaded {len(subjects)} subjects from fold {fold}")
        else:
            print(f"  Warning: File not found: {data_path}")

    all_brain_data = np.array(all_brain_data)
    all_brain_data = clean_brain_data(all_brain_data)
    return np.array(all_subjects), all_brain_data


def load_all_fc_data():
    """Load functional connectivity data across all folds"""
    all_subjects = []
    all_brain_data = []

    print("Loading FC data...")
    for fold in range(5):
        test_list_path = f"{FC_BASE}/fold_{fold}/test.csv"
        if not os.path.exists(test_list_path):
            print(f"  Warning: Test list not found: {test_list_path}")
            continue

        test_subjects = pd.read_csv(test_list_path)['eid'].values
        test_dir = f"{FC_BASE}/fold_{fold}/test"

        fold_subjects = []
        fold_brain_data = []

        for subject in test_subjects:
            fc_file = f"{test_dir}/{subject}_connectivity.csv"
            if os.path.exists(fc_file):
                fc_matrix = pd.read_csv(fc_file, index_col=0).values
                fold_subjects.append(subject)
                fold_brain_data.append(fc_matrix)

        all_subjects.extend(fold_subjects)
        all_brain_data.extend(fold_brain_data)
        print(f"  Loaded {len(fold_subjects)} subjects from fold {fold}")

    return np.array(all_subjects), all_brain_data


def clean_brain_data(brain_data):
    """Clean brain data by handling inf/NaN values"""
    print(f"  Cleaning data - original shape: {brain_data.shape}")

    inf_mask = np.isinf(brain_data)
    nan_mask = np.isnan(brain_data)

    if np.any(inf_mask):
        print(f"  Found {np.sum(inf_mask)} infinite values, replacing with NaN")
        brain_data[inf_mask] = np.nan

    if np.any(nan_mask):
        print(f"  Found {np.sum(nan_mask)} NaN values, replacing with column means")
        col_means = np.nanmean(brain_data, axis=0)
        for col in range(brain_data.shape[1]):
            col_nan_mask = np.isnan(brain_data[:, col])
            if np.any(col_nan_mask):
                brain_data[col_nan_mask, col] = col_means[col]

    # Final cleanup
    brain_data[np.isnan(brain_data)] = 0
    brain_data[np.isinf(brain_data)] = 0

    print(f"  Cleaned data shape: {brain_data.shape}")
    return brain_data


# =============================================================================
# CORRELATION ANALYSIS FUNCTIONS
# =============================================================================

def correlate_brain_with_predictions(subjects, brain_data, predictions, modality):
    """Correlate brain data with predicted g-factor"""
    print(f"\nCorrelating {modality.upper()} with predicted g...")

    # Match subjects between brain data and predictions
    pred_dict = dict(zip(predictions['subject_id'], predictions['stacked_pred']))

    matched_brain_data = []
    matched_pred = []
    matched_subjects = []

    for i, subject in enumerate(subjects):
        if subject in pred_dict:
            matched_brain_data.append(brain_data[i])
            matched_pred.append(pred_dict[subject])
            matched_subjects.append(subject)

    matched_brain_data = np.array(matched_brain_data)
    matched_pred = np.array(matched_pred)

    print(f"  Matched {len(matched_subjects)} subjects")

    if modality in ['alff', 'reho']:
        # Regional correlations for ALFF/ReHo
        n_regions = matched_brain_data.shape[1]
        correlations = []
        p_values = []

        for region in range(n_regions):
            region_data = matched_brain_data[:, region]
            r, p = pearsonr(region_data, matched_pred)
            correlations.append(r)
            p_values.append(p)

        return np.array(correlations), np.array(p_values), matched_subjects

    elif modality == 'functional_connectivity':
        # Connection-wise correlations for FC
        n_subjects = len(matched_brain_data)
        n_regions = matched_brain_data[0].shape[0]

        # Get upper triangle indices
        upper_tri_indices = np.triu_indices(n_regions, k=1)
        n_connections = len(upper_tri_indices[0])

        correlations = []
        p_values = []

        for conn_idx in range(n_connections):
            i, j = upper_tri_indices[0][conn_idx], upper_tri_indices[1][conn_idx]

            # Extract connection values across subjects
            connection_values = [fc_matrix[i, j] for fc_matrix in matched_brain_data]

            r, p = pearsonr(connection_values, matched_pred)
            correlations.append(r)
            p_values.append(p)

        return np.array(correlations), np.array(p_values), matched_subjects, upper_tri_indices


# =============================================================================
# BRAIN VISUALIZATION FUNCTIONS (from v13)
# =============================================================================

def create_surfplot_brain_correlation(correlations, modality):
    """Create brain surface visualization using HCP MMP parcellation with correlations"""
    try:
        import hcp_utils as hcp
        from neuromaps.datasets import fetch_fslr
        from surfplot import Plot
        import matplotlib.pyplot as plt

        print(f"Creating HCP MMP brain visualization for {modality.upper()} correlations...")

        # Ensure we have the right number of regions for HCP MMP (379)
        if len(correlations) < 379:
            padded_correlations = np.zeros(379)
            padded_correlations[:len(correlations)] = correlations
            correlations = padded_correlations
        elif len(correlations) > 379:
            correlations = correlations[:379]

        # Create brain map using HCP MMP parcellation
        brain_map = hcp.unparcellate(correlations, hcp.mmp)
        cortical_brain_map = hcp.cortex_data(brain_map)

        # Extract hemisphere data
        if isinstance(cortical_brain_map, (list, tuple)) and len(cortical_brain_map) == 2:
            L_cortx = np.asarray(cortical_brain_map[0], dtype=np.float64)
            R_cortx = np.asarray(cortical_brain_map[1], dtype=np.float64)
        else:
            cortical_array = np.asarray(cortical_brain_map, dtype=np.float64)
            mid_point = len(cortical_array) // 2
            L_cortx = cortical_array[:mid_point]
            R_cortx = cortical_array[mid_point:]

        # Get surfaces
        surfaces = fetch_fslr()
        lh, rh = surfaces['inflated']

        # Set up colormap for correlations
        vmax = max(abs(correlations)) if len(correlations) > 0 else 0.1
        vmin = -vmax
        cmap = 'RdBu_r'

        # Create plot
        p = Plot(lh, rh, layout='row', zoom=1.2, size=(1200, 600), brightness=0.1)

        # Main data layer
        p.add_layer({'left': L_cortx, 'right': R_cortx},
                    cbar=True,
                    cmap=cmap,
                    color_range=(vmin, vmax))

        # Add layer for black outlines
        p.add_layer({'left': L_cortx, 'right': R_cortx},
                    cmap='Greys',
                    as_outline=True,
                    cbar=False,
                    color_range=(0, 1))

        fig = p.build(cbar_kws={'decimals': 3})

        # Save
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = f"{OUTPUT_DIR}/{modality}_predicted_g_correlations_brain.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_file}")

        plt.close(fig)
        return True

    except Exception as e:
        print(f"  ✗ Brain visualization failed: {e}")
        return False


# =============================================================================
# FC VISUALIZATION FUNCTIONS (adapted from v8)
# =============================================================================

def load_cole_network_mapping():
    """Load Cole network mapping from CSV file"""
    try:
        mapping_df = pd.read_csv(COLE_MAPPING_FILE)
        region_to_network = {}
        for _, row in mapping_df.iterrows():
            region_name = row['name_Glasser']
            network_num = row['Cole_net_order']
            region_to_network[region_name] = network_num
        return region_to_network, mapping_df
    except Exception as e:
        print(f"Error loading Cole mapping: {e}")
        return {}, pd.DataFrame()


def create_region_to_network_mapping_fc():
    """Create region-to-network mapping for FC data (Cole network ordering)"""
    region_to_network, mapping_df = load_cole_network_mapping()

    if mapping_df.empty:
        print("Warning: Could not load Cole network mapping")
        return {}

    # FC uses Cole network ordering directly
    network_counts = mapping_df['Cole_net_order'].value_counts().sort_index()
    index_to_network = {}
    current_index = 0

    for network_num in sorted(network_counts.index):
        region_count = network_counts[network_num]
        for _ in range(region_count):
            index_to_network[current_index] = network_num
            current_index += 1

    # Add subcortical regions
    subcortical_networks = {
        'THALAMUS': 7, 'CAUDATE': 7, 'PUTAMEN': 3, 'PALLIDUM': 3,
        'BRAIN_STEM': 3, 'HIPPOCAMPUS': 9, 'AMYGDALA': 9,
        'ACCUMBENS': 7, 'DIENCEPHALON': 9, 'CEREBELLUM': 3
    }

    num_cortical = len(mapping_df)
    for i, region_name in enumerate(FC_REGION_NAMES[num_cortical:]):
        fc_index = num_cortical + i
        if fc_index < len(FC_REGION_NAMES):
            mapped = False
            for key, network in subcortical_networks.items():
                if key in region_name:
                    index_to_network[fc_index] = network
                    mapped = True
                    break
            if not mapped:
                index_to_network[fc_index] = 13  # Subcortical

    return index_to_network


def create_fc_correlation_heatmap(correlations, upper_tri_indices):
    """Create FC correlation heatmap with Cole network organization"""
    print("Creating FC correlation heatmap with Cole network organization...")

    # Load Cole network mapping for ordering
    region_to_network = create_region_to_network_mapping_fc()

    if not region_to_network:
        print("Warning: Could not load Cole network mapping")
        return False

    # Create correlation matrix
    n_regions = len(FC_REGION_NAMES)
    correlation_matrix = np.zeros((n_regions, n_regions))

    # Fill correlation matrix using upper triangle indices
    for conn_idx, corr_val in enumerate(correlations):
        i, j = upper_tri_indices[0][conn_idx], upper_tri_indices[1][conn_idx]
        correlation_matrix[i, j] = corr_val
        correlation_matrix[j, i] = corr_val  # Make symmetric

    # Sort regions by network for better visualization
    region_network_pairs = [(i, region_to_network.get(i, 13)) for i in range(n_regions)]
    region_network_pairs.sort(key=lambda x: x[1])  # Sort by network
    sorted_indices = [pair[0] for pair in region_network_pairs]

    # Create sorted matrix
    sorted_matrix = correlation_matrix[np.ix_(sorted_indices, sorted_indices)]
    sorted_region_names = [FC_REGION_NAMES[i] for i in sorted_indices]
    sorted_networks = [region_to_network.get(i, 13) for i in sorted_indices]

    # Create bottom-left triangular matrix
    mask = np.triu(np.ones_like(sorted_matrix, dtype=bool), k=1)
    sorted_matrix[mask] = np.nan

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Set up colormap for correlations
    vmax = max(abs(np.nanmin(sorted_matrix)), abs(np.nanmax(sorted_matrix)))
    vmin = -vmax
    cmap = 'RdBu_r'

    # Plot heatmap
    im = ax.imshow(sorted_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

    # Add network color bar on the left with labels (from v8 logic)
    def add_left_network_colorbar_with_labels(ax, sorted_networks, network_colors):
        label_positions = []
        current_network = None
        start_pos = 0

        for i, network in enumerate(sorted_networks):
            if network != current_network:
                if current_network is not None:
                    mid_pos = (start_pos + i - 1) / 2
                    label_positions.append((mid_pos, current_network))
                start_pos = i
                current_network = network

            # Add color bar on left
            color = network_colors.get(network, '#CCCCCC')
            ax.add_patch(patches.Rectangle((-25, i - 0.5), 20, 1,
                                           facecolor=color, transform=ax.transData,
                                           clip_on=False, linewidth=0))

        # Add last network
        if current_network is not None:
            mid_pos = (start_pos + len(sorted_networks) - 1) / 2
            label_positions.append((mid_pos, current_network))

        # Add left network labels
        for pos, network_id in label_positions:
            if network_id in NETWORK_NAMES:
                label_text = NETWORK_NAMES[network_id]
            else:
                label_text = f"Network_{network_id}"

            ax.text(-30, pos, label_text,
                    ha='right', va='center', rotation=0, fontsize=11,
                    transform=ax.transData, clip_on=False, fontweight='bold')

    # Add network color bar on the bottom x-axis
    def add_bottom_network_colorbar(ax, sorted_networks, network_colors):
        current_network = None
        start_pos = 0

        for i, network in enumerate(sorted_networks):
            if network != current_network:
                if current_network is not None:
                    # Add color bar for previous network
                    end_pos = i - 1
                    width = end_pos - start_pos + 1
                    color = network_colors.get(current_network, '#CCCCCC')
                    ax.add_patch(patches.Rectangle((start_pos - 0.5, len(sorted_networks) + 5), width, 15,
                                                   facecolor=color, transform=ax.transData,
                                                   clip_on=False, linewidth=0))
                start_pos = i
                current_network = network

        # Add color bar for last network
        if current_network is not None:
            end_pos = len(sorted_networks) - 1
            width = end_pos - start_pos + 1
            color = network_colors.get(current_network, '#CCCCCC')
            ax.add_patch(patches.Rectangle((start_pos - 0.5, len(sorted_networks) + 5), width, 15,
                                           facecolor=color, transform=ax.transData,
                                           clip_on=False, linewidth=0))

    # Add network color bars with labels
    add_left_network_colorbar_with_labels(ax, sorted_networks, NETWORK_COLORS)
    add_bottom_network_colorbar(ax, sorted_networks, NETWORK_COLORS)

    # Add grid lines at network boundaries
    network_boundaries = []
    current_network = sorted_networks[0]
    for i, network in enumerate(sorted_networks[1:], 1):
        if network != current_network:
            network_boundaries.append(i - 0.5)
            current_network = network

    for boundary in network_boundaries:
        ax.axvline(x=boundary, color='black', linewidth=2)
        ax.axhline(y=boundary, color='black', linewidth=2)

    # Customize plot
    ax.set_xlim(-0.5, len(sorted_indices) - 0.5)
    ax.set_ylim(len(sorted_indices) - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Title
    title_text = 'FC Correlations with Predicted g\n(Bottom-Left Triangle, Organized by Networks)'
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, aspect=25, pad=0.02)
    cbar.set_label('Correlation with Predicted g', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Save plot
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = f"{OUTPUT_DIR}/fc_predicted_g_correlations_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Saved FC correlation heatmap: {output_file}")
    return True


# =============================================================================
# NETWORK-LEVEL FC CORRELATION FUNCTION
# =============================================================================

def compute_network_level_fc_correlations(correlations, upper_tri_indices):
    """
    Computes network-level correlations with predicted g by aggregating all
    connections involving each Cole-Anticevic network.

    For each network, all connections where either region belongs to that network
    are collected. Individual Pearson r values are Fisher Z-transformed, averaged,
    then back-transformed to r. Networks are ranked by absolute mean correlation.

    Results are printed to terminal and saved as a text file.
    """
    print("\nComputing network-level FC correlations with predicted g...")

    # Load region-to-network mapping
    index_to_network = create_region_to_network_mapping_fc()

    if not index_to_network:
        print("  Warning: Could not load Cole network mapping, skipping network-level analysis")
        return

    # Collect connection r values for each network (all connections involving that network)
    network_connections = {net_id: [] for net_id in NETWORK_NAMES.keys()}

    for conn_idx, corr_val in enumerate(correlations):
        region_i = upper_tri_indices[0][conn_idx]
        region_j = upper_tri_indices[1][conn_idx]

        net_i = index_to_network.get(region_i, None)
        net_j = index_to_network.get(region_j, None)

        # Add to both networks this connection involves
        if net_i is not None and net_i in network_connections:
            network_connections[net_i].append(corr_val)
        if net_j is not None and net_j in network_connections and net_j != net_i:
            network_connections[net_j].append(corr_val)

    # Fisher Z transform, average within network, back-transform to r
    network_results = []

    for net_id, net_name in NETWORK_NAMES.items():
        r_values = np.array(network_connections[net_id])

        if len(r_values) == 0:
            print(f"  Warning: No connections found for {net_name}, skipping")
            continue

        # Clip to avoid arctanh blowing up at +/-1
        r_clipped = np.clip(r_values, -0.9999, 0.9999)

        # Fisher Z transform, average, back-transform
        z_values = np.arctanh(r_clipped)
        mean_z = np.mean(z_values)
        mean_r = np.tanh(mean_z)

        # Mean of absolute r values (unsigned average, for diagnostic comparison)
        mean_abs_r = np.tanh(np.mean(np.arctanh(np.abs(r_clipped))))

        network_results.append({
            'network_id': net_id,
            'network_name': net_name,
            'mean_r': mean_r,
            'abs_mean_r': abs(mean_r),
            'mean_abs_r': mean_abs_r,
            'n_connections': len(r_values)
        })

    # Sort by absolute mean r, descending
    network_results.sort(key=lambda x: x['abs_mean_r'], reverse=True)

    # Build output text
    output_lines = []
    output_lines.append("Network-Level FC Correlations with Predicted g")
    output_lines.append("=" * 60)
    output_lines.append("(Aggregated via Fisher Z transform across all connections")
    output_lines.append(" involving each network, sorted by |r|)")
    output_lines.append("")
    output_lines.append(f"{'Rank':<6}{'Network':<30}{'Mean r':<12}{'Mean |r|':<12}{'N connections'}")
    output_lines.append("-" * 72)

    for rank, result in enumerate(network_results, start=1):
        line = (
            f"{rank:<6}"
            f"{result['network_name']:<30}"
            f"{result['mean_r']:<12.4f}"
            f"{result['mean_abs_r']:<12.4f}"
            f"{result['n_connections']}"
        )
        output_lines.append(line)

    output_lines.append("")
    output_lines.append("Top 3 networks:")
    for result in network_results[:3]:
        output_lines.append(
            f"  {result['network_name']}: r = {result['mean_r']:.4f}, "
            f"|r| = {result['mean_abs_r']:.4f} "
            f"({result['n_connections']} connections)"
        )

    # Print to terminal
    print("\n".join(output_lines))

    # Save to text file
    output_file = f"{OUTPUT_DIR}/fc_network_level_correlations_with_predicted_g.txt"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("\n".join(output_lines))
        f.write("\n")

    print(f"\n  Saved network-level results to: {output_file}")
    return network_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("BRAIN CORRELATION VISUALIZATION WITH PREDICTED G")
    print("=" * 55)
    print("Features:")
    print("  ✓ Correlates raw brain data with predicted g-factor")
    print("  ✓ ALFF/ReHo: HCP MMP parcellation with surfplot")
    print("  ✓ FC: Cole network organization with correlation heatmaps")
    print("  ✓ Preserves L_H/R_H regions for proper surfplot functionality")
    print()

    # Check if required files exist
    if not os.path.exists(PRED_PATH):
        print(f"Error: Prediction path not found: {PRED_PATH}")
        return

    # Load stacked predictions
    print("Loading stacked model predictions...")
    predictions = load_stacked_predictions()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # ALFF ANALYSIS
    # =========================================================================
    print(f"\n{'=' * 50}")
    print("ALFF CORRELATION ANALYSIS")
    print(f"{'=' * 50}")

    try:
        subjects, brain_data = load_all_alff_reho_data('alff')
        correlations, p_values, matched_subjects = correlate_brain_with_predictions(
            subjects, brain_data, predictions, 'alff'
        )

        print(f"ALFF correlation results:")
        print(f"  Mean correlation: {correlations.mean():.3f}")
        print(f"  Range: [{correlations.min():.3f}, {correlations.max():.3f}]")
        print(f"  Regions with |r| > 0.1: {np.sum(np.abs(correlations) > 0.1)}")

        # Create brain visualization
        success = create_surfplot_brain_correlation(correlations, 'alff')
        if success:
            print("  ✓ ALFF brain correlation visualization created")

        # Save correlation data
        correlation_df = pd.DataFrame({
            'region_index': range(len(correlations)),
            'region_name': ALFF_REHO_REGION_NAMES[:len(correlations)],
            'correlation': correlations,
            'p_value': p_values
        })
        correlation_df.to_csv(f"{OUTPUT_DIR}/alff_predicted_g_correlations.csv", index=False)
        print("  ✓ ALFF correlation data saved")

    except Exception as e:
        print(f"Error in ALFF analysis: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # REHO ANALYSIS
    # =========================================================================
    print(f"\n{'=' * 50}")
    print("REHO CORRELATION ANALYSIS")
    print(f"{'=' * 50}")

    try:
        subjects, brain_data = load_all_alff_reho_data('reho')
        correlations, p_values, matched_subjects = correlate_brain_with_predictions(
            subjects, brain_data, predictions, 'reho'
        )

        print(f"ReHo correlation results:")
        print(f"  Mean correlation: {correlations.mean():.3f}")
        print(f"  Range: [{correlations.min():.3f}, {correlations.max():.3f}]")
        print(f"  Regions with |r| > 0.1: {np.sum(np.abs(correlations) > 0.1)}")

        # Create brain visualization
        success = create_surfplot_brain_correlation(correlations, 'reho')
        if success:
            print("  ✓ ReHo brain correlation visualization created")

        # Save correlation data
        correlation_df = pd.DataFrame({
            'region_index': range(len(correlations)),
            'region_name': ALFF_REHO_REGION_NAMES[:len(correlations)],
            'correlation': correlations,
            'p_value': p_values
        })
        correlation_df.to_csv(f"{OUTPUT_DIR}/reho_predicted_g_correlations.csv", index=False)
        print("  ✓ ReHo correlation data saved")

    except Exception as e:
        print(f"Error in ReHo analysis: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # FC ANALYSIS
    # =========================================================================
    print(f"\n{'=' * 50}")
    print("FC CORRELATION ANALYSIS")
    print(f"{'=' * 50}")

    try:
        subjects, fc_matrices = load_all_fc_data()
        correlations, p_values, matched_subjects, upper_tri_indices = correlate_brain_with_predictions(
            subjects, fc_matrices, predictions, 'functional_connectivity'
        )

        print(f"FC correlation results:")
        print(f"  Total connections: {len(correlations)}")
        print(f"  Mean correlation: {correlations.mean():.3f}")
        print(f"  Range: [{correlations.min():.3f}, {correlations.max():.3f}]")
        print(f"  Connections with |r| > 0.1: {np.sum(np.abs(correlations) > 0.1)}")

        # Create FC correlation heatmap
        success = create_fc_correlation_heatmap(correlations, upper_tri_indices)
        if success:
            print("  ✓ FC correlation heatmap created")

        # Save correlation data
        connection_names = []
        for conn_idx in range(len(correlations)):
            i, j = upper_tri_indices[0][conn_idx], upper_tri_indices[1][conn_idx]
            region_i = FC_REGION_NAMES[i] if i < len(FC_REGION_NAMES) else f"Region_{i}"
            region_j = FC_REGION_NAMES[j] if j < len(FC_REGION_NAMES) else f"Region_{j}"
            connection_names.append(f"{region_i}--{region_j}")

        correlation_df = pd.DataFrame({
            'connection_index': range(len(correlations)),
            'connection_name': connection_names,
            'region_i_index': upper_tri_indices[0],
            'region_j_index': upper_tri_indices[1],
            'correlation': correlations,
            'p_value': p_values
        })
        correlation_df.to_csv(f"{OUTPUT_DIR}/fc_predicted_g_correlations.csv", index=False)
        print("  ✓ FC correlation data saved")

        # Network-level aggregation
        compute_network_level_fc_correlations(correlations, upper_tri_indices)

    except Exception as e:
        print(f"Error in FC analysis: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'=' * 55}")
    print("CORRELATION ANALYSIS COMPLETE")
    print(f"{'=' * 55}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    print(f"  - ALFF brain correlation visualization")
    print(f"  - ReHo brain correlation visualization")
    print(f"  - FC correlation heatmap with Cole network organization")
    print(f"  - CSV files with detailed correlation results")
    print("\nKey features:")
    print(f"  ✓ Correlations with predicted g-factor from stacked model")
    print(f"  ✓ HCP MMP parcellation preserved for brain visualizations")
    print(f"  ✓ Cole network organization for FC visualization")
    print(f"  ✓ L_H/R_H regions maintained for surfplot compatibility")


if __name__ == "__main__":
    main()