#!/usr/bin/env python3
"""
Complete Neuroimaging Alignment Fix

Aligns ALL neuroimaging modalities to functional_connectivity baseline.
"""

import os
import pandas as pd
import logging
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = "/media/hcs-sci-psy-narun/Jack/First_level_model_ADHDRS_6month"

STRUCTURAL_MODALITIES = [
    "cortical_area",
    "cortical_thickness",
    "subcortical_volume",
    "total_brain_volume"
]

FUNCTIONAL_MODALITIES = [
    "alff",
    "reho"
    # functional_connectivity is the baseline, so we don't filter it
]


def main():
    """Complete alignment of all modalities to functional_connectivity baseline."""
    logger.info("=== COMPLETE NEUROIMAGING ALIGNMENT FIX ===")

    # Step 1: Get ALL timepoints from functional connectivity (baseline)
    logger.info("\n1. Loading functional connectivity timepoints (BASELINE)...")
    fc_timepoints = set()

    for fold in range(5):
        for split in ['train', 'test']:
            fc_path = os.path.join(BASE_DIR, "functional", "functional_connectivity", f"fold_{fold}", f"{split}.csv")
            if os.path.exists(fc_path):
                df = pd.read_csv(fc_path)
                timepoints_in_file = set(df['eid'].tolist())
                fc_timepoints.update(timepoints_in_file)
                logger.info(f"  FC fold {fold} {split}: {len(df)} timepoints")

    logger.info(f"  ✓ BASELINE: {len(fc_timepoints)} unique timepoints from functional_connectivity")

    # Step 2: Check all modalities and show differences
    logger.info("\n2. Checking alignment for ALL modalities...")
    modality_info = {}

    # Check structural modalities
    for modality in STRUCTURAL_MODALITIES:
        logger.info(f"\n  Checking structural/{modality}...")
        modality_timepoints = set()

        for fold in range(5):
            for split in ['train', 'test']:
                path = os.path.join(BASE_DIR, "structural", modality, f"fold_{fold}", f"{split}.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    timepoints_in_file = set(df['eid'].tolist())
                    modality_timepoints.update(timepoints_in_file)
                    logger.info(f"    {modality} fold {fold} {split}: {len(df)} timepoints")

        extra = modality_timepoints - fc_timepoints
        missing = fc_timepoints - modality_timepoints

        modality_info[f"structural/{modality}"] = {
            'total': len(modality_timepoints),
            'extra': len(extra),
            'missing': len(missing),
            'needs_filtering': len(extra) > 0 or len(missing) > 0
        }

        logger.info(f"    {modality} total: {len(modality_timepoints)}, extra: {len(extra)}, missing: {len(missing)}")

    # Check functional modalities
    for modality in FUNCTIONAL_MODALITIES:
        logger.info(f"\n  Checking functional/{modality}...")
        modality_timepoints = set()

        for fold in range(5):
            for split in ['train', 'test']:
                path = os.path.join(BASE_DIR, "functional", modality, f"fold_{fold}", f"{split}.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    timepoints_in_file = set(df['eid'].tolist())
                    modality_timepoints.update(timepoints_in_file)
                    logger.info(f"    {modality} fold {fold} {split}: {len(df)} timepoints")

        extra = modality_timepoints - fc_timepoints
        missing = fc_timepoints - modality_timepoints

        modality_info[f"functional/{modality}"] = {
            'total': len(modality_timepoints),
            'extra': len(extra),
            'missing': len(missing),
            'needs_filtering': len(extra) > 0 or len(missing) > 0
        }

        logger.info(f"    {modality} total: {len(modality_timepoints)}, extra: {len(extra)}, missing: {len(missing)}")

    # Step 3: Show summary
    logger.info("\n3. ALIGNMENT SUMMARY:")
    modalities_needing_filter = []

    for modality, info in modality_info.items():
        status = "NEEDS FILTERING" if info['needs_filtering'] else "ALIGNED"
        logger.info(
            f"  {modality}: {info['total']} timepoints, +{info['extra']} extra, -{info['missing']} missing → {status}")

        if info['needs_filtering']:
            modalities_needing_filter.append(modality)

    if not modalities_needing_filter:
        logger.info("   ALL MODALITIES ALREADY ALIGNED!")
        return

    logger.info(f"\n   Need to filter {len(modalities_needing_filter)} modalities to match FC baseline")

    # Step 4: Filter ALL modalities that need it
    logger.info("\n4. FILTERING ALL MODALITIES TO FUNCTIONAL_CONNECTIVITY BASELINE...")

    total_filtered_files = 0
    total_removed_timepoints = 0

    for modality_key in modalities_needing_filter:
        category, modality = modality_key.split('/')
        logger.info(f"\n  Filtering {modality_key}...")

        modality_removed = 0

        for fold in range(5):
            for split in ['train', 'test']:
                file_path = os.path.join(BASE_DIR, category, modality, f"fold_{fold}", f"{split}.csv")

                if os.path.exists(file_path):
                    # Load original
                    df = pd.read_csv(file_path)
                    original_count = len(df)

                    # Create backup
                    backup_path = file_path.replace('.csv', '_pre_fc_filter.csv')
                    if not os.path.exists(backup_path):
                        shutil.copy2(file_path, backup_path)
                        logger.info(f"    Backup: {backup_path}")

                    # Filter to only FC timepoints
                    filtered_df = df[df['eid'].isin(fc_timepoints)]
                    filtered_count = len(filtered_df)

                    # Save filtered version
                    filtered_df.to_csv(file_path, index=False)

                    removed = original_count - filtered_count
                    modality_removed += removed
                    total_removed_timepoints += removed
                    total_filtered_files += 1

                    if removed > 0:
                        logger.info(
                            f"    {modality} fold {fold} {split}: {original_count} → {filtered_count} (removed {removed})")
                    else:
                        logger.info(f"    {modality} fold {fold} {split}: {original_count} (no change)")

        logger.info(f"  ✓ {modality_key}: removed {modality_removed} total timepoints")

    logger.info(f"\n  FILTERING COMPLETE:")
    logger.info(f"    Files processed: {total_filtered_files}")
    logger.info(f"    Total timepoints removed: {total_removed_timepoints}")

    # Step 5: Verify ALL modalities are now aligned
    logger.info("\n5. VERIFICATION - Checking all modalities after filtering...")

    verification_passed = True

    # Re-check all modalities
    all_modalities = [f"structural/{m}" for m in STRUCTURAL_MODALITIES] + [f"functional/{m}" for m in
                                                                           FUNCTIONAL_MODALITIES]

    for modality_key in all_modalities:
        category, modality = modality_key.split('/')
        modality_timepoints = set()

        for fold in range(5):
            for split in ['train', 'test']:
                path = os.path.join(BASE_DIR, category, modality, f"fold_{fold}", f"{split}.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    timepoints_in_file = set(df['eid'].tolist())
                    modality_timepoints.update(timepoints_in_file)

        if modality_timepoints == fc_timepoints:
            logger.info(f"   {modality_key}: {len(modality_timepoints)} timepoints (PERFECTLY ALIGNED)")
        else:
            extra = len(modality_timepoints - fc_timepoints)
            missing = len(fc_timepoints - modality_timepoints)
            logger.error(
                f"   {modality_key}: {len(modality_timepoints)} timepoints (+{extra} extra, -{missing} missing)")
            verification_passed = False

    # Final status
    if verification_passed:
        logger.info(f"\n SUCCESS: ALL NEUROIMAGING MODALITIES NOW PERFECTLY ALIGNED!")
        logger.info(f"   All modalities have exactly {len(fc_timepoints)} timepoints")
        logger.info(f"   Ready for ADHD-RS alignment and Nature Medicine submission!")
    else:
        logger.error(f"\n FAILED: Some modalities still not aligned - check errors above")


if __name__ == "__main__":
    main()