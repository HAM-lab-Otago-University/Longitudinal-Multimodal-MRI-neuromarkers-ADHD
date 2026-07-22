#!/usr/bin/env python3
"""
Baseline G-Factor Hedges' g Analysis Script

This script extracts baseline (first timepoint) g-factor scores for each participant
and computes Hedges' g effect sizes with bootstrapped confidence intervals between
all pairwise combinations of phenotype groups.
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway
from scipy.stats import levene, shapiro
import itertools
import os
import logging
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple


# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.libqsturng import psturng
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("Statsmodels not available. Please install Statsmodels first.")

# File paths (same as original script)
BASE_DATA_PATH = '/media/hcs-sci-psy-narun/Jack'
G_FACTOR_FILE = os.path.join(BASE_DATA_PATH, 'g_factor_analysis/longitudinal_g_factor_regularized_with_metadata.csv')
NDAR_FILE = os.path.join(BASE_DATA_PATH, "ndar_subject01.csv")

# Define phenotype labels (same as original script)
PHENOTYPE_LABELS = {
    1: "Control",
    2: "Subthreshold",
    3: "ADHD",
    4: "Non-ADHD/Non-Control"
}

# Bootstrap parameters
N_BOOTSTRAP = 5000
CONFIDENCE_LEVEL = 0.95


@dataclass
class HedgesGResult:
    """Container for Hedges' g analysis results."""
    group1_name: str
    group2_name: str
    group1_n: int
    group2_n: int
    group1_mean: float
    group1_std: float
    group2_mean: float
    group2_std: float
    hedges_g: float
    ci_lower: float
    ci_upper: float
    p_value: float

@dataclass
class ANOVAResult:
    """Container for ANOVA analysis results."""
    f_statistic: float
    p_value: float
    groups: List[str]
    group_means: List[float]
    group_stds: List[float]
    group_ns: List[int]
    significant: bool


@dataclass
class PostHocResult:
    """Container for post-hoc comparison results."""
    group1_name: str
    group2_name: str
    group1_n: int
    group2_n: int
    group1_mean: float
    group1_std: float
    group2_mean: float
    group2_std: float
    hedges_g: float
    ci_lower: float
    ci_upper: float
    p_value_uncorrected: float
    p_value_corrected: float
    significant_corrected: bool
    correction_method: str


def load_ndar_phenotypes(ndar_file):
    """
    Load NDAR subjects file and extract phenotype information.
    (Copied from original script)
    """
    logger.info(f"Loading phenotype data from {ndar_file}...")

    try:
        ndar_df = pd.read_csv(ndar_file)
        logger.info(f"Loaded NDAR file with {len(ndar_df)} subjects and {len(ndar_df.columns)} columns")

        # Create mapping from NDAR ID to phenotype
        phenotype_map = {}
        raw_phenotype_map = {}

        for _, row in ndar_df.iterrows():
            # Extract subject key
            if 'subjectkey' in row and not pd.isna(row['subjectkey']):
                subject_key = str(row['subjectkey'])
            elif 'ndar_subject01_id' in row and not pd.isna(row['ndar_subject01_id']):
                subject_key = str(row['ndar_subject01_id'])
            else:
                continue

            raw_subject_key = subject_key
            subject_key = subject_key.replace("_", "")

            if not subject_key.upper().startswith("NDAR"):
                subject_key = "NDAR" + subject_key

            # Get phenotype
            phenotype = None
            if 'phenotype' in row and not pd.isna(row['phenotype']):
                try:
                    phenotype = int(float(row['phenotype']))
                except (ValueError, TypeError):
                    phenotype = None

            if phenotype is None and 'sibling_study' in row and not pd.isna(row['sibling_study']):
                try:
                    phenotype = int(float(row['sibling_study']))
                    if phenotype < 1 or phenotype > 4:
                        phenotype = None
                except (ValueError, TypeError):
                    phenotype = None

            if phenotype is not None:
                phenotype_map[subject_key.upper()] = phenotype
                raw_phenotype_map[raw_subject_key] = phenotype
                phenotype_map[subject_key.lower()] = phenotype
                phenotype_map[subject_key.replace("NDAR", "")] = phenotype

        logger.info(f"Found valid phenotype information for {len(raw_phenotype_map)} subjects")
        return phenotype_map, raw_phenotype_map

    except Exception as e:
        logger.error(f"Error loading NDAR phenotypes: {e}")
        return {}, {}


def load_g_factor_data(file_path):
    """
    Load the g-factor data from CSV file.
    (Adapted from original script)
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded g-factor data with {len(df)} rows and {len(df.columns)} columns")

        required_cols = ['subject_id', 'age_years', 'g']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()

        # Convert to numeric
        df['age_years'] = pd.to_numeric(df['age_years'], errors='coerce')
        df['g'] = pd.to_numeric(df['g'], errors='coerce')

        # Remove rows with NaN in essential columns
        valid_data = df.dropna(subset=['subject_id', 'age_years', 'g'])

        if len(valid_data) < len(df):
            logger.info(f"Filtered out {len(df) - len(valid_data)} rows with NaN values")
            df = valid_data

        return df

    except Exception as e:
        logger.error(f"Error loading g-factor data: {e}")
        return pd.DataFrame()


def add_phenotype_to_g_factor_data(g_df, phenotype_map):
    """
    Add phenotype information to g-factor data.
    (Copied from original script)
    """
    result_df = g_df.copy()
    result_df['phenotype'] = None
    match_count = 0

    for subject_id in result_df['subject_id'].unique():
        key_variants = [
            subject_id,
            subject_id.upper(),
            subject_id.lower(),
            subject_id.replace("NDAR", "") if subject_id.upper().startswith("NDAR") else subject_id,
            "NDAR" + subject_id if not subject_id.upper().startswith("NDAR") else subject_id
        ]

        phenotype = None
        for key_variant in key_variants:
            if key_variant in phenotype_map:
                phenotype = phenotype_map[key_variant]
                match_count += 1
                break

        if phenotype is not None:
            result_df.loc[result_df['subject_id'] == subject_id, 'phenotype'] = phenotype

    result_df['phenotype'] = pd.to_numeric(result_df['phenotype'], errors='ignore')

    total_subjects = g_df['subject_id'].nunique()
    match_percentage = match_count / total_subjects * 100 if total_subjects > 0 else 0
    logger.info(f"Added phenotypes to g-factor data. Found matches for {match_count}/{total_subjects} " +
                f"subjects ({match_percentage:.1f}%)")

    return result_df


def extract_baseline_g_scores(df):
    """
    Extract baseline (first/earliest timepoint) g-factor scores for each participant.

    Parameters:
    df (DataFrame): DataFrame with g-factor data and phenotypes

    Returns:
    DataFrame: One row per participant with their baseline g-score
    """
    logger.info("Extracting baseline g-factor scores...")

    baseline_data = []

    for subject_id in df['subject_id'].unique():
        subject_data = df[df['subject_id'] == subject_id].copy()

        # Skip if no valid age data
        if subject_data['age_years'].isna().all():
            continue

        # Find the earliest timepoint (minimum age)
        baseline_idx = subject_data['age_years'].idxmin()
        baseline_row = subject_data.loc[baseline_idx]

        baseline_data.append({
            'subject_id': subject_id,
            'baseline_age': baseline_row['age_years'],
            'baseline_g': baseline_row['g'],
            'phenotype': baseline_row['phenotype'] if 'phenotype' in baseline_row else None
        })

    baseline_df = pd.DataFrame(baseline_data)

    logger.info(f"Extracted baseline data for {len(baseline_df)} participants")

    # Log distribution by phenotype
    if 'phenotype' in baseline_df.columns:
        phenotype_counts = baseline_df['phenotype'].value_counts().sort_index()
        for phenotype, count in phenotype_counts.items():
            if phenotype in PHENOTYPE_LABELS:
                logger.info(f"  - {PHENOTYPE_LABELS[phenotype]}: {count} participants")

    return baseline_df


def calculate_hedges_g(group1_scores, group2_scores):
    """
    Calculate Hedges' g effect size between two groups.

    Parameters:
    group1_scores, group2_scores: arrays of scores for each group

    Returns:
    float: Hedges' g effect size
    """
    n1, n2 = len(group1_scores), len(group2_scores)
    mean1, mean2 = np.mean(group1_scores), np.mean(group2_scores)
    std1, std2 = np.std(group1_scores, ddof=1), np.std(group2_scores, ddof=1)

    # Calculate pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

    # Calculate Cohen's d
    cohens_d = (mean1 - mean2) / pooled_std

    # Apply small sample bias correction for Hedges' g
    j = 1 - (3 / (4 * (n1 + n2) - 9))
    hedges_g = cohens_d * j

    return hedges_g


def bootstrap_hedges_g(group1_scores, group2_scores, n_bootstrap=N_BOOTSTRAP):
    """
    Bootstrap Hedges' g to get confidence intervals.

    Parameters:
    group1_scores, group2_scores: arrays of scores for each group
    n_bootstrap: number of bootstrap samples

    Returns:
    tuple: (hedges_g_original, ci_lower, ci_upper, bootstrap_distribution)
    """
    # Calculate original Hedges' g
    original_hedges_g = calculate_hedges_g(group1_scores, group2_scores)

    # Bootstrap
    bootstrap_hedges_g = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        boot_group1 = np.random.choice(group1_scores, size=len(group1_scores), replace=True)
        boot_group2 = np.random.choice(group2_scores, size=len(group2_scores), replace=True)

        # Calculate Hedges' g for bootstrap sample
        boot_hedges_g = calculate_hedges_g(boot_group1, boot_group2)
        bootstrap_hedges_g.append(boot_hedges_g)

    bootstrap_hedges_g = np.array(bootstrap_hedges_g)

    # Calculate confidence intervals
    alpha = 1 - CONFIDENCE_LEVEL
    ci_lower = np.percentile(bootstrap_hedges_g, (alpha / 2) * 100)
    ci_upper = np.percentile(bootstrap_hedges_g, (1 - alpha / 2) * 100)

    return original_hedges_g, ci_lower, ci_upper, bootstrap_hedges_g


def perform_pairwise_comparisons(baseline_df):
    """
    Perform all pairwise comparisons between phenotype groups.

    Parameters:
    baseline_df: DataFrame with baseline g-scores and phenotypes

    Returns:
    List[HedgesGResult]: Results for each pairwise comparison
    """
    logger.info("Performing pairwise comparisons with bootstrapped confidence intervals...")

    results = []

    # Get unique phenotypes (excluding NaN)
    phenotypes = [p for p in baseline_df['phenotype'].unique() if not pd.isna(p) and p in PHENOTYPE_LABELS]
    phenotypes.sort()

    # Generate all pairwise combinations
    for phenotype1, phenotype2 in itertools.combinations(phenotypes, 2):
        group1_data = baseline_df[baseline_df['phenotype'] == phenotype1]['baseline_g'].dropna()
        group2_data = baseline_df[baseline_df['phenotype'] == phenotype2]['baseline_g'].dropna()

        if len(group1_data) < 2 or len(group2_data) < 2:
            logger.warning(
                f"Insufficient data for comparison: {PHENOTYPE_LABELS[phenotype1]} vs {PHENOTYPE_LABELS[phenotype2]}")
            continue

        # Calculate descriptive statistics
        group1_mean = np.mean(group1_data)
        group1_std = np.std(group1_data, ddof=1)
        group2_mean = np.mean(group2_data)
        group2_std = np.std(group2_data, ddof=1)

        # Perform t-test for p-value
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)  # Welch's t-test



        # Bootstrap Hedges' g
        hedges_g, ci_lower, ci_upper, bootstrap_dist = bootstrap_hedges_g(group1_data, group2_data)

        # Create result object
        result = HedgesGResult(
            group1_name=PHENOTYPE_LABELS[phenotype1],
            group2_name=PHENOTYPE_LABELS[phenotype2],
            group1_n=len(group1_data),
            group2_n=len(group2_data),
            group1_mean=group1_mean,
            group1_std=group1_std,
            group2_mean=group2_mean,
            group2_std=group2_std,
            hedges_g=hedges_g,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value
        )

        results.append(result)

        logger.info(f"{result.group1_name} vs {result.group2_name}: " +
                    f"Hedges' g = {hedges_g:.3f} [{ci_lower:.3f}, {ci_upper:.3f}], p = {p_value:.4f}")

    return results


def save_results_to_csv(results, baseline_df, output_path):
    """
    Save results to CSV file.

    Parameters:
    results: List of HedgesGResult objects
    baseline_df: DataFrame with baseline data for descriptive statistics
    output_path: Path to save CSV file
    """
    # Create summary table
    summary_data = []

    for result in results:
        summary_data.append({
            'Comparison': f"{result.group1_name} vs {result.group2_name}",
            'Group1_Name': result.group1_name,
            'Group1_N': result.group1_n,
            'Group1_Mean': result.group1_mean,
            'Group1_SD': result.group1_std,
            'Group2_Name': result.group2_name,
            'Group2_N': result.group2_n,
            'Group2_Mean': result.group2_mean,
            'Group2_SD': result.group2_std,
            'Hedges_g': result.hedges_g,
            'CI_Lower': result.ci_lower,
            'CI_Upper': result.ci_upper,
            'P_Value': result.p_value,
            'Significant': 'Yes' if result.p_value < 0.05 else 'No',
            'Effect_Size_Interpretation': interpret_effect_size(abs(result.hedges_g))
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

    # Also create descriptive statistics by group
    desc_path = output_path.replace('.csv', '_descriptives.csv')
    create_descriptive_table(baseline_df, desc_path)


def interpret_effect_size(hedges_g):
    """Interpret Hedges' g effect size magnitude."""
    if hedges_g < 0.2:
        return "Negligible"
    elif hedges_g < 0.5:
        return "Small"
    elif hedges_g < 0.8:
        return "Medium"
    else:
        return "Large"


def create_descriptive_table(baseline_df, output_path):
    """Create descriptive statistics table by phenotype group."""
    desc_data = []

    for phenotype in sorted(baseline_df['phenotype'].unique()):
        if pd.isna(phenotype) or phenotype not in PHENOTYPE_LABELS:
            continue

        group_data = baseline_df[baseline_df['phenotype'] == phenotype]['baseline_g'].dropna()

        if len(group_data) > 0:
            desc_data.append({
                'Group': PHENOTYPE_LABELS[phenotype],
                'N': len(group_data),
                'Mean': np.mean(group_data),
                'SD': np.std(group_data, ddof=1),
                'Min': np.min(group_data),
                'Max': np.max(group_data),
                'Median': np.median(group_data),
                'Q25': np.percentile(group_data, 25),
                'Q75': np.percentile(group_data, 75)
            })

    desc_df = pd.DataFrame(desc_data)
    desc_df.to_csv(output_path, index=False)
    logger.info(f"Descriptive statistics saved to {output_path}")


def print_results_summary(results):
    """Print a formatted summary of the results."""
    print("\n" + "=" * 80)
    print("BASELINE G-FACTOR HEDGES' G ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Bootstrap samples: {N_BOOTSTRAP:,}")
    print(f"Confidence level: {CONFIDENCE_LEVEL * 100}%")
    print()

    for result in results:
        print(f"{result.group1_name} vs {result.group2_name}:")
        print(f"  Sample sizes: n₁={result.group1_n}, n₂={result.group2_n}")
        print(f"  Means: M₁={result.group1_mean:.3f}, M₂={result.group2_mean:.3f}")
        print(f"  Hedges' g = {result.hedges_g:.3f} [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
        print(
            f"  p-value = {result.p_value:.4f} {'***' if result.p_value < 0.001 else '**' if result.p_value < 0.01 else '*' if result.p_value < 0.05 else ''}")
        print(f"  Effect size: {interpret_effect_size(abs(result.hedges_g))}")
        print()


def perform_anova_analysis(baseline_df):
    """
    Perform one-way ANOVA on baseline g-scores across phenotype groups.

    Parameters:
    baseline_df: DataFrame with baseline g-scores and phenotypes

    Returns:
    ANOVAResult: ANOVA results
    """
    logger.info("Performing one-way ANOVA...")

    # Get unique phenotypes (excluding NaN)
    phenotypes = [p for p in baseline_df['phenotype'].unique()
                  if not pd.isna(p) and p in PHENOTYPE_LABELS]
    phenotypes.sort()

    # Prepare data for ANOVA
    groups_data = []
    group_names = []
    group_means = []
    group_stds = []
    group_ns = []

    for phenotype in phenotypes:
        group_data = baseline_df[baseline_df['phenotype'] == phenotype]['baseline_g'].dropna()
        if len(group_data) >= 2:  # Need at least 2 observations per group
            groups_data.append(group_data.values)
            group_names.append(PHENOTYPE_LABELS[phenotype])
            group_means.append(np.mean(group_data))
            group_stds.append(np.std(group_data, ddof=1))
            group_ns.append(len(group_data))

    if len(groups_data) < 2:
        logger.error("Need at least 2 groups for ANOVA")
        return None

    # Perform one-way ANOVA
    f_stat, p_value = f_oneway(*groups_data)

    result = ANOVAResult(
        f_statistic=f_stat,
        p_value=p_value,
        groups=group_names,
        group_means=group_means,
        group_stds=group_stds,
        group_ns=group_ns,
        significant=p_value < 0.05
    )

    logger.info(
        f"ANOVA: F({len(groups_data) - 1}, {sum(group_ns) - len(groups_data)}) = {f_stat:.3f}, p = {p_value:.4f}")

    return result


def perform_posthoc_analysis(baseline_df, method='tukey'):
    """
    Perform post-hoc analysis following significant ANOVA.

    Parameters:
    baseline_df: DataFrame with baseline g-scores and phenotypes
    method: 'tukey' for Tukey's HSD, 'bonferroni' for Bonferroni correction

    Returns:
    List[PostHocResult]: Results for each pairwise comparison
    """
    logger.info(f"Performing post-hoc analysis with {method} correction...")

    results = []

    # Get unique phenotypes (excluding NaN)
    phenotypes = [p for p in baseline_df['phenotype'].unique()
                  if not pd.isna(p) and p in PHENOTYPE_LABELS]
    phenotypes.sort()

    # Prepare data for post-hoc tests
    if method == 'tukey' and STATSMODELS_AVAILABLE:
        try:
            # Try the main Tukey implementation first
            results = _perform_tukey_hsd_robust(baseline_df, phenotypes)
        except Exception as e:
            logger.warning(f"Primary Tukey method failed: {e}")
            logger.warning("Falling back to Bonferroni correction")
            results = _perform_bonferroni_correction(baseline_df, phenotypes)
    else:
        # Use Bonferroni correction with pairwise t-tests
        results = _perform_bonferroni_correction(baseline_df, phenotypes)

    return results


def perform_posthoc_analysis(baseline_df, method='tukey'):
    """
    Perform post-hoc analysis following significant ANOVA.

    Parameters:
    baseline_df: DataFrame with baseline g-scores and phenotypes
    method: 'tukey' for Tukey's HSD, 'bonferroni' for Bonferroni correction

    Returns:
    List[PostHocResult]: Results for each pairwise comparison
    """
    logger.info(f"Performing post-hoc analysis with {method} correction...")

    results = []

    # Get unique phenotypes (excluding NaN)
    phenotypes = [p for p in baseline_df['phenotype'].unique()
                  if not pd.isna(p) and p in PHENOTYPE_LABELS]
    phenotypes.sort()

    # Prepare data for post-hoc tests
    if method == 'tukey' and STATSMODELS_AVAILABLE:
        try:
            # Try the robust Tukey implementation
            results = _perform_tukey_hsd_robust(baseline_df, phenotypes)
        except Exception as e:
            logger.warning(f"Tukey method failed: {e}")
            logger.warning("Falling back to Bonferroni correction")
            results = _perform_bonferroni_correction(baseline_df, phenotypes)
    else:
        # Use Bonferroni correction with pairwise t-tests
        results = _perform_bonferroni_correction(baseline_df, phenotypes)

    return results


def _perform_tukey_hsd_robust(baseline_df, phenotypes):
    """Robust Tukey's HSD post-hoc test with better error handling."""
    results = []

    # Prepare data for statsmodels
    all_scores = []
    all_groups = []

    for phenotype in phenotypes:
        group_data = baseline_df[baseline_df['phenotype'] == phenotype]['baseline_g'].dropna()
        all_scores.extend(group_data.values)
        all_groups.extend([PHENOTYPE_LABELS[phenotype]] * len(group_data))

    # Perform Tukey's HSD
    tukey_results = pairwise_tukeyhsd(all_scores, all_groups, alpha=0.05)

    # Convert results to a more usable format
    tukey_df = pd.DataFrame(data=tukey_results.summary().data[1:],
                            columns=tukey_results.summary().data[0])

    logger.info(f"Tukey results columns: {list(tukey_df.columns)}")
    logger.info(f"Tukey results shape: {tukey_df.shape}")

    # Extract pairwise results
    for group1, group2 in itertools.combinations([PHENOTYPE_LABELS[p] for p in phenotypes], 2):

        # Get data for each group
        group1_phenotype = [k for k, v in PHENOTYPE_LABELS.items() if v == group1][0]
        group2_phenotype = [k for k, v in PHENOTYPE_LABELS.items() if v == group2][0]

        group1_data = baseline_df[baseline_df['phenotype'] == group1_phenotype]['baseline_g'].dropna()
        group2_data = baseline_df[baseline_df['phenotype'] == group2_phenotype]['baseline_g'].dropna()

        if len(group1_data) < 2 or len(group2_data) < 2:
            continue

        # Calculate descriptive statistics
        group1_mean = np.mean(group1_data)
        group1_std = np.std(group1_data, ddof=1)
        group2_mean = np.mean(group2_data)
        group2_std = np.std(group2_data, ddof=1)

        # Get uncorrected p-value from t-test
        t_stat, p_uncorrected = stats.ttest_ind(group1_data, group2_data, equal_var=False)

        # Find the corresponding row in Tukey results
        p_corrected = p_uncorrected
        significant = False

        # Look for the comparison in both directions
        tukey_row = tukey_df[
            ((tukey_df.iloc[:, 0] == group1) & (tukey_df.iloc[:, 1] == group2)) |
            ((tukey_df.iloc[:, 0] == group2) & (tukey_df.iloc[:, 1] == group1))
            ]

        if not tukey_row.empty:
            try:
                # Try different possible column names for p-adjusted
                possible_p_cols = ['p-adj', 'p_adj', 'padj', 'pvalue_adj', 'p-value', 'pvalue']
                p_col_found = None

                for col_name in possible_p_cols:
                    if col_name in tukey_df.columns:
                        p_col_found = col_name
                        break

                if p_col_found:
                    p_corrected = float(tukey_row.iloc[0][p_col_found])
                else:
                    # If no p-adj column found, try by position (usually column 6 or 4)
                    if tukey_df.shape[1] > 6:
                        p_corrected = float(tukey_row.iloc[0, 6])
                    elif tukey_df.shape[1] > 4:
                        p_corrected = float(tukey_row.iloc[0, 4])
                    else:
                        logger.warning(f"Could not find p-adjusted column for {group1} vs {group2}")
                        p_corrected = p_uncorrected

                significant = p_corrected < 0.05
                logger.info(f"Tukey result for {group1} vs {group2}: p_adj = {p_corrected:.6f}")

            except (ValueError, IndexError, KeyError) as e:
                logger.warning(f"Error parsing Tukey result for {group1} vs {group2}: {e}")
                p_corrected = p_uncorrected
                significant = p_corrected < 0.05
        else:
            logger.warning(f"Could not find Tukey result for {group1} vs {group2}")
            p_corrected = p_uncorrected
            significant = p_corrected < 0.05

        # Bootstrap Hedges' g
        hedges_g, ci_lower, ci_upper, _ = bootstrap_hedges_g(group1_data, group2_data)

        result = PostHocResult(
            group1_name=group1,
            group2_name=group2,
            group1_n=len(group1_data),
            group2_n=len(group2_data),
            group1_mean=group1_mean,
            group1_std=group1_std,
            group2_mean=group2_mean,
            group2_std=group2_std,
            hedges_g=hedges_g,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value_uncorrected=p_uncorrected,
            p_value_corrected=p_corrected,
            significant_corrected=significant,
            correction_method='Tukey HSD'
        )

        results.append(result)

    return results



def _perform_bonferroni_correction(baseline_df, phenotypes):
    """Perform Bonferroni correction on pairwise t-tests."""
    results = []

    # Calculate number of comparisons
    n_comparisons = len(list(itertools.combinations(phenotypes, 2)))
    bonferroni_alpha = 0.05 / n_comparisons

    for phenotype1, phenotype2 in itertools.combinations(phenotypes, 2):
        group1_data = baseline_df[baseline_df['phenotype'] == phenotype1]['baseline_g'].dropna()
        group2_data = baseline_df[baseline_df['phenotype'] == phenotype2]['baseline_g'].dropna()

        if len(group1_data) < 2 or len(group2_data) < 2:
            continue

        # Calculate descriptive statistics
        group1_mean = np.mean(group1_data)
        group1_std = np.std(group1_data, ddof=1)
        group2_mean = np.mean(group2_data)
        group2_std = np.std(group2_data, ddof=1)

        # Perform t-test
        t_stat, p_uncorrected = stats.ttest_ind(group1_data, group2_data, equal_var=False)

        # Apply Bonferroni correction
        p_corrected = min(p_uncorrected * n_comparisons, 1.0)
        significant = p_corrected < 0.05

        # Bootstrap Hedges' g (keep your existing function)
        hedges_g, ci_lower, ci_upper, _ = bootstrap_hedges_g(group1_data, group2_data)

        result = PostHocResult(
            group1_name=PHENOTYPE_LABELS[phenotype1],
            group2_name=PHENOTYPE_LABELS[phenotype2],
            group1_n=len(group1_data),
            group2_n=len(group2_data),
            group1_mean=group1_mean,
            group1_std=group1_std,
            group2_mean=group2_mean,
            group2_std=group2_std,
            hedges_g=hedges_g,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value_uncorrected=p_uncorrected,
            p_value_corrected=p_corrected,
            significant_corrected=significant,
            correction_method=f'Bonferroni (α = {bonferroni_alpha:.4f})'
        )

        results.append(result)

    return results


def calculate_anova_effect_size(baseline_df, anova_result):
    """
    Calculate eta squared (η²) and omega squared (ω²) for ANOVA.
    """
    # Get data for effect size calculation
    phenotypes = [p for p in baseline_df['phenotype'].unique()
                  if not pd.isna(p) and p in PHENOTYPE_LABELS]

    # Calculate total sum of squares
    all_scores = baseline_df['baseline_g'].dropna()
    grand_mean = np.mean(all_scores)
    ss_total = np.sum((all_scores - grand_mean) ** 2)

    # Calculate between-groups sum of squares
    ss_between = 0
    for phenotype in phenotypes:
        group_data = baseline_df[baseline_df['phenotype'] == phenotype]['baseline_g'].dropna()
        if len(group_data) > 0:
            group_mean = np.mean(group_data)
            ss_between += len(group_data) * (group_mean - grand_mean) ** 2

    # Calculate within-groups sum of squares
    ss_within = ss_total - ss_between

    # Degrees of freedom
    df_between = len(anova_result.groups) - 1
    df_within = sum(anova_result.group_ns) - len(anova_result.groups)
    df_total = df_between + df_within

    # Effect sizes
    eta_squared = ss_between / ss_total
    omega_squared = (ss_between - df_between * (ss_within / df_within)) / (ss_total + (ss_within / df_within))

    return eta_squared, omega_squared, ss_between, ss_within, ss_total


def check_anova_assumptions(baseline_df):
    """
    Check ANOVA assumptions: normality and homogeneity of variance.
    """
    assumption_results = {}

    # Get data by groups
    phenotypes = [p for p in baseline_df['phenotype'].unique()
                  if not pd.isna(p) and p in PHENOTYPE_LABELS]

    group_data = []
    group_names = []

    for phenotype in phenotypes:
        data = baseline_df[baseline_df['phenotype'] == phenotype]['baseline_g'].dropna()
        if len(data) >= 3:  # Need at least 3 for Shapiro-Wilk
            group_data.append(data.values)
            group_names.append(PHENOTYPE_LABELS[phenotype])

    # Test homogeneity of variance (Levene's test)
    if len(group_data) >= 2:
        levene_stat, levene_p = levene(*group_data)
        assumption_results['levene_statistic'] = levene_stat
        assumption_results['levene_p_value'] = levene_p
        assumption_results['homogeneity_assumption'] = "Met" if levene_p > 0.05 else "Violated"

    # Test normality for each group (Shapiro-Wilk)
    normality_results = {}
    for i, data in enumerate(group_data):
        if len(data) >= 3 and len(data) <= 5000:  # Shapiro-Wilk limits
            shapiro_stat, shapiro_p = shapiro(data)
            normality_results[group_names[i]] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'normal': shapiro_p > 0.05
            }

    assumption_results['normality_by_group'] = normality_results

    # Overall normality assessment
    all_normal = all(result['normal'] for result in normality_results.values())
    assumption_results['overall_normality'] = "Met" if all_normal else "Violated"

    return assumption_results


def calculate_mean_differences_with_ci(baseline_df, phenotypes, alpha=0.05):
    """
    Calculate mean differences with confidence intervals for post-hoc comparisons.
    """
    results = []

    for phenotype1, phenotype2 in itertools.combinations(phenotypes, 2):
        group1_data = baseline_df[baseline_df['phenotype'] == phenotype1]['baseline_g'].dropna()
        group2_data = baseline_df[baseline_df['phenotype'] == phenotype2]['baseline_g'].dropna()

        if len(group1_data) >= 2 and len(group2_data) >= 2:
            # Calculate mean difference
            mean_diff = np.mean(group1_data) - np.mean(group2_data)

            # Calculate pooled standard error
            n1, n2 = len(group1_data), len(group2_data)
            s1, s2 = np.std(group1_data, ddof=1), np.std(group2_data, ddof=1)

            # Pooled standard error for independent samples
            pooled_se = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)

            # Degrees of freedom (Welch-Satterthwaite equation)
            df = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2 / ((s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1))

            # Critical t-value
            t_critical = stats.t.ppf(1 - alpha / 2, df)

            # Confidence interval
            ci_lower = mean_diff - t_critical * pooled_se
            ci_upper = mean_diff + t_critical * pooled_se

            results.append({
                'group1': PHENOTYPE_LABELS[phenotype1],
                'group2': PHENOTYPE_LABELS[phenotype2],
                'mean_difference': mean_diff,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'se': pooled_se
            })

    return results

def format_p_value(p_value):
    """Format p-values appropriately."""
    if p_value < 0.001:
        return "< .001"
    elif p_value < 0.01:
        return f"= {p_value:.3f}"
    else:
        return f"= {p_value:.3f}"

def save_anova_results(anova_result, posthoc_results, baseline_df, output_dir):
    """
    Save ANOVA and post-hoc results to CSV files.
    """
    # Save ANOVA summary
    anova_summary = pd.DataFrame([{
        'Test': 'One-way ANOVA',
        'F_statistic': anova_result.f_statistic,
        'df_between': len(anova_result.groups) - 1,
        'df_within': sum(anova_result.group_ns) - len(anova_result.groups),
        'P_value': anova_result.p_value,
        'Significant': 'Yes' if anova_result.significant else 'No',
        'Groups_tested': ', '.join(anova_result.groups)
    }])

    anova_path = os.path.join(output_dir, "anova_results.csv")
    anova_summary.to_csv(anova_path, index=False)

    # Save post-hoc results
    posthoc_data = []
    for result in posthoc_results:
        posthoc_data.append({
            'Comparison': f"{result.group1_name} vs {result.group2_name}",
            'Group1_Name': result.group1_name,
            'Group1_N': result.group1_n,
            'Group1_Mean': result.group1_mean,
            'Group1_SD': result.group1_std,
            'Group2_Name': result.group2_name,
            'Group2_N': result.group2_n,
            'Group2_Mean': result.group2_mean,
            'Group2_SD': result.group2_std,
            'Hedges_g': result.hedges_g,
            'CI_Lower': result.ci_lower,
            'CI_Upper': result.ci_upper,
            'P_Value_Uncorrected': result.p_value_uncorrected,
            'P_Value_Corrected': result.p_value_corrected,
            'Significant_Corrected': 'Yes' if result.significant_corrected else 'No',
            'Correction_Method': result.correction_method,
            'Effect_Size_Interpretation': interpret_effect_size(abs(result.hedges_g))
        })

    posthoc_df = pd.DataFrame(posthoc_data)
    posthoc_path = os.path.join(output_dir, "posthoc_results.csv")
    posthoc_df.to_csv(posthoc_path, index=False)

    logger.info(f"ANOVA results saved to {anova_path}")
    logger.info(f"Post-hoc results saved to {posthoc_path}")


def enhanced_anova_summary(anova_result, posthoc_results, baseline_df):
    """Print enhanced ANOVA summary with all required statistics."""

    # Calculate effect sizes
    eta_squared, omega_squared, ss_between, ss_within, ss_total = calculate_anova_effect_size(baseline_df, anova_result)

    # Check assumptions
    assumptions = check_anova_assumptions(baseline_df)

    # Calculate mean differences
    phenotypes = [p for p in baseline_df['phenotype'].unique()
                  if not pd.isna(p) and p in PHENOTYPE_LABELS]
    mean_diffs = calculate_mean_differences_with_ci(baseline_df, sorted(phenotypes))

    print("\n" + "=" * 80)
    print("ENHANCED BASELINE G-FACTOR ANOVA ANALYSIS")
    print("=" * 80)

    # ANOVA results with effect sizes
    df_between = len(anova_result.groups) - 1
    df_within = sum(anova_result.group_ns) - len(anova_result.groups)

    print(f"One-way ANOVA: F({df_between}, {df_within}) = {anova_result.f_statistic:.3f}, "
          f"p {format_p_value(anova_result.p_value)}")
    print(f"Effect size: η² = {eta_squared:.3f}, ω² = {omega_squared:.3f}")
    print()

    # Assumption testing results
    print("Assumption Testing:")
    print(f"  Homogeneity of variance (Levene): F = {assumptions['levene_statistic']:.3f}, "
          f"p {format_p_value(assumptions['levene_p_value'])} ({assumptions['homogeneity_assumption']})")
    print(f"  Normality assumption: {assumptions['overall_normality']}")
    print()

    # Group descriptives
    print("Group Descriptives:")
    for i, group in enumerate(anova_result.groups):
        print(f"  {group}: n = {anova_result.group_ns[i]}, "
              f"M = {anova_result.group_means[i]:.3f}, SD = {anova_result.group_stds[i]:.3f}")
    print()

    # Post-hoc results with mean differences
    if anova_result.significant and posthoc_results:
        correction_method = posthoc_results[0].correction_method
        print(f"Post-hoc comparisons ({correction_method}):")

        for result in posthoc_results:
            # Find corresponding mean difference
            mean_diff_result = None
            for md in mean_diffs:
                if ((md['group1'] == result.group1_name and md['group2'] == result.group2_name) or
                        (md['group1'] == result.group2_name and md['group2'] == result.group1_name)):
                    mean_diff_result = md
                    break

            significance_marker = ('***' if result.p_value_corrected < 0.001 else
                                   '**' if result.p_value_corrected < 0.01 else
                                   '*' if result.p_value_corrected < 0.05 else '')

            print(f"  {result.group1_name} vs {result.group2_name}:")
            print(f"    Mean difference = {result.group1_mean - result.group2_mean:.3f} "
                  f"[{mean_diff_result['ci_lower']:.3f}, {mean_diff_result['ci_upper']:.3f}]" if mean_diff_result else "")
            print(f"    Hedges' g = {result.hedges_g:.3f} [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
            print(f"    p {format_p_value(result.p_value_corrected)} {significance_marker}")
            print()

    return {
        'eta_squared': eta_squared,
        'omega_squared': omega_squared,
        'assumptions': assumptions,
        'mean_differences': mean_diffs
    }


def print_anova_summary(anova_result, posthoc_results):
    """Print formatted summary of ANOVA and post-hoc results."""
    print("\n" + "=" * 80)
    print("BASELINE G-FACTOR ANOVA ANALYSIS RESULTS")
    print("=" * 80)

    # ANOVA results
    df_between = len(anova_result.groups) - 1
    df_within = sum(anova_result.group_ns) - len(anova_result.groups)

    print(f"One-way ANOVA: F({df_between}, {df_within}) = {anova_result.f_statistic:.3f}, "
          f"p = {anova_result.p_value:.4f} {'***' if anova_result.p_value < 0.001 else '**' if anova_result.p_value < 0.01 else '*' if anova_result.p_value < 0.05 else ''}")
    print()

    # Group descriptives
    print("Group Descriptives:")
    for i, group in enumerate(anova_result.groups):
        print(f"  {group}: n = {anova_result.group_ns[i]}, "
              f"M = {anova_result.group_means[i]:.3f}, SD = {anova_result.group_stds[i]:.3f}")
    print()

    # Post-hoc results
    if anova_result.significant and posthoc_results:
        correction_method = posthoc_results[0].correction_method
        print(f"Post-hoc comparisons ({correction_method}):")
        for result in posthoc_results:
            significance_marker = '***' if result.p_value_corrected < 0.001 else '**' if result.p_value_corrected < 0.01 else '*' if result.p_value_corrected < 0.05 else ''

            print(f"  {result.group1_name} vs {result.group2_name}:")
            print(f"    Hedges' g = {result.hedges_g:.3f} [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
            print(f"    p = {result.p_value_corrected:.4f} {significance_marker}")
            print(f"    Effect size: {interpret_effect_size(abs(result.hedges_g))}")
            print()
    else:
        print("ANOVA was not significant - post-hoc tests not performed.")

def main():
    """Main function to run the baseline Hedges' g analysis."""
    try:
        logger.info("Starting baseline g-factor Hedges' g analysis...")

        # Load phenotype data
        phenotype_map, raw_phenotype_map = load_ndar_phenotypes(NDAR_FILE)

        # Load g-factor data
        g_factor_df = load_g_factor_data(G_FACTOR_FILE)
        if g_factor_df.empty:
            logger.error("No valid g-factor data loaded. Exiting.")
            return

        # Add phenotype information
        g_factor_with_phenotype = add_phenotype_to_g_factor_data(g_factor_df, phenotype_map)

        # Extract baseline g-scores
        baseline_df = extract_baseline_g_scores(g_factor_with_phenotype)

        # Perform pairwise comparisons
        results = perform_pairwise_comparisons(baseline_df)

        # Print results summary
        print_results_summary(results)

        # Perform ANOVA
        anova_result = perform_anova_analysis(baseline_df)
        if anova_result is None:
            logger.error("ANOVA analysis failed. Exiting.")
            return

        # Perform post-hoc tests if ANOVA is significant
        posthoc_results = []
        if anova_result.significant:
            # Choose method: 'tukey' or 'bonferroni'
            method = 'tukey' if STATSMODELS_AVAILABLE else 'bonferroni'
            posthoc_results = perform_posthoc_analysis(baseline_df, method=method)

        # Print results
        print_anova_summary(anova_result, posthoc_results)
        # Print enhanced results
        enhanced_anova_summary(anova_result, posthoc_results, baseline_df)

        # Save results to CSV
        output_dir = os.path.join(os.path.dirname(G_FACTOR_FILE), "baseline_analysis")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "baseline_hedges_g_results.csv")
        save_results_to_csv(results, baseline_df, output_path)
        save_anova_results(anova_result, posthoc_results, baseline_df, output_dir)
        logger.info("Baseline Hedges' g analysis completed successfully")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()

