import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'figure.titlesize': 24,
    'figure.dpi': 300
})


def calculate_ci(x, y, new_x, confidence=0.95):
    """Calculate confidence interval for regression line"""
    n = len(x)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    pred_y = slope * new_x + intercept

    # Calculate standard error of prediction
    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean) ** 2)
    sxy = np.sum((x - x_mean) * (y - np.mean(y)))
    syy = np.sum((y - np.mean(y)) ** 2)

    s_yx = np.sqrt((syy - slope * sxy) / (n - 2))
    se_pred = s_yx * np.sqrt(1 / n + (new_x - x_mean) ** 2 / sxx)

    t_val = stats.t.ppf((1 + confidence) / 2, n - 2)
    ci = t_val * se_pred

    return pred_y - ci, pred_y + ci


def create_correlation_plot(ax, x_data, y_data, x_label, y_label, title, pearson_r, spearman_rho, sample_size,
                            y_limits=None, y_ticks=None):
    """Create a standardized correlation plot without statistics text box"""
    # Scatterplot
    ax.scatter(x_data, y_data, alpha=0.6, s=30, color='steelblue', edgecolors='none')

    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
    line_x = np.linspace(x_data.min(), x_data.max(), 100)
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, color='red', linewidth=2, alpha=0.8)

    # Add confidence interval
    ci_lower, ci_upper = calculate_ci(x_data, y_data, line_x)
    ax.fill_between(line_x, ci_lower, ci_upper, alpha=0.2, color='red')

    # Labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Set y-axis limits if specified
    if y_limits is not None:
        ax.set_ylim(y_limits)

    # Set y-axis ticks if specified
    if y_ticks is not None:
        ax.set_yticks(y_ticks)

    # Clean box outline
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1)


# Load the data
print("Loading data...")
deviation_data = pd.read_csv('deviation_scores_for_python.csv')
deviation_stats = pd.read_csv('correlation_stats.csv')
means_data = pd.read_csv('participant_means_for_python.csv')
means_stats = pd.read_csv('between_person_correlation_stats.csv')

# Extract within-person statistics
within_pearson_r = deviation_stats[deviation_stats['metric'] == 'pearson_r']['value'].iloc[0]
within_spearman_rho = deviation_stats[deviation_stats['metric'] == 'spearman_rho']['value'].iloc[0]
within_sample_size = int(deviation_stats[deviation_stats['metric'] == 'sample_size']['value'].iloc[0])

# Extract between-person statistics
between_pearson_r = means_stats[means_stats['metric'] == 'pearson_r']['value'].iloc[0]
between_spearman_rho = means_stats[means_stats['metric'] == 'spearman_rho']['value'].iloc[0]
between_sample_size = int(means_stats[means_stats['metric'] == 'sample_size']['value'].iloc[0])

print(f"Within-person: n = {within_sample_size:,} observations")
print(f"Between-person: n = {between_sample_size:,} participants")

# 1. Individual within-person plot
print("Creating within-person plot...")
fig1, ax1 = plt.subplots(figsize=(10, 8))
create_correlation_plot(ax1,
                        deviation_data['stacked_pred_z_dev'],
                        deviation_data['true_g_z_dev'],
                        'Predicted g',
                        'Observed g',
                        'Intraindividual cognition (deviations)',
                        within_pearson_r, within_spearman_rho, within_sample_size,
                        y_limits=(-2, 2), y_ticks=[-2, -1, 0, 1, 2])
plt.tight_layout()
plt.savefig('within_person_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Individual between-person plot
print("Creating between-person plot...")
fig2, ax2 = plt.subplots(figsize=(10, 8))
create_correlation_plot(ax2,
                        means_data['stacked_pred_z_mean'],
                        means_data['true_g_z_mean'],
                        'Predicted g',
                        'Observed g',
                        'Interindividual cognition (means)',
                        between_pearson_r, between_spearman_rho, between_sample_size
                        )
plt.tight_layout()
plt.savefig('between_person_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Side-by-side comparison
print("Creating side-by-side comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Within-person plot (left)
create_correlation_plot(ax1,
                        deviation_data['stacked_pred_z_dev'],
                        deviation_data['true_g_z_dev'],
                        'Predicted g',
                        'Observed g',
                        'Within-Person Relationship',
                        within_pearson_r, within_spearman_rho, within_sample_size,
                        y_limits=(-2, 2), y_ticks=[-2, -1, 0, 1, 2])

# Between-person plot (right)
create_correlation_plot(ax2,
                        means_data['stacked_pred_z_mean'],
                        means_data['true_g_z_mean'],
                        'Predicted g',
                        'Observed g',
                        'Between-Person Relationship',
                        between_pearson_r, between_spearman_rho, between_sample_size)

# Add panel labels
ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, fontsize=24, fontweight='bold')
ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, fontsize=24, fontweight='bold')

plt.tight_layout()
plt.savefig('correlation_comparison_sidebyside.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print(f"\n=== SUMMARY STATISTICS ===")
print(f"WITHIN-PERSON (deviation scores):")
print(f"  Sample size: {within_sample_size:,} observations")
print(f"  Pearson r = {within_pearson_r:.4f}")
print(f"  Spearman rho = {within_spearman_rho:.4f}")
print(f"\nBETWEEN-PERSON (participant means):")
print(f"  Sample size: {between_sample_size:,} participants")
print(f"  Pearson r = {between_pearson_r:.4f}")
print(f"  Spearman rho = {between_spearman_rho:.4f}")