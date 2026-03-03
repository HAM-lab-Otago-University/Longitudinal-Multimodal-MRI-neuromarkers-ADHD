import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 20,
    'figure.titlesize': 24,
    'figure.dpi': 300
})


def create_spaghetti_plot(ax, data, x_col, y_col, x_label, y_label, title):
    """
    Create a spaghetti plot showing individual participant trajectories.

    Parameters:
    -----------
    ax : matplotlib axis
        The axis to plot on
    data : DataFrame
        The data containing participant_id, x and y variables
    x_col : str
        Column name for x-axis variable
    y_col : str
        Column name for y-axis variable (observed g)
    x_label : str
        Label for x-axis
    y_label : str
        Label for y-axis
    title : str
        Plot title
    """

    # Get unique participants
    participants = data['participant_id'].unique()
    n_participants = len(participants)

    # Create color map for individual trajectories
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    # Track plotted participants for legend
    has_multiple = False
    has_single = False

    # Plot each participant's trajectory
    for i, participant in enumerate(participants):
        participant_data = data[data['participant_id'] == participant].copy()

        # Sort by symptom score to create sensible lines
        participant_data = participant_data.sort_values(x_col)

        # Choose color
        color = colors[i % len(colors)]

        # Determine if participant has multiple timepoints
        n_points = len(participant_data)

        if n_points > 1:
            # Plot line connecting points
            ax.plot(participant_data[x_col], participant_data[y_col],
                    color=color, alpha=0.3, linewidth=1, zorder=1)
            has_multiple = True
        else:
            has_single = True

        # Plot points
        ax.scatter(participant_data[x_col], participant_data[y_col],
                   color=color, alpha=0.6, s=30, zorder=2, edgecolors='none')

    # Add overall regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(data[x_col], data[y_col])
    line_x = np.linspace(data[x_col].min(), data[x_col].max(), 100)
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, color='black', linewidth=3, alpha=0.8,
            label=f'r = {r_value:.3f})', zorder=3)

    # Labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Add legend
    legend_elements = [plt.Line2D([0], [0], color='black', linewidth=3,
                                  label=f'r = {r_value:.3f}')]
    #if has_multiple:
    #    legend_elements.append(plt.Line2D([0], [0], color='gray', alpha=0.3, linewidth=1,
    #                                      label='Individual trajectories'))

    ax.legend(handles=legend_elements, loc='best', framealpha=0.9)

    # Add sample size info
    # stats_text = f'n = {len(data):,} observations\n{n_participants} participants'
    # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
    #        verticalalignment='top', horizontalalignment='left',
    #        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
    #                  edgecolor='black', alpha=0.9))

    # Clean box outline
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1)


def create_decomposed_comparison(data, symptom_col, symptom_mean_col, symptom_dev_col,
                                 symptom_name, output_dir):
    """
    Create a 3-panel comparison showing overall, between-person, and within-person relationships.
    """

    fig, axes = plt.subplots(1, 3, figsize=(30, 8))

    # Panel 1: Overall relationship
    create_spaghetti_plot(
        axes[0], data,
        symptom_col, 'true_g_z',
        f'{symptom_name} (z)', 'Observed g (z)',
        f'Overall: Observed g vs {symptom_name}'
    )

    # Panel 2: Between-person relationship
    create_spaghetti_plot(
        axes[1], data,
        symptom_mean_col, 'true_g_z',
        f'{symptom_name} (between-person)', 'Observed g (z)',
        f'Between-Person: Observed g vs {symptom_name}'
    )

    # Panel 3: Within-person relationship
    create_spaghetti_plot(
        axes[2], data,
        symptom_dev_col, 'true_g_z',
        f'{symptom_name} (within-person)', 'Observed g (z)',
        f'Within-Person: Observed g vs {symptom_name}'
    )

    plt.tight_layout()

    # Save plot
    filename = f'observed_g_vs_{symptom_name.lower()}_decomposed.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def verify_filtering(data, data_name):
    """
    Verify that data has been filtered for participants with multiple timepoints.
    """
    print(f"\n{data_name} Filtering Verification:")
    print("=" * 60)

    # Count timepoints per participant
    timepoint_counts = data.groupby('participant_id').size()

    print(f"Total participants: {len(timepoint_counts)}")
    print(f"Total observations: {len(data)}")
    print(f"\nTimepoints per participant:")
    print(f"  1 timepoint: {sum(timepoint_counts == 1)} participants")
    print(f"  2 timepoints: {sum(timepoint_counts == 2)} participants")
    print(f"  3 timepoints: {sum(timepoint_counts == 3)} participants")
    print(f"  4+ timepoints: {sum(timepoint_counts >= 4)} participants")

    if sum(timepoint_counts == 1) > 0:
        print("\nWARNING: Data contains participants with only 1 timepoint!")
        print("These should have been filtered out in R.")
    else:
        print("\nCONFIRMED: All participants have 2+ timepoints")


# Main execution
if __name__ == "__main__":
    # Create output directory
    output_dir = '/media/hcs-sci-psy-narun/Jack/Linear_mixed_effects_OREGON/symptom_viz_v2'
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    hyp_data = pd.read_csv('hyperactivity_data_for_python.csv')
    inatt_data = pd.read_csv('inattention_data_for_python.csv')

    # Verify filtering
    verify_filtering(hyp_data, "Hyperactivity")
    verify_filtering(inatt_data, "Inattention")

    # Create hyperactivity plots
    print("\n\nCreating hyperactivity plots...")
    print("=" * 60)
    create_decomposed_comparison(
        hyp_data,
        'hyperactivity_z', 'hyperactivity_z_mean', 'hyperactivity_z_dev',
        'Hyperactivity', output_dir
    )

    # Create inattention plots
    print("\nCreating inattention plots...")
    print("=" * 60)
    create_decomposed_comparison(
        inatt_data,
        'inattention_z', 'inattention_z_mean', 'inattention_z_dev',
        'Inattention', output_dir
    )

    # Create individual simple plots (overall relationship only)
    print("\nCreating simple overall plots...")
    print("=" * 60)

    # Hyperactivity simple
    fig, ax = plt.subplots(figsize=(10, 8))
    create_spaghetti_plot(
        ax, hyp_data,
        'hyperactivity_z', 'true_g_z',
        'Hyperactivity Symptoms (z)', 'Observed g (z)',
        'Observed g vs Hyperactivity Symptoms'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'observed_g_vs_hyperactivity_simple.png'),
                dpi=300, bbox_inches='tight')
    print("Saved: observed_g_vs_hyperactivity_simple.png")
    plt.close()

    # Inattention simple
    fig, ax = plt.subplots(figsize=(10, 8))
    create_spaghetti_plot(
        ax, inatt_data,
        'inattention_z', 'true_g_z',
        'Inattention Symptoms (z)', 'Observed g (z)',
        'Observed g vs Inattention Symptoms'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'observed_g_vs_inattention_simple.png'),
                dpi=300, bbox_inches='tight')
    print("Saved: observed_g_vs_inattention_simple.png")
    plt.close()

    print(f"\n\nAll plots saved to: {output_dir}")
    print("\nFiles created:")
    print("  - observed_g_vs_hyperactivity_decomposed.png (3-panel)")
    print("  - observed_g_vs_inattention_decomposed.png (3-panel)")
    print("  - observed_g_vs_hyperactivity_simple.png")
    print("  - observed_g_vs_inattention_simple.png")