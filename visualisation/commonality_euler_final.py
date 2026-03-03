import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_set_diagrams import EulerDiagram
import numpy as np

# Configuration
CLEAN_DIAGRAM = True  # Set to False to show all labels and percentages
MIN_THRESHOLD = 0  # Minimum percentage threshold to include

# Load data
df = pd.read_csv('m5_neuro_age_adjusted.csv', index_col=0)

# Filter out the Total row
df = df[df['Component'] != 'Total']

# Define the order of sets for the boolean tuples
# Order: stacked_pred_z_dev, stacked_pred_z_mean, age_dev, age_mean
set_order = ['stacked_pred_z_dev', 'stacked_pred_z_mean', 'age_deviation', 'age_mean']
set_labels = [
    'Within-person\ntrajectories',
    'Between-person\nmeans',
    'Age\nwithin-person',
    'Age\nbetween-person'
]

# Create mapping from component names to boolean tuples
def component_to_tuple(component_str, set_order):
    # Initialize all False
    result = [False] * len(set_order)
    
    # Check which components are mentioned
    for i, set_name in enumerate(set_order):
        if set_name in component_str:
            result[i] = True
    
    return tuple(result)

# Build the subset_sizes dictionary
subset_sizes = {}

for idx, row in df.iterrows():
    component = row['Component'].strip()
    variance = row['Total_var_adjusted_perc']
    
    # Skip if below threshold
    if variance < MIN_THRESHOLD:
        continue
    
    # Convert component to boolean tuple
    bool_tuple = component_to_tuple(component, set_order)
    
    # Only add if at least one set is included
    if any(bool_tuple):
        subset_sizes[bool_tuple] = variance

print("Subset sizes for Euler diagram:")
for key, value in sorted(subset_sizes.items()):
    set_names = [set_order[i] for i, included in enumerate(key) if included]
    print(f"  {key} ({', '.join(set_names)}): {value:.2f}%")

# Calculate total variance for each set
totals = {}
for i, set_name in enumerate(set_order):
    total = 0
    for bool_tuple, variance in subset_sizes.items():
        if bool_tuple[i]:  # If this set is included in this subset
            total += variance
    totals[set_name] = total

print("\nTotal variance explained by each set:")
for set_name, total in zip(set_order, totals.values()):
    print(f"  {set_name}: {total:.2f}%")

def create_clean_euler(subset_sizes, set_labels, ax, clean=True):
    """
    Create Euler diagram and optionally remove all labels
    """
    # Create the diagram
    if clean:
        # Pass None for labels and empty formatter
        euler_obj = EulerDiagram(
            subset_sizes=subset_sizes,
            set_labels=None,
            subset_label_formatter=lambda x, y: "",
            ax=ax
        )
    else:
        euler_obj = EulerDiagram(
            subset_sizes=subset_sizes,
            set_labels=set_labels,
            ax=ax
        )
    
    # Customize appearance of circles
    for artist in euler_obj.subset_artists.values():
        artist.set_alpha(0.5)
        artist.set_linewidth(2.5)
        artist.set_edgecolor('black')
    
    # If clean mode, remove all text labels
    if clean:
        # Remove set labels (the circle names) - it's a list
        if hasattr(euler_obj, 'set_label_artists'):
            for artist in euler_obj.set_label_artists:
                artist.set_visible(False)
        
        # Remove subset labels (the percentages in overlaps) - it's a dict
        if hasattr(euler_obj, 'subset_label_artists'):
            for artist in euler_obj.subset_label_artists.values():
                artist.set_visible(False)
    
    return euler_obj

# Create individual versions
for threshold, suffix in [(0, 'all'), (1, 'filtered')]:
    for clean_mode in [True, False]:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        if threshold == 0:
            data = subset_sizes
        else:
            data = {k: v for k, v in subset_sizes.items() if v >= threshold}
        
        try:
            clean_suffix = "_clean" if clean_mode else ""
            euler_obj = create_clean_euler(data, set_labels, ax, clean=clean_mode)
            
            threshold_text = f" (>{threshold}%)" if threshold > 0 else " (All non-zero)"
            clean_text = " - Clean" if clean_mode else ""
            ax.set_title(f'Commonality Analysis: Area-Proportional Euler Diagram{threshold_text}{clean_text}',
                        fontsize=14, weight='bold', pad=20)
            
            if not clean_mode:
                fig.text(0.5, 0.02, 
                        'Circle sizes are proportional to total variance explained.\n'
                        'Overlapping regions show shared variance between components.',
                        ha='center', fontsize=10, style='italic', color='gray')
            
            plt.tight_layout(rect=[0, 0.05 if not clean_mode else 0, 1, 1])
            
            output_filename = f'commonality_euler_{suffix}{clean_suffix}.png'
            plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved: {output_filename}")
            plt.close()
            
        except Exception as e:
            clean_suffix = "_clean" if clean_mode else ""
            print(f"Error creating {suffix}{clean_suffix} version: {e}")
            plt.close()

print("\nVisualization complete!")
print(f"\nCLEAN_DIAGRAM mode: {CLEAN_DIAGRAM}")
print(f"MIN_THRESHOLD: {MIN_THRESHOLD}%")
print("\nTotal variance in model: 84.80%")
print("Components by total variance explained:")
for set_name in set_order:
    print(f"  {set_name}: {totals[set_name]:.2f}%")
