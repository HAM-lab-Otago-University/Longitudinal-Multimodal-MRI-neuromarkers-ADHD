import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend to avoid PyCharm issues
import matplotlib.pyplot as plt
import numpy as np

# Font size configuration - easily adjustable
TITLE_FONT_SIZE = 24
AXIS_LABEL_FONT_SIZE = 22
Y_TICK_LABEL_FONT_SIZE = 20
X_TICK_LABEL_FONT_SIZE = 28
LEGEND_FONT_SIZE = 18

# Load and process data
df = pd.read_csv('stacked_ridge_feature_importance.csv')

# Average haufe_importance across folds for each feature
avg_importance = df.groupby('feature')['haufe_importance'].mean().reset_index()

# Function to shorten feature names
def shorten_feature_name(feature):
    # Apply all the shortening rules
    feature = feature.replace('xgboost', 'xgb')
    feature = feature.replace('functional_connectivity', 'func_connect')
    feature = feature.replace('cortical', 'cort')
    return feature

# Apply name shortening
avg_importance['feature_short'] = avg_importance['feature'].apply(shorten_feature_name)

# Categorize features for coloring
def categorize_feature(feature):
    if any(x in feature for x in ['func_connect', 'alff', 'reho']):
        return 'Functional'
    elif any(x in feature for x in ['cort_area', 'cort_thickness']):
        return 'Structural (Cortical)'
    elif any(x in feature for x in ['subcort_volume', 'total_brain_volume']):
        return 'Structural (Subcortical)'
    else:
        return 'Other'

avg_importance['category'] = avg_importance['feature_short'].apply(categorize_feature)

# Sort by importance (descending)
avg_importance = avg_importance.sort_values('haufe_importance', ascending=True)

# Define colors for each category
color_map = {
    'Functional': '#90EE90',  # Light green
    'Structural (Cortical)': '#FFA07A',  # Light salmon/red
    'Structural (Subcortical)': '#87CEEB'  # Sky blue
}

# Create the plot
fig, ax = plt.subplots(figsize=(12, 10))

# Create horizontal bar chart
bars = ax.barh(range(len(avg_importance)),
               avg_importance['haufe_importance'],
               color=[color_map[cat] for cat in avg_importance['category']])

# Customize the plot
ax.set_yticks(range(len(avg_importance)))
ax.set_yticklabels(avg_importance['feature_short'], fontsize=Y_TICK_LABEL_FONT_SIZE)
ax.tick_params(axis='x', labelsize=X_TICK_LABEL_FONT_SIZE)
ax.set_xlabel('Average Haufe Importance (|correlation|)', fontsize=AXIS_LABEL_FONT_SIZE)
ax.set_title('Stacked Ridge Model Feature Haufe Importance',
             fontsize=TITLE_FONT_SIZE, pad=20)

# Add grid for better readability
ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Create legend
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[cat], label=cat)
                  for cat in color_map.keys()]
ax.legend(handles=legend_elements, loc='lower right', fontsize=LEGEND_FONT_SIZE)

# Set x-axis limits with some padding
max_val = avg_importance['haufe_importance'].max()
ax.set_xlim(0, max_val * 1.05)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure (optional)
# plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

# Print summary statistics
print("\nFeature Importance Summary:")
print(f"Total features: {len(avg_importance)}")
print(f"Top 5 most important features:")
top_5 = avg_importance.nlargest(5, 'haufe_importance')
for _, row in top_5.iterrows():
    print(f"  {row['feature_short']}: {row['haufe_importance']:.3f}")