import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the decomposition data
df = pd.read_csv('m5_r2mlm_decomposition.csv')

print("Loaded raw r2mlm data:")
print(df)

# CORRECTED mapping based on r2mlm structure:
# Decompositions are ordered as rows in the output matrix:
# Row 1: fixed, within (f1)
# Row 2: fixed, between (f2)
# Row 3: slope variation (v) - will be 0 for random intercepts only
# Row 4: mean variation (m) - random INTERCEPTS variance
# Row 5: sigma2 (residual)

r2_mapping = {
    'Decompositions1': 'f1',        # Fixed within
    'Decompositions2': 'f2',        # Fixed between
    'Decompositions3': 'v',         # Slope variation (should be ~0)
    'Decompositions4': 'm',         # Mean/Intercept variation (NOT slope!)
    'Decompositions5': 'residual',  # Residual variance
}

# Extract values
r2_values = {}
for comp, label in r2_mapping.items():
    value = df[df['component'] == comp]['value'].values
    if len(value) > 0 and not pd.isna(value[0]):
        r2_values[label] = value[0]
    else:
        r2_values[label] = 0

print("\nExtracted R² values:")
for k, v in r2_values.items():
    print(f"{k}: {v:.4f}")

# Verify slope variation is near zero
if r2_values.get('v', 0) > 0.01:
    print("\nWARNING: Slope variation > 0.01, but model has no random slopes!")

# Create the structured dataframe for plotting
# Component order from bottom to top in stack
plot_data = pd.DataFrame({
    'Component': [
        'Fixed slopes (intra)',
        'Fixed intercept (inter)',
        'Slope variation (intra)',
        'Intercept variation (inter)',
        'Residual (intra)'
    ],
    'Total': [
        r2_values.get('f1', 0),       # Fixed within
        r2_values.get('f2', 0),       # Fixed between
        r2_values.get('v', 0),        # Slope variation (should be ~0)
        r2_values.get('m', 0),        # Intercept variation
        r2_values.get('residual', 0)  # Residual
    ],
    'Intraindividual': [
        r2_values.get('f1', 0),       # Fixed slopes contribute to within
        np.nan,                        # Fixed intercepts don't contribute to within
        r2_values.get('v', 0),        # Slope variation contributes to within
        np.nan,                        # Intercept variation doesn't contribute to within
        r2_values.get('residual', 0)  # Residual contributes to within
    ],
    'Interindividual': [
        np.nan,                        # Fixed slopes don't contribute to between
        r2_values.get('f2', 0),       # Fixed intercepts contribute to between
        np.nan,                        # Slope variation doesn't contribute to between
        r2_values.get('m', 0),        # Intercept variation contributes to between
        np.nan                         # Residual doesn't contribute to between
    ]
})

print("\nStructured data for plotting:")
print(plot_data)

# Create the stacked bar chart
fig, ax = plt.subplots(figsize=(10, 8))

# Define colors for each component
colors = {
    'Fixed slopes (intra)': '#C5504B',      # Red
    'Fixed intercept (inter)': '#4F81BD',   # Blue
    'Slope variation (intra)': '#9BBB59',   # Green
    'Intercept variation (inter)': '#8064A2', # Purple
    'Residual (intra)': '#F5F5F5'           # Slightly grey
}

# Columns to plot (the three bars)
columns = ['Total', 'Intraindividual', 'Interindividual']

# Initialize bottom values for stacking
bottoms = {col: 0 for col in columns}

# Plot each component as a stacked bar
for idx, row in plot_data.iterrows():
    component = row['Component']
    color = colors.get(component, 'gray')
    
    # Plot bars for each column where value is not NaN
    for col in columns:
        value = row[col]
        if pd.notna(value) and value > 0.001:  # Only plot if not NaN and > 0.001
            ax.bar(col, value, bottom=bottoms[col], 
                  color=color, alpha=0.8, 
                  edgecolor='black', linewidth=0.5,
                  label=component if col == columns[0] else "")
            bottoms[col] += value

# Customize the plot
ax.set_ylim(0, 1)
ax.set_ylabel('Proportion of Variance', fontsize=12)
ax.set_title('R² Decomposition: Model 5 (Neuroimaging + Age)', fontsize=14, fontweight='bold')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add legend
# Create legend manually to avoid duplicates
legend_elements = []
legend_labels = []
for component, color in colors.items():
    # Only add to legend if component has non-zero contribution
    component_has_data = False
    for col in columns:
        comp_row = plot_data[plot_data['Component'] == component]
        if len(comp_row) > 0:
            val = comp_row[col].values[0]
            if pd.notna(val) and val > 0.001:
                component_has_data = True
                break
    
    if component_has_data:
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, edgecolor='black'))
        legend_labels.append(component)

ax.legend(legend_elements, legend_labels, 
         loc='center left', bbox_to_anchor=(1, 0.5),
         frameon=False)

# Add grid
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()

# Save to current directory
plt.savefig('m5_r2_decomposition_corrected.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'm5_r2_decomposition_corrected.png'")

# Print summary
print("\nSummary of bars:")
for col in columns:
    print(f"{col}: {bottoms[col]:.3f} total ({bottoms[col]*100:.1f}%)")
    
print("\nComponent breakdown:")
for idx, row in plot_data.iterrows():
    component = row['Component']
    has_contribution = False
    for col in columns:
        value = row[col]
        if pd.notna(value) and value > 0.001:
            has_contribution = True
            break
    
    if has_contribution:
        print(f"\n{component}:")
        for col in columns:
            value = row[col]
            if pd.notna(value) and value > 0.001:
                print(f"  {col}: {value:.3f} ({value*100:.1f}%)")
