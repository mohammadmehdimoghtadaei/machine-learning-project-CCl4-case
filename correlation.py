import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor  # Import CatBoost

# Assume your dataset has already been loaded
data = pd.read_csv('DatasetFileName.csv')

# Define X and y variables
X = data.drop('Adsorption', axis=1, inplace=False)
y = data['Adsorption']

# Modeling with CatBoost to assess feature importance
model = CatBoostRegressor(        verbose=0, iterations=300,
        depth=8, learning_rate=0.05, colsample_bylevel=0.5, l2_leaf_reg=5, bagging_temperature=0.5,
        min_child_samples=10, subsample=0.7,
        random_state=42)  # CatBoost model
model.fit(X, y)

# Calculate feature importances
importances = model.get_feature_importance()
indices = np.argsort(importances)[::-1]

# Calculate percentage contribution of each feature
total_importance = np.sum(importances)
percentages = (importances / total_importance) * 100

# Sort features by percentage (descending order)
sorted_indices = np.argsort(percentages)[::-1]
sorted_features = X.columns[sorted_indices]
sorted_percentages = percentages[sorted_indices]

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Create pie chart
plt.figure(figsize=(12, 12))  # Increase figure size
colors = plt.cm.viridis(np.linspace(0, 1, len(X.columns)))  # Define colors

# Create pie chart with smaller radius
wedges, _ = plt.pie(importances[sorted_indices], startangle=140, colors=colors, labels=[None]*len(X.columns), radius=0.4)

# Prepare legend labels with percentages and proper formatting for powers
labels_with_percentages = []
for feature, percentage in zip(sorted_features, sorted_percentages):
    # Format m^2 and cm^3 in feature names
    if 'm2' in feature:
        feature = feature.replace('m2', r'm$^2$')
    if 'cm3' in feature:
        feature = feature.replace('cm3', r'cm$^3$')
    labels_with_percentages.append(f"{feature}: {percentage:.2f}%")

# Add legend with bold title and larger font size
legend = plt.legend(wedges, labels_with_percentages, title="**Features**", loc="center left", bbox_to_anchor=(1, 0.6), 
                    fontsize=19, title_fontsize='22', frameon=True, shadow=True)
legend.set_title("Features", prop={'size': 22, 'weight': 'bold'})

plt.title(' ', fontsize=16, weight='bold')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Add labels 'a' and 'b' to the chart
plt.text(-0.1, 1.1, 'b', fontsize=20, fontfamily='Times New Roman', weight='bold', transform=plt.gca().transAxes)
# You can adjust coordinates if needed

# Save the chart with high quality
plt.savefig('feature_importance_pie_chart.svg', dpi=1200, bbox_inches='tight', pad_inches=0.1)

# Display the chart with high quality
plt.show()

# Prepare column and index names for heatmap
corr_matrix = data.corr()

# Format m^2 and cm^3 in heatmap labels
corr_matrix.columns = [col.replace('m2', r'm$^2$').replace('cm3', r'cm$^3$') for col in corr_matrix.columns]
corr_matrix.index = [idx.replace('m2', r'm$^2$').replace('cm3', r'cm$^3$') for idx in corr_matrix.index]

# --- Plot feature correlation heatmap ---
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap='YlOrRd', annot=False)  # Adjustable color map

# Title and axis font settings
plt.title(' ', weight='bold', fontsize=16)
plt.yticks(fontsize=9.5, weight='bold')
plt.xticks(rotation=55, ha='right', fontsize=8.5, weight='bold')

# Add label 'a' to the heatmap
plt.text(-0.1, 1.05, 'a', fontsize=20, fontfamily='Times New Roman', weight='bold', transform=plt.gca().transAxes)

# Adjust margins to prevent labels from being cut off
plt.tight_layout()

# Save heatmap with high quality
plt.savefig('heatmap_feature_correlations.svg', dpi=900, bbox_inches='tight')
# Display the plot
plt.show()