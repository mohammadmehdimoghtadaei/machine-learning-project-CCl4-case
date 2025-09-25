import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('DatasetFileName.csv')

# Rename column 'All metals' to 'All Metals'
data.rename(columns={'All metals': 'All Metals'}, inplace=True)

# Drop non-numeric columns
data_numeric = data.drop(columns=['All Metals', 'filename'])

# 📌 Remove outliers using IQR method based on numeric columns
Q1 = data_numeric.quantile(0.25)
Q3 = data_numeric.quantile(0.75)
IQR = Q3 - Q1
mask = ~((data_numeric < (Q1 - 1.5 * IQR)) | (data_numeric > (Q3 + 1.5 * IQR))).any(axis=1)
data_numeric = data_numeric[mask]
data = data[mask]  # Apply mask to original DataFrame

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(data_numeric)

# Add cluster labels to dataset
data['KMeans_Cluster'] = kmeans_labels

# Calculate mean Adsorption and sample counts for each KMeans cluster
kmeans_adsorption_means = data.groupby('KMeans_Cluster')['Adsorption'].mean()
kmeans_sample_counts = data.groupby('KMeans_Cluster').size()

# Sort clusters by mean adsorption (ascending = low to high)
sorted_clusters = kmeans_adsorption_means.sort_values(ascending=True).index
kmeans_adsorption_means = kmeans_adsorption_means.loc[sorted_clusters]
kmeans_sample_counts = kmeans_sample_counts.loc[sorted_clusters]

# Find cluster with highest mean adsorption in KMeans
best_kmeans_cluster = kmeans_adsorption_means.idxmax()
best_kmeans_data = data[data['KMeans_Cluster'] == best_kmeans_cluster]

# Use CatBoost to find most important features in best KMeans cluster
X_kmeans = best_kmeans_data.drop(columns=['Adsorption', 'KMeans_Cluster', 'All Metals', 'filename'])
y_kmeans = best_kmeans_data['Adsorption']

catboost_kmeans = CatBoostRegressor(
    verbose=0, iterations=300,
    depth=8, learning_rate=0.05, colsample_bylevel=0.5,
    l2_leaf_reg=5, bagging_temperature=0.5,
    min_child_samples=10, subsample=0.7,
    random_state=42
)
catboost_kmeans.fit(X_kmeans, y_kmeans)
kmeans_feature_importance = catboost_kmeans.get_feature_importance()

# Convert feature importances to DataFrame
kmeans_feature_importance_df = pd.DataFrame({'Feature': X_kmeans.columns, 'Importance': kmeans_feature_importance})

# Sort feature importances descending (highest importance on top)
kmeans_feature_importance_df_sorted = kmeans_feature_importance_df.sort_values(by='Importance', ascending=True)

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Set font for axis titles and texts
font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 12}
plt.rc('font', **font)

# Use colormap='viridis' to match cluster colors
colors = plt.cm.viridis(np.linspace(0, 1, 3))  # 3 colors for 3 clusters

# First plot: mean adsorption for each cluster (sorted)
x = np.arange(len(kmeans_adsorption_means))
width = 0.35
bars = ax1.bar(x, kmeans_adsorption_means, width, color=colors, label='Mean Adsorption')
ax1.set_xlabel('Cluster', fontdict=font)
ax1.set_ylabel('Mean Adsorption', fontdict=font)
ax1.set_xticks(x)
ax1.set_xticklabels([str(c) for c in sorted_clusters])
ax1.text(-0.1, 1.05, 'a', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

# Write number of samples inside each bar
for i, (mean, count) in enumerate(zip(kmeans_adsorption_means, kmeans_sample_counts)):
    ax1.text(i, mean / 2, f'n = {count}', ha='center', va='center', fontsize=10, color='white', fontweight='normal')

# Add legend for n = number of samples
legend_font = {'family': 'Times New Roman', 'size': 10, 'weight': 'normal'}
ax1.legend(['n = number of samples'], loc='upper left', frameon=True, edgecolor='gray', prop=legend_font, handlelength=0, handletextpad=0)

# Second plot: feature importances (sorted)
bar_color_gmm = '#B0B0B0'
ax2.barh(kmeans_feature_importance_df_sorted['Feature'], kmeans_feature_importance_df_sorted['Importance'], color=bar_color_gmm)
ax2.set_xlabel('Importance', fontdict=font)
ax2.set_ylabel('Feature', fontdict=font)
ax2.text(-0.1, 1.05, 'b', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

# Smaller font for y-axis labels in feature importance plot
ax2.tick_params(axis='y', labelsize=10)

# Space between the two plots
plt.subplots_adjust(wspace=0.5)

# Save figure
plt.savefig('combined_plot.svg', dpi=1200, bbox_inches='tight')
plt.show()
