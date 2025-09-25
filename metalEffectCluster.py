import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans

# Load data
data = pd.read_csv('DatasetFileName.csv')
data.rename(columns={'All metals': 'All Metals'}, inplace=True)

# Drop non-numeric columns
data_numeric = data.drop(columns=['All Metals', 'filename'])

# Run KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(data_numeric)
data['KMeans_Cluster'] = kmeans_labels

# Process 'All Metals' column
metal_counts_per_cluster = {i: Counter() for i in range(3)}

for _, row in data.iterrows():
    metals = row['All Metals'].split(',')  # Split metals
    cluster = row['KMeans_Cluster']
    metal_counts_per_cluster[cluster].update(metals)

# Convert metal counts to DataFrame
metal_freq_df = pd.DataFrame(metal_counts_per_cluster).fillna(0)
metal_freq_df.columns = [f'Cluster {i}' for i in range(3)]

# Select top 20 metals with highest total frequency
top_20_metals = metal_freq_df.sum(axis=1).sort_values(ascending=False).head(20)
metal_freq_df = metal_freq_df.loc[top_20_metals.index]

# Calculate average Adsorption for each metal
metal_adsorption = {}
for _, row in data.iterrows():
    metals = row['All Metals'].split(',')
    adsorption = row['Adsorption']
    for metal in metals:
        if metal not in metal_adsorption:
            metal_adsorption[metal] = []
        metal_adsorption[metal].append(adsorption)

# Compute average adsorption per metal
metal_adsorption_avg = {metal: np.mean(values) for metal, values in metal_adsorption.items()}

# Select top 10 metals with highest average adsorption
top_10_metals = sorted(metal_adsorption_avg.items(), key=lambda x: x[1], reverse=True)[:10]
top_10_metals_df = pd.DataFrame(top_10_metals, columns=['Metal', 'Avg Adsorption'])

# Count occurrences of each metal in the dataset
metal_counts = {metal: len(values) for metal, values in metal_adsorption.items()}

# Add count to top_10_metals_df
top_10_metals_df['Count'] = top_10_metals_df['Metal'].map(metal_counts)

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

# Plot both charts in a single figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))  # Increase figure width

# Chart 1: Frequency distribution of top 20 metals in each cluster
metal_freq_df.plot(kind='bar', ax=ax1, colormap='viridis', width=0.8)
ax1.set_xlabel('Metals', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_title('')
ax1.legend(title='Clusters', labels=['0', '1', '2'])
ax1.tick_params(axis='x', rotation=0)
ax1.text(-0.1, 1.05, 'a', transform=ax1.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

# Chart 2: Top 10 metals with highest average adsorption
bars = ax2.barh(top_10_metals_df['Metal'], top_10_metals_df['Avg Adsorption'], color='gray')

# Add text n = number of metal inside bars
for bar, count in zip(bars, top_10_metals_df['Count']):
    width = bar.get_width()
    ax2.text(width / 2, bar.get_y() + bar.get_height() / 2, f'n={count}', 
             va='center', ha='center', color='white', fontweight='bold')

# Add legend with frame (no title, no color bar)
legend_labels = ['n = number of the metal']
legend = ax2.legend(legend_labels, frameon=True, loc='lower right', handlelength=0, handletextpad=0)
legend.get_frame().set_edgecolor('gray')  # Add frame to legend

# Expand x-axis to prevent legend overlapping bars
ax2.set_xlim(right=max(top_10_metals_df['Avg Adsorption']) * 1.2)  # Increase x-axis by 20%

# Remaining settings
ax2.set_xlabel('Average Adsorption', fontweight='bold')
ax2.set_ylabel('Metal', fontweight='bold')
ax2.set_title('')
ax2.invert_yaxis()
ax2.text(-0.1, 1.05, 'b', transform=ax2.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

# Adjust spacing between the two plots
fig.subplots_adjust(wspace=0.3)

# Show and save the figure
plt.savefig('Metal_effect.svg', dpi=1200, bbox_inches='tight')
plt.show()
