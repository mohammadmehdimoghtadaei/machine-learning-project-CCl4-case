import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt


data = pd.read_csv('filename.csv')


data.rename(columns={'All metals': 'All Metals'}, inplace=True)

data_numeric = data.drop(columns=['All Metals', 'filename'])


kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(data_numeric)


data['KMeans_Cluster'] = kmeans_labels


kmeans_adsorption_means = data.groupby('KMeans_Cluster')['Adsorption'].mean().sort_values()
kmeans_sample_counts = data.groupby('KMeans_Cluster').size().reindex(kmeans_adsorption_means.index)


best_kmeans_cluster = kmeans_adsorption_means.idxmax()
best_kmeans_data = data[data['KMeans_Cluster'] == best_kmeans_cluster]


X_kmeans = best_kmeans_data.drop(columns=['Adsorption', 'KMeans_Cluster', 'All Metals', 'filename'])
y_kmeans = best_kmeans_data['Adsorption']

catboost_kmeans = CatBoostRegressor(        verbose=0, iterations=300,
        depth=8, learning_rate=0.05, colsample_bylevel=0.5, l2_leaf_reg=5, bagging_temperature=0.5,
        min_child_samples=10, subsample=0.7,
        random_state=42)
catboost_kmeans.fit(X_kmeans, y_kmeans)
kmeans_feature_importance = catboost_kmeans.get_feature_importance()


kmeans_feature_importance_df = pd.DataFrame({'Feature': X_kmeans.columns, 'Importance': kmeans_feature_importance})
kmeans_feature_importance_df_sorted = kmeans_feature_importance_df.sort_values(by='Importance', ascending=True)

# قالب دو ساب‌پلات کنار هم
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.77, 4.5))  # double-column مناسب


plt.rcParams['font.family'] = 'Times New Roman'


colors = plt.cm.viridis(np.linspace(0, 1, 3))


x = np.arange(len(kmeans_adsorption_means))
width = 0.35
bars = ax1.bar(x, kmeans_adsorption_means.values, width, color=colors)

ax1.set_xlabel('Cluster', fontsize=7)
ax1.set_ylabel('Mean Adsorption', fontsize=7)
ax1.set_xticks(x)
ax1.set_xticklabels(kmeans_adsorption_means.index.astype(str), fontsize=8)
ax1.tick_params(axis='y', labelsize=8)


for i, (mean, count) in enumerate(zip(kmeans_adsorption_means.values, kmeans_sample_counts.values)):
    ax1.text(i, mean / 2, f'n = {count}', ha='center', va='center', fontsize=5, fontweight='bold', color='white')


ax1.legend(['n = number of samples'], loc='upper left', frameon=True, edgecolor='gray', fontsize=8, handlelength=0, handletextpad=0)
ax1.text(-0.15, 1.05, 'a', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')


bar_color = '#B0B0B0'
ax2.barh(kmeans_feature_importance_df_sorted['Feature'], kmeans_feature_importance_df_sorted['Importance'], color=bar_color)
ax2.set_xlabel('Importance', fontsize=7)
ax2.set_ylabel('Feature', fontsize=7, labelpad=2)
ax2.tick_params(axis='y', labelsize=8)
ax2.text(-0.15, 1.05, 'b', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')


plt.subplots_adjust(wspace=0.8)


plt.savefig('combined_plot_kmeans_wiley.svg', dpi=1200, bbox_inches='tight')
plt.show()
