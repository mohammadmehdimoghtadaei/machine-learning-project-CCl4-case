import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# Load dataset
data = pd.read_csv('DatasetFileName.csv')

# Select top 50 structures based on Adsorption
top_50_data = data.nlargest(50, 'Adsorption')

# Separate features and target
X_top_50 = top_50_data.drop('Adsorption', axis=1)
y_top_50 = top_50_data['Adsorption']

# Split into two groups: top 25 and bottom 25
group1 = X_top_50.iloc[:25]   # higher adsorption
group2 = X_top_50.iloc[25:]   # lower adsorption

# Perform Mann–Whitney U test for each feature
p_values = []
threshold = 0.05

for feature in X_top_50.columns:
    stat, p_value = mannwhitneyu(group1[feature], group2[feature])
    p_values.append((feature, p_value))

# Create DataFrame with p-values
p_value_df = pd.DataFrame(p_values, columns=['Feature', 'p-value']).sort_values(by='p-value')

# Select significant features
significant_features = p_value_df[p_value_df['p-value'] < threshold]
print("Significant Features (p-value < threshold):\n", significant_features)

# Plot p-values
plt.figure(figsize=(12, 8))
plt.barh(p_value_df['Feature'], p_value_df['p-value'], color='skyblue')
plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
plt.xlabel('p-value', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.legend(loc='lower right')
plt.tight_layout()
plt.grid(False)
plt.show()
