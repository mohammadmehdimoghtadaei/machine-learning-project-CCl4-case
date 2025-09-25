import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# 📌 Load data
data = pd.read_csv('DatasetFileName.csv')

# 📌 Remove outliers using IQR method only for numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
Q1 = data[numeric_cols].quantile(0.25)
Q3 = data[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
mask = ~((data[numeric_cols] < (Q1 - 1.5 * IQR)) | (data[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
data = data[mask]

# 📌 Separate features and target
X = data.drop('Adsorption', axis=1)
y = data['Adsorption']

# 📌 Define models
models = {
    'CatBoost': CatBoostRegressor(
        verbose=0, iterations=300,
        depth=8, learning_rate=0.05, colsample_bylevel=0.5, l2_leaf_reg=5, bagging_temperature=0.5,
        min_child_samples=10, subsample=0.7,
        random_state=42
    ),
    'LightGBM': lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.1, num_leaves=20, lambda_l1=0.5, lambda_l2=1.0, verbose=-1, random_state=42
    ),
    'XGBoost': XGBRegressor(
        n_estimators=166,
        learning_rate=0.0614,
        max_depth=6,
        reg_lambda=3.0,
        reg_alpha=0.5,
        verbosity=0, 
        random_state=42
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=3, max_features="sqrt", random_state=42
    )
}

# 📌 Dictionary to store feature importances
feature_importances = {}

# 📌 Train models and store feature importances
for model_name, model in models.items():
    model.fit(X, y)
    
    if model_name == 'CatBoost':
        importances = np.array(model.get_feature_importance(), dtype=float)
    elif model_name == 'LightGBM':
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        importances = np.array(result.importances_mean, dtype=float)
    elif model_name == 'RandomForest':
        importances = np.array(model.feature_importances_, dtype=float)
    elif model_name == 'XGBoost':
        importances = np.array(model.feature_importances_, dtype=float)
    
    # Ensure correct length
    if len(importances) != X.shape[1]:
        raise ValueError(f"{model_name} feature importance length mismatch: {len(importances)} vs {X.shape[1]}")
    
    feature_importances[model_name] = importances
    
    # Print feature importance for each model
    print(f"\nFeature Importances ({model_name}):")
    for feat, imp in zip(X.columns, importances):
        print(f"{feat}: {imp:.4f}")

# 📌 Create a DataFrame of feature importances
feature_importance_df = pd.DataFrame(feature_importances, index=X.columns)

# 📌 Compute mean importance for each feature
feature_importance_df['Mean_Importance'] = feature_importance_df.mean(axis=1)

# Print mean importances
mean_importance_sorted = feature_importance_df['Mean_Importance'].sort_values(ascending=False)
print("\nMean Feature Importances across models:")
print(mean_importance_sorted)

# 📌 Plot mean feature importance
plt.figure(figsize=(10, 6))
sns.barplot(
    x=mean_importance_sorted.values, 
    y=mean_importance_sorted.index, 
    palette=sns.color_palette("coolwarm", len(mean_importance_sorted))[::-1]  # reversed colors
)
plt.title('Mean Feature Importance', fontsize=16)
plt.xlabel('Mean Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(False)
plt.tight_layout()
#plt.savefig('mean_feature_importance_sorted.svg', dpi=1200, bbox_inches='tight')
plt.show()
