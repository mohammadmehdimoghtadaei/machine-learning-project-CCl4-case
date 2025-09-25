import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Load the data
data = pd.read_csv('filtered_data.csv')

# Independent (X) and dependent (y) variables
X = data.drop('Adsorption', axis=1)
y = data['Adsorption']

# Function to remove outliers using IQR for y
def remove_outliers(y):
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask = (y >= lower_bound) & (y <= upper_bound)
    return mask

# Remove outliers from y
mask = remove_outliers(y)
X = X[mask]
y = y[mask]

# Define the models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=3, max_features="sqrt", random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Support Vector Regression": SVR(),
    "LightGBM": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=20, lambda_l1=0.5, lambda_l2=1.0, verbose=-1, random_state=42),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=166,
        learning_rate=0.0614,
        max_depth=6,
        reg_lambda=3.0,
        reg_alpha=0.5,
        verbosity=0, 
        random_state=42),
    "CatBoost": cb.CatBoostRegressor(
        verbose=0, iterations=300,
        depth=8, learning_rate=0.05, colsample_bylevel=0.5, l2_leaf_reg=5, bagging_temperature=0.5,
        min_child_samples=10, subsample=0.7,
        random_state=42),
    "KNeighbors Regressor": KNeighborsRegressor(),
    "Bagging": BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=50, random_state=42),
    "HistGradient Boosting": HistGradientBoostingRegressor(),
    "Multiple Linear Regression (MLR)": LinearRegression(),
    "Artificial Neural Network (ANN)": MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500),
    "Voting Regressor": VotingRegressor(
        estimators=[
            ('rf', RandomForestRegressor()),
            ('gb', GradientBoostingRegressor()),
            ('svr', SVR())
        ]
    ),
}

# Create a figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# List of selected models for combined plot
selected_models = ['CatBoost', 'LightGBM', 'XGBoost', 'Random Forest']
labels = ['a', 'b', 'c', 'd']  # Labels for each subplot

# Define 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterate over selected models and plot in the 2x2 grid
for idx, model_name in enumerate(selected_models):
    model = models[model_name]
    # Cross-validated predictions
    y_pred = cross_val_predict(model, X, y, cv=cv)
    
    # Determine subplot position (row, col)
    row = idx // 2
    col = idx % 2
    
    # Scatter plot
    axs[row, col].scatter(y, y_pred, color='blue', alpha=0.8)
    
    # Red solid diagonal line
    axs[row, col].plot([0, 1.0], [0, 1.0], 'r-', lw=2)
    
    # Set axis limits to 0 to 1.0
    axs[row, col].set_xlim(0, 0.8)
    axs[row, col].set_ylim(0, 0.8)
    
    # Set axis labels conditionally
    if model_name in ['XGBoost', 'Random Forest']:
        axs[row, col].set_xlabel('Actual Values', fontsize=14, fontfamily='Times New Roman')
    if model_name in ['CatBoost', 'XGBoost']:
        axs[row, col].set_ylabel('Predicted Values', fontsize=14, fontfamily='Times New Roman')
    
    # Add letter label (a, b, c, d) to the top left corner
    axs[row, col].text(0.05, 0.95, labels[idx], fontsize=14, weight='bold', fontfamily='Times New Roman', transform=axs[row, col].transAxes)

    # Remove grid
    axs[row, col].grid(False)

    # Ensure all spines (frame) are visible
    axs[row, col].spines['top'].set_visible(True)
    axs[row, col].spines['right'].set_visible(True)
    axs[row, col].spines['bottom'].set_visible(True)
    axs[row, col].spines['left'].set_visible(True)

    # Set ticks to point outward
    axs[row, col].tick_params(axis='both', which='both', direction='out', length=6)
    axs[row, col].tick_params(labelsize=12)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
