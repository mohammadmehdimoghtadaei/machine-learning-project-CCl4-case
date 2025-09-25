import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.linear_model import (
    ElasticNet, Lasso, Ridge, LinearRegression, HuberRegressor,
    PassiveAggressiveRegressor, TheilSenRegressor, OrthogonalMatchingPursuit,
    BayesianRidge)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor,
    HistGradientBoostingRegressor, VotingRegressor)
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('DatasetFileName.csv')

# --- Start fixes for handling non-numeric data ---
# Ensure the target column is numeric and convert non-numeric values to NaN
# Assume the target column is named 'Adsorption'
if 'Adsorption' not in data.columns:
    raise ValueError("Target column 'Adsorption' not found in the CSV file.")
data['Adsorption'] = pd.to_numeric(data['Adsorption'], errors='coerce')

# Ensure feature columns are numeric and convert non-numeric values to NaN
feature_columns = data.drop('Adsorption', axis=1).columns
for col in feature_columns:
    if col in data.columns:  # Check column existence before conversion
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows containing any NaN values (in features or target)
original_rows = len(data)
data.dropna(inplace=True)
new_rows = len(data)
print(f"Removed {original_rows - new_rows} rows with non-numeric values.")


# Check if any data remains
if data.empty:
    raise ValueError("No data remaining after converting to numeric and removing NaN values. Check your input CSV file.")
# --- End fixes for handling non-numeric data ---

# Independent (X) and dependent (y) variables (now with cleaned numeric data)
X = data.drop('Adsorption', axis=1)
y = data['Adsorption']

# Function to remove outliers using IQR
def remove_outliers_iqr(df, column):
    # Ensure the column exists before calculating quantiles
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found for outlier removal. Skipping.")
        return df
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# --- Start outlier removal fixes ---
# First, create a copy of the cleaned dataframe to avoid SettingWithCopyWarning
data_cleaned = data.copy()

# Apply outlier removal on the target variable 'Adsorption'
print(f"Rows before outlier removal on 'Adsorption': {len(data_cleaned)}")
data_cleaned = remove_outliers_iqr(data_cleaned, 'Adsorption')
print(f"Rows after outlier removal on 'Adsorption': {len(data_cleaned)}")


# Optionally, apply outlier removal on features (X) as well
for col in X.columns:
     # Ensure the column exists in the potentially modified data_cleaned DataFrame
     if col in data_cleaned.columns:
        rows_before_feature_outlier = len(data_cleaned)
        data_cleaned = remove_outliers_iqr(data_cleaned, col)
        rows_after_feature_outlier = len(data_cleaned)
        if rows_before_feature_outlier != rows_after_feature_outlier:
            print(f"Rows after outlier removal on feature '{col}': {rows_after_feature_outlier} (Removed {rows_before_feature_outlier - rows_after_feature_outlier})")


# Check if any data remains after outlier removal
if data_cleaned.empty:
    raise ValueError("No data remaining after removing outliers. Check outlier thresholds or data distribution.")
# --- End outlier removal fixes ---

# Update X and y after removing outliers
X = data_cleaned.drop('Adsorption', axis=1)
y = data_cleaned['Adsorption']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nData shapes after cleaning and splitting:")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


# Define the models with updated parameters to reduce overfitting
# (Ensure parameters are compatible with your data)
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=3, max_features="sqrt", random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),  # Add random_state for reproducibility
    "Support Vector Regression": SVR(),
    "LightGBM": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=20, lambda_l1=0.5, lambda_l2=1.0, verbose=-1, random_state=42),  # verbose=-1 to fully suppress output
    "XGBoost": xgb.XGBRegressor(
        n_estimators=166,
        learning_rate=0.0614,
        max_depth=6,
        reg_lambda=3.0,
        reg_alpha=0.5,
        verbosity=0, 
        random_state=42
    ),
    "CatBoost": cb.CatBoostRegressor(
        verbose=0, iterations=300,
        depth=8, learning_rate=0.05, colsample_bylevel=0.5, l2_leaf_reg=5, bagging_temperature=0.5,
        min_child_samples=10, subsample=0.7,
        random_state=42
    ),
    "KNeighbors Regressor": KNeighborsRegressor(),
    "Bagging": BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42),  # random_state for reproducibility in base tree
        n_estimators=50,
        random_state=42
    ),
    "HistGradient Boosting": HistGradientBoostingRegressor(
        max_depth=9,
        l2_regularization=1.0,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42
    ),
    "Multiple Linear Regression (MLR)": LinearRegression(),
    "Artificial Neural Network (ANN)": MLPRegressor(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,  # Add random_state
        early_stopping=True,  # Enable early stopping to prevent overfitting
        n_iter_no_change=10
    ),
    "Voting Regressor": VotingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(max_depth=10, min_samples_split=5, random_state=42)),
            ('gb', GradientBoostingRegressor(random_state=42)),
            ('svr', SVR())  # You can also tune SVR parameters
        ]
    ),
    "AdaBoost": AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=5, random_state=42),  # random_state for reproducibility in base tree
        n_estimators=100,
        random_state=42
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        min_samples_split=5,  # Add regularization parameters
        min_samples_leaf=2
    ),
}


# Dictionary to store the metrics for direct evaluation
train_r2_scores = {}
test_r2_scores = {}
train_rmse_scores = {}
test_rmse_scores = {}

print("\n--- Starting Direct Evaluation ---")
# Loop through each model, fit, and calculate metrics for direct evaluation
for name, model in models.items():
    print(f"Evaluating {name}...")
    try:
        # Fit the model
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate R²
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Calculate RMSE
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Store the results
        train_r2_scores[name] = train_r2
        test_r2_scores[name] = test_r2
        train_rmse_scores[name] = train_rmse
        test_rmse_scores[name] = test_rmse
        print(f"{name}: Train R2={train_r2:.3f}, Test R2={test_r2:.3f} | Train RMSE={train_rmse:.3f}, Test RMSE={test_rmse:.3f}")

    except Exception as e:
        print(f"Error during direct evaluation for model {name}: {e}")
        # Store NaN or default values if evaluation fails
        train_r2_scores[name] = np.nan
        test_r2_scores[name] = np.nan
        train_rmse_scores[name] = np.nan
        test_rmse_scores[name] = np.nan

# Convert the results into a DataFrame for easier plotting
r2_df = pd.DataFrame({
    'Model': list(train_r2_scores.keys()) * 2,
    'R² Score': list(train_r2_scores.values()) + list(test_r2_scores.values()),
    'Dataset': ['Train'] * len(models) + ['Test'] * len(models)
}).dropna(subset=['R² Score'])  # Exclude models that failed from the plot

rmse_df = pd.DataFrame({
    'Model': list(train_rmse_scores.keys()) * 2,
    'RMSE': list(train_rmse_scores.values()) + list(test_rmse_scores.values()),
    'Dataset': ['Train'] * len(models) + ['Test'] * len(models)
}).dropna(subset=['RMSE'])  # Exclude models that failed from the plot

# Plotting R² Scores (Train and Test)
if not r2_df.empty:
    plt.figure(figsize=(14, 7))  # Slightly larger plot size
    bar_plot = sns.barplot(x='Model', y='R² Score', hue='Dataset', data=r2_df, palette=['#66b3ff', '#ff9999'])
    for p in bar_plot.patches:
        if not np.isnan(p.get_height()) and p.get_height() != 0:  # Check for NaN and zero height
            bar_plot.annotate(format(p.get_height(), '.2f'),
                              (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha='center', va='center',
                              xytext=(0, 5),
                              textcoords='offset points')
    plt.title('R² Score Comparison for Train and Test Sets (Direct Evaluation)')
    plt.xticks(rotation=45, ha='right')
    # Dynamically set y-axis limits based on actual values with some margin
    min_r2 = r2_df['R² Score'].min()
    max_r2 = r2_df['R² Score'].max()
    plt.ylim(min(0, min_r2 - 0.1), max(1.0, max_r2 + 0.1))  # More dynamic range
    plt.legend(loc='best')  # Better legend placement
    plt.tight_layout()
    #plt.savefig('R2_direct_evaluation.svg', dpi=1200, bbox_inches='tight')
    plt.show()
else:
    print("Skipping R2 direct evaluation plot as no models were successfully evaluated.")


# Plotting RMSE Scores (Train and Test)
if not rmse_df.empty:
    plt.figure(figsize=(14, 7))  # Slightly larger plot size
    bar_plot = sns.barplot(x='Model', y='RMSE', hue='Dataset', data=rmse_df, palette=['#66b3ff', '#ff9999'])
    for p in bar_plot.patches:
         if not np.isnan(p.get_height()) and p.get_height() > 0:  # Check for NaN and positive height
            bar_plot.annotate(format(p.get_height(), '.2f'),
                              (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha='center', va='center',
                              xytext=(0, 5),
                              textcoords='offset points')
    plt.title('RMSE Comparison for Train and Test Sets (Direct Evaluation)')
    plt.xticks(rotation=45, ha='right')
    max_rmse = rmse_df['RMSE'].max()
    plt.ylim(0, max(max_rmse * 1.1, 0.1))  # Increase y-axis range with some margin
    plt.legend(loc='best')  # Better legend placement
    plt.tight_layout()
    #plt.savefig('RMSE_direct_evaluation.svg', dpi=1200, bbox_inches='tight')
    plt.show()
else:
    print("Skipping RMSE direct evaluation plot as no models were successfully evaluated.")


#-------------------------------------------------------------------

# Cross-Validation Evaluation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Dictionary to store the metrics for cross-validation
cv_train_r2_scores = {}
cv_test_r2_scores = {}
cv_train_rmse_scores = {}
cv_test_rmse_scores = {}

print("\n--- Starting Cross-Validation ---")
# Loop through each model and perform cross-validation
for name, model in models.items():
    print(f"Cross-validating {name}...")
    try:  # --- Start try block ---
        # Perform cross-validation and calculate both train and test scores
        cv_results = cross_validate(
            model, X, y, cv=cv,
            scoring=('r2', 'neg_root_mean_squared_error'),
            return_train_score=True,
            error_score='raise'  # Important: raise errors instead of returning NaN
        )

        # Store the results (mean over folds)
        cv_train_r2 = np.mean(cv_results['train_r2'])
        cv_test_r2 = np.mean(cv_results['test_r2'])
        # Negate the mean of neg_rmse to get positive RMSE
        cv_train_rmse = -np.mean(cv_results['train_neg_root_mean_squared_error'])
        cv_test_rmse = -np.mean(cv_results['test_neg_root_mean_squared_error'])

        cv_train_r2_scores[name] = cv_train_r2
        cv_test_r2_scores[name] = cv_test_r2
        cv_train_rmse_scores[name] = cv_train_rmse
        cv_test_rmse_scores[name] = cv_test_rmse
        print(f"{name}: CV Train R2={cv_train_r2:.3f}, CV Test R2={cv_test_r2:.3f} | CV Train RMSE={cv_train_rmse:.3f}, CV Test RMSE={cv_test_rmse:.3f}")

    except Exception as e:  # --- except block to catch errors ---
        print(f"Error during cross-validation for model {name}: {e}")
        # Record invalid values (NaN) for models that failed
        cv_train_r2_scores[name] = np.nan
        cv_test_r2_scores[name] = np.nan
        cv_train_rmse_scores[name] = np.nan
        cv_test_rmse_scores[name] = np.nan


# Convert the cross-validation results into DataFrames for plotting
# Exclude models that failed during CV from the plot DataFrames
cv_r2_df = pd.DataFrame({
    'Model': list(cv_train_r2_scores.keys()) * 2,
    'R² Score': list(cv_train_r2_scores.values()) + list(cv_test_r2_scores.values()),
    'Dataset': ['Train (CV Avg)'] * len(models) + ['Test (CV Avg)'] * len(models)
}).dropna(subset=['R² Score'])

cv_rmse_df = pd.DataFrame({
    'Model': list(cv_train_rmse_scores.keys()) * 2,
    'RMSE': list(cv_train_rmse_scores.values()) + list(cv_test_rmse_scores.values()),
    'Dataset': ['Train (CV Avg)'] * len(models) + ['Test (CV Avg)'] * len(models)
}).dropna(subset=['RMSE'])

# Plotting R² Scores (Cross-Validation)
if not cv_r2_df.empty:
    plt.figure(figsize=(14, 7))
    bar_plot = sns.barplot(x='Model', y='R² Score', hue='Dataset', data=cv_r2_df, palette=['#66b3ff', '#99ff99'])  # Changed second color
    for p in bar_plot.patches:
        if not np.isnan(p.get_height()) and p.get_height() != 0:  # Check for NaN and zero
            bar_plot.annotate(format(p.get_height(), '.2f'),
                              (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha='center', va='center',
                              xytext=(0, 5),
                              textcoords='offset points')
    plt.title('R² Score Comparison (Cross-Validation Average)')
    plt.xticks(rotation=45, ha='right')
    min_cv_r2 = cv_r2_df['R² Score'].min()
    max_cv_r2 = cv_r2_df['R² Score'].max()
    plt.ylim(min(0, min_cv_r2 - 0.1), max(1.0, max_cv_r2 + 0.1))  # More dynamic range
    plt.legend(loc='best')
    plt.tight_layout()
    #plt.savefig('R2_CrossValidation.svg', dpi=1200, bbox_inches='tight')
    plt.show()
else:
    print("Skipping R2 cross-validation plot as no models were successfully cross-validated.")


# Plotting RMSE Scores (Cross-Validation)
if not cv_rmse_df.empty:
    plt.figure(figsize=(14, 7))
    bar_plot = sns.barplot(x='Model', y='RMSE', hue='Dataset', data=cv_rmse_df, palette=['#66b3ff', '#99ff99'])
    for p in bar_plot.patches:
        if not np.isnan(p.get_height()) and p.get_height() > 0:  # Check for NaN and positive values
            bar_plot.annotate(format(p.get_height(), '.2f'),
                              (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha='center', va='center',
                              xytext=(0, 5),
                              textcoords='offset points')
    plt.title('RMSE Comparison (Cross-Validation Average)')
    plt.xticks(rotation=45, ha='right')
    max_cv_rmse = cv_rmse_df['RMSE'].max()
    plt.ylim(0, max(max_cv_rmse * 1.1, 0.1))  # Increase y-axis range with margin
    plt.legend(title=None, loc='best')  # Remove legend title and improve placement
    plt.tight_layout()
    #plt.savefig('RMSE_CrossValidation.svg', dpi=1200, bbox_inches='tight')
    plt.show()
else:
    print("Skipping RMSE cross-validation plot as no models were successfully cross-validated.")


# Print R² and RMSE for Cross-Validation (using .get() for safety)
print("\n--- Final Cross-Validation Results (Mean Scores) ---")
print("R² Scores (Cross-Validation Average):")
# Sort by test R2 score (highest to lowest)
sorted_test_r2 = sorted(cv_test_r2_scores.items(), key=lambda item: item[1] if not np.isnan(item[1]) else -np.inf, reverse=True)
for model, score in sorted_test_r2:
    train_score = cv_train_r2_scores.get(model, np.nan)  # Safely retrieve value
    print(f"{model:<25}: Train = {train_score:.3f}, Test = {score:.3f}")

print("\nRMSE Scores (Cross-Validation Average):")
# Sort by test RMSE score (lowest to highest)
sorted_test_rmse = sorted(cv_test_rmse_scores.items(), key=lambda item: item[1] if not np.isnan(item[1]) else np.inf)
for model, score in sorted_test_rmse:
    train_score = cv_train_rmse_scores.get(model, np.nan)  # Safely retrieve value
    print(f"{model:<25}: Train = {train_score:.3f}, Test = {score:.3f}")

print("\n--- Script Finished ---")