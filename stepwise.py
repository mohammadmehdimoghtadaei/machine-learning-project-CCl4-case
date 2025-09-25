import pandas as pd
import statsmodels.api as sm

# Load data
data = pd.read_csv('DatasetFileName.csv')
data = data.drop(columns=["filename"])
# List of independent variables
variables = [ 'HOA', 'Average Henry coefficient', 'LCD', 'PLD', 'LCD/PLD', 
                    'LFPD', 'SAV(cm3/g)', 'VSA(m2/cm3)', 'GSA(m2/g)', 'Oxygen/Metal', 
                    'Metal Percentage', 'Nitrogen to Oxygen Ratio', 'Average Electronegativity', 
                    'VF', 'Molar Mass', 'Mass/HOA', 'Mass/PLD',  'tot_atoms', 
                    'non_metals', 'metals_count', 'total_bonds', 'single_bonds', 
                    'double_bonds', 'bonds_to_metal', 'cell_volume', 'H-C', 
                    'C-N', 'C-O']

# Independent variables (features)
X = data[variables]

# Dependent variable (target)
y = data['Adsorption']

# Function to perform Stepwise Regression
def stepwise_selection(X, y, significance_in=0.05, significance_out=0.05):
    initial_list = []
    included = list(initial_list)
    
    while True:
        changed = False
        
        # Check for adding variables
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(X[included + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < significance_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            print(f"Added: {best_feature}, p-value: {best_pval}")
        
        # Check for removing variables
        model = sm.OLS(y, sm.add_constant(X[included])).fit()
        pvalues = model.pvalues.iloc[1:]  # Exclude the constant term
        worst_pval = pvalues.max()
        if worst_pval > significance_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True
            print(f"Removed: {worst_feature}, p-value: {worst_pval}")
        
        if not changed:
            break

    return included

# Run stepwise selection
selected_variables = stepwise_selection(X, y)

print("Remaining important variables:", selected_variables)

# Remove unimportant variables from the data
columns_to_keep = ['Adsorption'] + selected_variables  # Keep the dependent variable as well
filtered_data = data[columns_to_keep]

# Save the new data to a CSV file
output_file = 'output.csv'
filtered_data.to_csv(output_file, index=False)

print(f"New file with important variables saved to {output_file}.")
