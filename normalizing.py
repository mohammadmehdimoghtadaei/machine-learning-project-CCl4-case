import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data from CSV file
input_file = 'DatasetFileName.csv'  # Your input file name
data = pd.read_csv(input_file)

# List of columns to ignore
ignore_columns = ['filename']

# Separate the columns to be ignored
data_to_normalize = data.drop(columns=ignore_columns)

# Normalize the remaining data to range [0, 1]
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data_to_normalize), columns=data_to_normalize.columns)

# Re-add the ignored columns to the normalized data
final_data = pd.concat([normalized_data, data[ignore_columns]], axis=1)

# Save the normalized data to a new CSV file
output_file = 'output.csv'  # Your output file name
final_data.to_csv(output_file, index=False)

print(f"Data has been normalized and saved to {output_file}.")
