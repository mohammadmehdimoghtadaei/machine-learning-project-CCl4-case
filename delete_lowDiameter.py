import pandas as pd

# Read the CSV file
df = pd.read_csv('DatasetFileName.csv')

# Filter the specific column based on the condition
column_name = 'LCD'  # Replace with the actual column name
condition = df[column_name] < 6.4 
df = df[~condition]  # Use ~ to negate the condition and keep the rows where the condition is False

# Export the modified data to a new CSV file
df.to_csv('ouput.csv', index=False)
