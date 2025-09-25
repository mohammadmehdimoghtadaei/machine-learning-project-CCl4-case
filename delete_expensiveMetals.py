import pandas as pd


# Read the CSV file into a pandas DataFrame
input_file = "DatasetFileName.csv"
output_file = "output.csv"
df = pd.read_csv(input_file)

# Define the keyword(s) and column(s) to search for
keywords = ["Au", "Ag","Pt","Pd","Ru","Y", "In","Hf","Ga","Dy","Pr", "Te","U","Se","Mo","Sm", "Nd","Pr","Gd","Ir"]  # Add your specific keywords here
columns_to_search = ["All_Metals"]    # Add the specific column name(s) to search

# Filter rows that do not contain the specified keyword(s) in the specified column(s)
filtered_df = df[~df[columns_to_search].apply(lambda x: x.str.contains('|'.join(keywords), case=False)).any(axis=1)]

# Export the filtered DataFrame to a new CSV file
filtered_df.to_csv(output_file, index=False)

print("Filtered CSV file has been created successfully.")