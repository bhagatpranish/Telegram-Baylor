import os
import pandas as pd

# Set the path to the folder containing the CSV files
csv_folder = 'csv'

# Get a list of all the CSV files in the folder
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

# Initialize a list to store the DataFrames
dataframes = []

# Loop through the list of CSV files and read the data
for csv_file in csv_files:
    file_path = os.path.join(csv_folder, csv_file)
    csv_data = pd.read_csv(file_path, low_memory=False)
    dataframes.append(csv_data)

# Combine all the DataFrames using pandas.concat
combined_csv_data = pd.concat(dataframes, ignore_index=True)

# Save the combined data to a new CSV file in the 'csv' folder
output_file = os.path.join(csv_folder, 'combined_csv_data.csv')
combined_csv_data.to_csv(output_file, index=False)

print("All CSV files have been combined and saved as 'combined_csv_data.csv' in the 'csv' folder.")
