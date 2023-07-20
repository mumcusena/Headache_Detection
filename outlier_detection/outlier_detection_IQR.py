import pandas as pd
import numpy as np

# Load the CSV file into a Pandas DataFrame
data = pd.read_csv('deneme.csv')
dict = {'Male': 0, 'Female': 1, 'female': 1, 'male':0}
data["Gender"].replace(dict, inplace=True)
# Create a copy of the DataFrame for outlier detection
data_copy = data.copy()

# Set the threshold for outlier detection (e.g., 1.5 times the IQR)
threshold = 0.5

# Identify the outlier rows for each column
outlier_indices = []
for col in data_copy.columns:
    # Calculate the first quartile (Q1) and third quartile (Q3) for the column
    q1 = data_copy[col].quantile(0.25)
    q3 = data_copy[col].quantile(0.75)

    # Calculate the interquartile range (IQR) for the column
    iqr = q3 - q1

    # Define the bounds for outlier detection
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    # Identify the outlier rows for the column
    outliers = (data_copy[col] < lower_bound) | (data_copy[col] > upper_bound)

    # Collect the indices of the outlier rows
    outlier_indices.extend(data_copy.index[outliers])

# Remove duplicate outlier indices
outlier_indices = list(set(outlier_indices))

# Remove the outlier rows from the original data
filtered_data = data.drop(outlier_indices)

# Write the outlier rows to a separate file
outlier_data = data.loc[outlier_indices]

dict2 = {0:"Male", 1:"Female"}
filtered_data["Gender"].replace(dict2, inplace=True)
outlier_data["Gender"].replace(dict2, inplace=True)

outlier_data.to_csv('outlier_data_05_IQR.csv', index=False)

# Write the filtered data (without outliers) to another file
filtered_data.to_csv('DataS_10_IQR.csv', index=False)

# Print the indices of the outliers
print("Outlier indices:")
print(outlier_indices)
