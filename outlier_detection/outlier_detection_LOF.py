import pandas as pd
import numpy as np
import os
from sklearn.neighbors import LocalOutlierFactor

# Load the CSV file into a Pandas DataFrame
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, 'deneme.csv')

data = pd.read_csv(data_path)

# Convert the "gender" column to numeric values (0 for "Male", 1 for "Female")
dict = {'Male': 0, 'Female': 1, 'female': 1, 'male':0}
data["Gender"].replace(dict, inplace=True)
# Create a Local Outlier Factor instance
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.50)  # Adjust parameters as needed

# Fit the model on the selected columns
clf.fit_predict(data)

# Obtain the negative outlier factor scores
outlier_scores = clf.negative_outlier_factor_

# Get the indices of the outlier rows
outlier_indices = data.index[outlier_scores < -2]  # Adjust the threshold as needed

# Remove the outlier rows from the original data
filtered_data = data.drop(outlier_indices)

# Convert the "gender" column back to its original string values
dict2 = {0:"Male", 1:"Female"}
filtered_data["Gender"].replace(dict2, inplace=True)
data["Gender"].replace(dict2, inplace=True)
# Write the outlier rows to a separate file
outlier_data = data.loc[outlier_indices]
outlier_data.to_csv('outlier_data_50_LOF.csv', index=False)

# Write the filtered data (without outliers) to another file
filtered_data.to_csv('DataS_50_LOF.csv', index=False)

# Print the indices of the outliers
print("Outlier indices:")
print(outlier_indices)
