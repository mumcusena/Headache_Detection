import pandas as pd
from sklearn.ensemble import IsolationForest

# Load the CSV file into a Pandas DataFrame
data = pd.read_csv('deneme.csv')
dict = {'Male': 0, 'Female': 1, 'female': 1, 'male':0}
data["Gender"].replace(dict, inplace=True)
# Create an Isolation Forest instance
clf = IsolationForest(n_estimators=100, contamination=0.5)  # Adjust the contamination parameter as needed

# Fit the model on the entire data
clf.fit(data)

# Predict the outliers
outlier_predictions = clf.predict(data)

# Get the indices of the outlier rows
outlier_indices = data.index[outlier_predictions == -1]

# Remove the outlier rows from the original data and create a separate DataFrame
outlier_data = data.loc[outlier_indices]

# Drop the outlier rows from the original data
filtered_data = data.drop(outlier_indices)

dict2 = {0:"Male", 1:"Female"}
filtered_data["Gender"].replace(dict2, inplace=True)
outlier_data["Gender"].replace(dict2, inplace=True)

# Write the outlier rows to a separate file
outlier_data.to_csv('outlier_data_50_IF.csv', index=False)
# Write the filtered data (without outliers) to another file
filtered_data.to_csv('DataS_50_IF.csv', index=False)

# Print the indices of the outliers
print("Outlier indices:")
print(outlier_indices)
