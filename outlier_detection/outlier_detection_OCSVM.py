import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler

# Load the CSV file into a Pandas DataFrame
data = pd.read_csv('deneme.csv')
dict = {'Male': 0, 'Female': 1, 'female': 1, 'male':0}
data["Gender"].replace(dict, inplace=True)
# Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

# Create an instance of OneClassSVM
svm = OneClassSVM(nu=0.1)  # Adjust the nu parameter according to your requirements

# Fit the model on the normalized data
svm.fit(normalized_data)

# Predict outliers
outlier_predictions = svm.predict(normalized_data)

# Get the indices of the outlier rows
outlier_indices = data.index[outlier_predictions == -1]

# Remove the outlier rows from the original data
filtered_data = data.drop(outlier_indices)

# Write the outlier rows to a separate file
outlier_data = data.loc[outlier_indices]

dict2 = {0:"Male", 1:"Female"}
filtered_data["Gender"].replace(dict2, inplace=True)
outlier_data["Gender"].replace(dict2, inplace=True)

outlier_data.to_csv('outlier_data_10_OCSVM.csv', index=False)

# Write the filtered data (without outliers) to another file
filtered_data.to_csv('DataS_10_OCSVM.csv', index=False)

# Print the indices of the outliers
print("Outlier indices:")
print(outlier_indices)
