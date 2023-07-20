import numpy as np
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, multilabel_confusion_matrix
from skmultilearn.problem_transform import LabelPowerset
from sklearn.preprocessing import LabelBinarizer

# Assuming you have your input features in X and labels in y
# X should be a 2D array where each row represents a sample and each column represents a feature
# y should be a 2D array where each row represents a sample and each column represents a label
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, 'deneme_nonzero.csv')

head_ache = pd.read_csv(data_path)
dict = {'Male': 0, 'Female': 1, 'female':1, 'male':0}
head_ache["Gender"].replace(dict, inplace=True)

label4 = head_ache.iloc[:, -1:]
head_ache = head_ache.iloc[:, :-1]
label3 = head_ache.iloc[:, -1:]
head_ache = head_ache.iloc[:, :-1]
label2 = head_ache.iloc[:, -1:]
head_ache = head_ache.iloc[:, :-1]
label1 = head_ache.iloc[:, -1:]
head_ache = head_ache.iloc[:, :-1]

feature_cols = head_ache.iloc[:, 2:].columns.to_list()
for i in range(17):
    #if i == 0 or i == 2 or i ==4:
     #   continue
    feature_cols.remove("trigger_food_{}".format(i+1))
X = head_ache[feature_cols].values

y4= label4.values
label_binarizer = LabelBinarizer()
y4 = label_binarizer.fit_transform(y4)
y3= label3.values
label_binarizer = LabelBinarizer()
y3 = label_binarizer.fit_transform(y3)
y2= label2.values
label_binarizer = LabelBinarizer()
y2 = label_binarizer.fit_transform(y2)
y1= label1.values
label_binarizer = LabelBinarizer()
y1 = label_binarizer.fit_transform(y1)


# Define the number of folds for K-fold cross-validation
num_folds = 9

confusion_matrix_sum1 = np.zeros((2, 2))
# Initialize the evaluation metrics
accuracies1 = []
precisions1 = []
recalls1 = []

confusion_matrix_sum2 = np.zeros((2, 2))
# Initialize the evaluation metrics
accuracies2 = []
precisions2 = []
recalls2 = []

confusion_matrix_sum3 = np.zeros((2, 2))
# Initialize the evaluation metrics
accuracies3 = []
precisions3 = []
recalls3 = []

confusion_matrix_sum4 = np.zeros((2, 2))
# Initialize the evaluation metrics
accuracies4 = []
precisions4 = []
recalls4 = []

total_accuracies = []
total_accuracies2 = []
# Perform K-fold cross-validation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=1)

for train_index, test_index in kf.split(X):
    # Split the data into training and test sets for the current fold
    X_train, X_test = X[train_index], X[test_index]
    y4_train, y4_test = y4[train_index], y4[test_index]

    # Create and train the decision tree classifier
    classifier4 = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5)
    classifier4.fit(X_train, y4_train)

    # Make predictions on the test set
    y4_pred = classifier4.predict(X_test)

    # Calculate the evaluation metrics
    accuracy4 = accuracy_score(y4_test, y4_pred)
    precision4 = precision_score(y4_test, y4_pred)
    recall4 = recall_score(y4_test, y4_pred)

    # Calculate the confusion matrix
    cm4 = confusion_matrix(y4_test, y4_pred)
    confusion_matrix_sum4 += cm4

   # Append the metrics to the lists
    accuracies4.append(accuracy4)
    precisions4.append(precision4)
    recalls4.append(recall4)

    y3_train, y3_test = y3[train_index], y3[test_index]

    # Create and train the decision tree classifier
    classifier3 = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5)
    classifier3.fit(X_train, y3_train)

    # Make predictions on the test set
    y3_pred = classifier3.predict(X_test)

    # Calculate the evaluation metrics
    accuracy3 = accuracy_score(y3_test, y3_pred)
    precision3 = precision_score(y3_test, y3_pred, average='micro')
    recall3 = recall_score(y3_test, y3_pred, average='micro')

    # Calculate the confusion matrix
    cm3 = confusion_matrix(y3_test, y3_pred)
    confusion_matrix_sum3 += cm3

   # Append the metrics to the lists
    accuracies3.append(accuracy3)
    precisions3.append(precision3)
    recalls3.append(recall3)


    y2_train, y2_test = y2[train_index], y2[test_index]

    # Create and train the decision tree classifier
    classifier2 = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5)
    classifier2.fit(X_train, y2_train)

    # Make predictions on the test set
    y2_pred = classifier2.predict(X_test)

    # Calculate the evaluation metrics
    accuracy2 = accuracy_score(y2_test, y2_pred)
    precision2 = precision_score(y2_test, y2_pred, average='micro')
    recall2 = recall_score(y2_test, y2_pred, average='micro')

    # Calculate the confusion matrix
    cm2 = confusion_matrix(y2_test, y2_pred)
    confusion_matrix_sum2 += cm2

   # Append the metrics to the lists
    accuracies2.append(accuracy2)
    precisions2.append(precision2)
    recalls2.append(recall2)

    y_train, y_test = y1[train_index], y1[test_index]

    # Create and train the decision tree classifier
    classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5)
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Calculate the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrix_sum1 += cm

   # Append the metrics to the lists
    accuracies1.append(accuracy)
    precisions1.append(precision)
    recalls1.append(recall)
    print(y_pred)
    summation = 0
    summation2 = 0
    for j in range(len(y_test)):
        if y_pred[j] == y_test[j] and y2_pred[j] == y2_test[j] and y3_pred[j] == y3_test[j] and y4_pred[j] == y4_test[j]:
            summation += 1
        if y_pred[j] == y_test[j]:
            summation2 += 1
        if y2_pred[j] == y2_test[j]:
            summation2 += 1
        if y3_pred[j] == y3_test[j]:
            summation2 += 1
        if y4_pred[j] == y4_test[j]:
            summation2 += 1
    total_accuracies.append(summation/(1.0*len(y_test)))
    total_accuracies2.append(summation2/(4.0*len(y_test)))

# Calculate the average metrics across all folds
avg_accuracy4 = np.mean(accuracies4)
avg_precision4 = np.mean(precisions4)
avg_recall4 = np.mean(recalls4)

# Calculate the average confusion matrix across all folds
avg_confusion_matrix4 = confusion_matrix_sum4 / num_folds
normalized_confusion_matrix4 = avg_confusion_matrix4 / avg_confusion_matrix4.sum(axis=1, keepdims=True)

# Print the results
print("Average Accuracy(4):", avg_accuracy4)
print("Average Precision(4):", avg_precision4)
print("Average Recall(4):", avg_recall4)
print("Average Confusion Matrix(4):")
print(normalized_confusion_matrix4)

# Calculate the average metrics across all folds
avg_accuracy3 = np.mean(accuracies3)
avg_precision3 = np.mean(precisions3)
avg_recall3 = np.mean(recalls3)

# Calculate the average confusion matrix across all folds
avg_confusion_matrix3 = confusion_matrix_sum3 / num_folds
normalized_confusion_matrix3 = avg_confusion_matrix3 / avg_confusion_matrix3.sum(axis=1, keepdims=True)
# Print the results
print("Average Accuracy(3):", avg_accuracy3)
print("Average Precision(3):", avg_precision3)
print("Average Recall(3):", avg_recall3)
print("Average Confusion Matrix(3):")
print(normalized_confusion_matrix3)

# Calculate the average metrics across all folds
avg_accuracy2 = np.mean(accuracies2)
avg_precision2 = np.mean(precisions2)
avg_recall2 = np.mean(recalls2)

# Calculate the average confusion matrix across all folds
avg_confusion_matrix2 = confusion_matrix_sum2 / num_folds
normalized_confusion_matrix2 = avg_confusion_matrix2 / avg_confusion_matrix2.sum(axis=1, keepdims=True)

# Print the results
print("Average Accuracy(2):", avg_accuracy2)
print("Average Precision(2):", avg_precision2)
print("Average Recall(2):", avg_recall2)
print("Average Confusion Matrix(2):")
print(normalized_confusion_matrix2)

# Calculate the average metrics across all folds
avg_accuracy1 = np.mean(accuracies1)
avg_precision1 = np.mean(precisions1)
avg_recall1 = np.mean(recalls1)

# Calculate the average confusion matrix across all folds
avg_confusion_matrix1 = confusion_matrix_sum1 / num_folds
normalized_confusion_matrix1 = avg_confusion_matrix1 / avg_confusion_matrix1.sum(axis=1, keepdims=True)

# Print the results
print("Average Accuracy(1):", avg_accuracy1)
print("Average Precision(1):", avg_precision1)
print("Average Recall(1):", avg_recall1)
print("Average Confusion Matrix(1):")
print(normalized_confusion_matrix1)


print("Total Average Precision:", ((avg_precision1+avg_precision2+avg_precision3+avg_precision4)/4))
print("Total Average Recall:", ((avg_recall1+avg_recall2+avg_recall3+avg_recall4)/4))

total_avg_accuracies = np.mean(total_accuracies)
print("Total Average Accuracy:", total_avg_accuracies)
total_avg_accuracies2 = np.mean(total_accuracies2)
print("Total Average Accuracy (Binary):", total_avg_accuracies2)