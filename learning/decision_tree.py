# Load libraries
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split, cross_validate, KFold # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, 'dataS_00_IF_aug_nonzero.csv')

head_ache = pd.read_csv(data_path)
dict = {'Male': 0, 'Female': 1, 'female': 1, 'male':0}
head_ache["Gender"].replace(dict, inplace=True)

last_4_columns = head_ache.iloc[:, -4:]
decimal_values = last_4_columns.apply(lambda row: int(''.join(map(str, [int(bit) for bit in row])), 2), axis=1)
y = pd.DataFrame({'labels': decimal_values})

feature_cols = head_ache.iloc[:, :-4].columns.to_list()

X = head_ache[feature_cols].values
y = y.values


kfold = KFold(n_splits=10, shuffle=True, random_state=1)

for train_index, test_index in kfold.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred, normalize='true'))

# def cross_validation(model, _X, _y, _cv=5):
#       '''Function to perform 5 Folds Cross-Validation
#        Parameters
#        ----------
#       model: Python Class, default=None
#               This is the machine learning algorithm to be used for training.
#       _X: array
#            This is the matrix of features.
#       _y: array
#            This is the target variable.
#       _cv: int, default=5
#           Determines the number of folds for cross-validation.
#        Returns
#        -------
#        The function returns a dictionary containing the metrics 'accuracy', 'precision',
#        'recall', 'f1' for both training set and validation set.
#       '''
#       _scoring = ['accuracy', 'precision', 'recall', 'f1']
#       results = cross_validate(estimator=model,
#                                X=_X,
#                                y=_y,
#                                cv=_cv,
#                                scoring=_scoring,
#                                return_train_score=True,
#                                average='micro')
      
#       return {"Training Accuracy scores": results['train_accuracy'],
#               "Mean Training Accuracy": results['train_accuracy'].mean()*100,
#               "Training Precision scores": results['train_precision'],
#               "Mean Training Precision": results['train_precision'].mean(),
#               "Training Recall scores": results['train_recall'],
#               "Mean Training Recall": results['train_recall'].mean(),
#               "Training F1 scores": results['train_f1'],
#               "Mean Training F1 Score": results['train_f1'].mean(),
#               "Validation Accuracy scores": results['test_accuracy'],
#               "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
#               "Validation Precision scores": results['test_precision'],
#               "Mean Validation Precision": results['test_precision'].mean(),
#               "Validation Recall scores": results['test_recall'],
#               "Mean Validation Recall": results['test_recall'].mean(),
#               "Validation F1 scores": results['test_f1'],
#               "Mean Validation F1 Score": results['test_f1'].mean()
#               }

# decision_tree_model = DecisionTreeClassifier(criterion="entropy", random_state=0)
# decision_tree_result = cross_validation(decision_tree_model, X, y, 5)
# print(decision_tree_model)