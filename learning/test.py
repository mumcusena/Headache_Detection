#Import the necessary libraries
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#Create the NumPy array for actual and predicted labels.
actual = np.array(
['Dog','Dog','Dog','Not Dog','Dog','Not Dog','Dog','Dog','Not Dog','Not Dog'])
predicted = np.array(
['Dog','Not Dog','Dog','Not Dog','Dog','Dog','Dog','Dog','Not Dog','Not Dog'])

#compute the confusion matrix.
cm = confusion_matrix(actual,predicted)
cm=cm/cm.sum(axis=1, keepdims=True)
print(cm)
