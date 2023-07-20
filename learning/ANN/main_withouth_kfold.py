import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow import keras
from keras.models import Sequential
from keras.callbacks import History
from keras.layers import Dense, Dropout
import os
from keras.layers import LeakyReLU
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from keras import callbacks
from sklearn.preprocessing import MultiLabelBinarizer
import keras.backend as K


def define_model():
    model = Sequential()
    model.add(Dense(65, input_shape=(57,), activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(65, activation='tanh'))
    model.add(Dense(65, activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(8, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(4, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.BinaryAccuracy(threshold=.5), keras.metrics.Precision(), keras.metrics.Recall()])
    return model


def log_model_details(filepath):
    with open(filepath, 'w') as file:
        model = define_model()
        model.summary(print_fn=lambda x: file.write(x + '\n'))

        file.write('\nLayer Details:\n')
        for layer in model.layers:
            file.write(f'Layer: {layer.name}\n')
            file.write(f'Layer Size: {layer.input_shape[1]}\n')
            if not isinstance(layer, Dropout):
                file.write(f'Activation Function: {layer.activation.__name__}\n')
            if isinstance(layer, Dropout):
                file.write(f'Dropout Rate: {layer.rate}\n')
            file.write('\n')

def evaluate_model(trainX, trainY, testX, testY,filepath, n_folds=5, epochs=8):
    train_accs, test_accs, val_accs = list(), list(), list()
    test_losses, val_losses = list(), list()
    train_precs, test_precs, val_precs = list(), list(), list()
    train_recs, test_recs, val_recs = list(), list(), list()
    hand_test_acc_low, hand_test_acc_high = list(), list() 



    
    model = define_model()
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=32, validation_data=(testX, testY), verbose=0)
    _, train_acc, train_prec, train_rec = model.evaluate(trainX, trainY, verbose=0)
    _, test_acc, test_prec, test_rec = model.evaluate(testX, testY, verbose=0)
    _, val_acc, val_prec, val_rec = model.evaluate(testX, testY, verbose=0)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    val_accs.append(val_acc)

    train_precs.append(train_prec)
    test_precs.append(test_prec)
    val_precs.append(val_prec)

    train_recs.append(train_rec)
    test_recs.append(test_rec)
    val_recs.append(val_rec)

    test_loss = history.history['val_loss'][-1]
    val_loss = history.history['loss'][-1]
    test_losses.append(test_loss)
    val_losses.append(val_loss)
        
    print(model.evaluate(testX, testY, verbose=0, return_dict=True))
 
    with open(filepath, 'a') as file:
        file.write(f"Train Accuracy: {train_acc}\n")
        file.write(f"Test Accuracy: {test_acc}\n")
        file.write(f"Validation Accuracy: {val_acc}\n")
        file.write(f"Test Loss: {test_loss}\n")
        file.write(f"Validation Loss: {val_loss}\n\n")
    print(testX)
    y_prediction = model.predict(testX)
    print(y_prediction)
    print(type(y_prediction))
    y_prediction[y_prediction>= 0.5] = int(1)
    y_prediction[y_prediction< 0.5] = int(0)
    print(y_prediction)
    print(type(testY))
    correct_low = 0
    correct_high = 0
    for i in range(len(y_prediction)):
        if testY[i][0] == y_prediction[i][0] and testY[i][1] == y_prediction[i][1] and testY[i][2] == y_prediction[i][2] and testY[i][3] == y_prediction[i][3]:
            correct_low += 1
        for k in range(4):
            if testY[i][k] == y_prediction[i][k]:
                correct_high += 1
    hand_calc_acc_low = correct_low/(len(y_prediction))
    hand_calc_acc_high = correct_high/(len(y_prediction)*4)

    print(hand_calc_acc_low)
    print(hand_calc_acc_high)
    hand_test_acc_high.append(hand_calc_acc_high)
    hand_test_acc_low.append(hand_calc_acc_low)
    print(testY)
    #Create confusion matrix and normalizes it over predicted (columns)
    result1 = confusion_matrix(testY.argmax(axis=1),y_prediction.argmax(axis=1))
    print(result1)
    result = confusion_matrix(testY.argmax(axis=1),y_prediction.argmax(axis=1) , normalize='true')
    print(result)
    return train_accs, test_accs, val_accs, test_losses, val_losses, train_precs, test_precs, val_precs, train_recs, test_recs, val_recs, model, hand_test_acc_high, hand_test_acc_low

def summarize_performance(train_accs, test_accs, val_accs, test_losses, val_losses, train_precs, test_precs, val_precs, train_recs, test_recs, val_recs, hand_test_acc_high, hand_test_acc_low):
    print('Train Accuracy: mean=%.3f std=%.3f' % (np.mean(train_accs) * 100, np.std(train_accs) * 100))
    print('Test Accuracy: mean=%.3f std=%.3f' % (np.mean(test_accs) * 100, np.std(test_accs) * 100))
    print('Validation Accuracy: mean=%.3f std=%.3f' % (np.mean(val_accs) * 100, np.std(val_accs) * 100))
    print('Test Loss: mean=%.3f std=%.3f' % (np.mean(test_losses), np.std(test_losses)))
    print('Validation Loss: mean=%.3f std=%.3f' % (np.mean(val_losses), np.std(val_losses)))
    print('Train Precision: mean=%.3f std=%.3f' % (np.mean(train_precs), np.std(train_precs)))
    print('Test Precision: mean=%.3f std=%.3f' % (np.mean(test_precs), np.std(test_precs)))
    print('Validation Precision: mean=%.3f std=%.3f' % (np.mean(val_precs), np.std(val_precs)))
    print('Train Recall: mean=%.3f std=%.3f' % (np.mean(train_recs), np.std(train_recs)))
    print('Test Recall: mean=%.3f std=%.3f' % (np.mean(test_recs), np.std(test_recs)))
    print('Validation Recall: mean=%.3f std=%.3f' % (np.mean(val_recs), np.std(val_recs)))
    print('Test Accuracy(All-labels): mean=%.3f std=%.3f' % (np.mean(hand_test_acc_low), np.std(hand_test_acc_low)))
    print('Test Accuracy(Binary Accuracy): mean=%.3f std=%.3f' % (np.mean(hand_test_acc_high), np.std(hand_test_acc_high)))

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, 'dataS_05_IF_aug_nonzero.csv')
print(data_path)
data = pd.read_csv(data_path)
dict = {'Male': 0, 'Female': 1, 'female': 1, 'male': 0}
data["Gender"].replace(dict, inplace=True)

last_4_columns = data.iloc[:, -4:]
y = last_4_columns.values
label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform(y)

data.drop(columns=data.columns[0], axis=1, inplace=True)

data['Age'] = data['Age'].astype('int')

epoch_values = [8]  # Specify the epoch values to try

# Assuming your dataset is stored in a DataFrame called 'df'
# where the last 4 columns are the labels and the remaining 57 columns are features.


# Assuming your dataset is stored in a DataFrame called 'df'
# where the last 4 columns are the labels and the remaining 57 columns are features.

# Shuffle the dataset randomly
data_shuffled = data.sample(frac=1, random_state=42)

# Separate the label columns from the features
labels = data_shuffled.iloc[:, -4:]
features = data_shuffled.iloc[:, :-4]

# Initialize empty DataFrames for test and train datasets
test_data = pd.DataFrame()
train_data = pd.DataFrame()

# Iterate over unique label combinations
for _, group in labels.groupby(labels.columns.tolist()):
    # Select the first 2 examples for each unique label combination as test data
    test_data = pd.concat([test_data, features.loc[group.head(2).index]])
    # Exclude the selected examples from the shuffled dataset
    features = features.drop(group.head(2).index)
    labels = labels.drop(group.head(2).index)
    
    # Break the loop if the number of test data points reaches 200
    if len(test_data) >= 200:
        break

# Fill the remaining test data points from the shuffled dataset
remaining_test_points = 200 - len(test_data)
test_data = pd.concat([test_data, features.head(remaining_test_points)])
features = features.tail(features.shape[0] - remaining_test_points)
labels = labels.tail(labels.shape[0] - remaining_test_points)

# Select the remaining examples as train data
train_data = features

# Drop the first column from train and test features
train_features = train_data.iloc[:, :].values
train_labels = labels.values
test_features = test_data.iloc[:, :].values
test_labels = test_data.iloc[:, -4:].values
print(test_labels)
# Reset the index for the test data
test_data.reset_index(drop=True, inplace=True)




for epoch in epoch_values:
     print(f"Epoch: {epoch}")
     log_file_path = os.path.join(THIS_DIR, f'2model_details_epoch_{epoch}.txt')
     log_model_details(log_file_path)
     log_results_path = os.path.join(THIS_DIR, f'2model_results_epoch_{epoch}.txt')
     train_accs, test_accs, val_accs, test_losses, val_losses, train_precs, test_precs, val_precs, train_recs, test_recs, val_recs, model, hand_test_acc_high, hand_test_acc_low = evaluate_model(train_features, train_labels, test_features, test_labels , epochs=epoch, filepath=log_results_path)
     summarize_performance(train_accs, test_accs, val_accs, test_losses, val_losses, train_precs, test_precs, val_precs, train_recs, test_recs, val_recs, hand_test_acc_high, hand_test_acc_low)
     print()
# #model.save("model.h5")
