import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer

def define_model():
    model = Sequential()
    model.add(Dense(65, input_shape=(57,), activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(65, activation='tanh'))
    model.add(Dense(65, activation='tanh'))
    model.add(Dense(65, activation='tanh'))
    model.add(Dense(65, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(4, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=6):
    scores, histories = list(), list()

    #prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    #enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        model = define_model()
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc*100))

        scores.append(acc)
        histories.append(history)
    return scores, histories

# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))
	# box and whisker plots of results
	plt.boxplot(scores)
	plt.show()


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, 'dataS_05_IF_aug_nonzero.csv')

data = pd.read_csv(data_path)
dict = {'Male': 0, 'Female': 1, 'female':1, 'male':0}
data["Gender"].replace(dict, inplace=True)

last_4_columns = data.iloc[:, -4:]
y = last_4_columns.values
label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform(y)

data.drop(columns=data.columns[0], axis=1, inplace=True)
feature_cols = data.iloc[:, : -4].columns.to_list()

# for i in range(17):
#     feature_cols.remove("trigger_food_{}".format(i+1))
# print(len(feature_cols))

X = data[feature_cols].values

scores, histories = evaluate_model(X, y)
summarize_performance(scores)

# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

# # later...
 
# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
 
# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))