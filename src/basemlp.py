from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import output_file_creator

#https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
#from sklearn.utils._testing import ignore_warnings
#from sklearn.exceptions import ConvergenceWarning

import numpy as np

#Base MLP is not converging! Code still runs so it's ok. Though it's not exactly optimal.
#@ignore_warnings(category=ConvergenceWarning)
def run(test, train, info, dataset):
    print("Running Base Multi Layer Perceptron...")

    test_x = np.delete(test, -1, 1)
    test_y = test[:,-1]

    train_x = np.delete(train, -1, 1)
    train_y = train[:,-1]

    #warnings.filterwarnings("ignore", category=ConvergenceWarning)

    #1 hidden layer of 100 neurons, sigmoid/logistic activation function, stochastic gradient descent, rest is default
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation="logistic", solver="sgd") 
    y_pred = mlp.fit(train_x, train_y).predict(test_x)

    print("Done!")

    print('Creating output file...')

    class_labels = np.arange(0, len(info)) 
    confusion_mat = confusion_matrix(test_y, y_pred)
    class_report = classification_report(test_y, y_pred, labels = class_labels, output_dict = True, zero_division=0)
    output_file_creator.create_csv('Base-MLP', np.arange(1, len(test_x)+1), y_pred, confusion_mat, class_report,dataset)

    print("Done!")

    return y_pred
