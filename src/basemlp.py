from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

#https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
#from sklearn.utils._testing import ignore_warnings
#from sklearn.exceptions import ConvergenceWarning

import numpy as np

#Base MLP is not converging! Need to supress warnings to allow code to continue to execute.
#@ignore_warnings(category=ConvergenceWarning)
def run(test, train, val):
    print("Running Base Multi Layer Perceptron...")

    test_x = np.delete(test, -1, 1)
    test_y = test[:,-1]                 #test_y isnt used, so test_without_label should be passed?

    train_x = np.delete(train, -1, 1)
    train_y = train[:,-1]

    #warnings.filterwarnings("ignore", category=ConvergenceWarning)

    #1 hidden layer of 100 neurons, sigmoid/logistic activation function, stochastic gradient descent, rest is default
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation="logistic", solver="sgd") 
    pred_y = mlp.fit(train_x, train_y).predict(test_x)

    print("Done!")

    #print(confusion_matrix(test_y, pred_y))
    #print(classification_report(test_y, pred_y))

    return pred_y
