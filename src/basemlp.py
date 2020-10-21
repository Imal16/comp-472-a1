from sklearn.neural_network import MLPClassifier

#https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
#from sklearn.utils._testing import ignore_warnings
#from sklearn.exceptions import ConvergenceWarning

import numpy as np

#Base MLP is not converging! Code still runs so it's ok. Though it's not exactly optimal.
#@ignore_warnings(category=ConvergenceWarning)
def run(test, train):
    print("Running Base Multi Layer Perceptron...")

    #test_x = np.delete(test, -1, 1)
    #test_y = test[:,-1]

    train_x = np.delete(train, -1, 1)
    train_y = train[:,-1]

    #warnings.filterwarnings("ignore", category=ConvergenceWarning)

    #1 hidden layer of 100 neurons, sigmoid/logistic activation function, stochastic gradient descent, rest is default
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation="logistic", solver="sgd") 
    y_pred = mlp.fit(train_x, train_y).predict(test)

    print("Done!")

    return y_pred
