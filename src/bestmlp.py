from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

import numpy as np
import time

#def run(test, train, val, func, layers, nodes, solver):
def run(test, train, val, narch1, narch2):
    print("Running Grid Search to find the Best Multi Layer Perceptron configuration...")

    print("Network architecture 1:")
    print("\tNumber of hidden layers: " + str(len(narch1)))

    for i in range(len(narch1)):
        print("\tNumber of nodes in hidden layer #" + str(i+1) + ": " + str(narch1[i]))


    print("Network architecture 2:")
    print("\tNumber of hidden layers: " + str(len(narch1)))

    for i in range(len(narch2)):
        print("\tNumber of nodes in hidden layer #" + str(i+1) + ": " + str(narch2[i]))

    test_x = np.delete(test, -1, 1)
    test_y = test[:,-1]                 #test_y isnt used, so test_without_label should be passed?

    train_x = np.delete(train, -1, 1)
    train_y = train[:,-1]

    mlp = MLPClassifier()

    params = {
        "hidden_layer_sizes": [narch1,narch2],
        "activation": ["logistic", "relu", "tanh", "identity"],
        "solver": ["adam", "sgd"]
    }

    start_time = time.time()

    clf = GridSearchCV(mlp, params, n_jobs=-1)
    pred_y = clf.fit(train_x, train_y).predict(test_x)

    print("Done!\nExecution time: " + str(round(time.time() - start_time, 2)) + " seconds.")

    print("Best parameters: ", clf.best_params_)

    return pred_y
