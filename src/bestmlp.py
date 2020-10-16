from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np


def run(test, train, narch1, narch2):
    print("Running Grid Search to find the Best Multi Layer Perceptron configuration...")

    print("Network architecture 1:")
    print("\tNumber of hidden layers: " + str(len(narch1)))

    for i in range(len(narch1)):
        print("\tNumber of nodes in hidden layer #" + str(i+1) + ": " + str(narch1[i]))


    print("Network architecture 2:")
    print("\tNumber of hidden layers: " + str(len(narch2)))

    for i in range(len(narch2)):
        print("\tNumber of nodes in hidden layer #" + str(i+1) + ": " + str(narch2[i]))

    test_x = np.delete(test, -1, 1)
    test_y = test[:,-1]

    train_x = np.delete(train, -1, 1)
    train_y = train[:,-1]

    mlp = MLPClassifier()

    params = {
        "hidden_layer_sizes": [narch1,narch2],
        "activation": ["logistic", "relu", "tanh", "identity"],
        "solver": ["adam", "sgd"]
    }

    clf = GridSearchCV(mlp, params, n_jobs=-1)
    y_pred = clf.fit(train_x, train_y).predict(test_x)

    print("Best parameters: ", clf.best_params_)
    print("Done!")

    return test_y, y_pred
