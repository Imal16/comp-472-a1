from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import output_file_creator


def run(test, train, val, info, dataset, narch1, narch2):
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

    print("Best parameters found: ", clf.best_params_)
    print("Done!")

    print('Creating output file...')

    class_labels = np.arange(0, len(info))
    confusion_mat = confusion_matrix(test_y, y_pred)
    class_report = classification_report(test_y, y_pred, labels = class_labels, output_dict = True)
    output_file_creator.create_csv('Best-MLP', np.arange(1, len(test_x)+1), y_pred, confusion_mat, class_report,dataset)

    print("Done!")
    

    return y_pred
