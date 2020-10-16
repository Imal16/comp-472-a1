from sklearn.naive_bayes import GaussianNB
import numpy as np

#Runs Gaussian Naives Bayes with default parameters
def run(test, train, val):
    print("Running Gaussian Naive Bayes...")

    #datasets need to be split up between x (features) and y (label)
    #need to populate train_x, train_y, test_x, test_y, preferrably using numpy for performance purposes

    test_x = np.delete(test, -1, 1)
    test_y = test[:,-1]

    train_x = np.delete(train, -1, 1)
    train_y = train[:,-1]

    gnb = GaussianNB()
    y_pred = gnb.fit(train_x, train_y).predict(test_x)

    #Next up, need to:
    #plot confusion matrix
    #precision, recall and f1 measure for each class
    #accuracy, macro average f1 and weighted-average f1 of the model

    print("Done!")

    return y_pred