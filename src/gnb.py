from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import output_file_creator

#Runs Gaussian Naives Bayes with default parameters
def run(test, train, info, dataset):
    print("Running Gaussian Naive Bayes...")

    #datasets need to be split up between x (features) and y (label)
    #need to populate train_x, train_y, test_x, test_y, preferrably using numpy for performance purposes

    test_x = np.delete(test, -1, 1)
    test_y = test[:,-1]

    train_x = np.delete(train, -1, 1)
    train_y = train[:,-1]

    gnb = GaussianNB()
    y_pred = gnb.fit(train_x, train_y).predict(test_x)
    
    print("Done!")


    print('Creating output file...')

    class_labels = np.arange(0, len(info))
    confusion_mat = confusion_matrix(test_y, y_pred)
    class_report = classification_report(test_y, y_pred, labels = class_labels, output_dict = True)
    output_file_creator.create_csv('GNB', np.arange(1, len(test_x)+1), y_pred, confusion_mat, class_report,dataset)

    print("Done!")
 

    return y_pred