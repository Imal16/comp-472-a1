from sklearn import tree
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import output_file_creator

def run(test, train, info, dataset):
    print("Running Base Decision Tree...")
    
    train_x = np.delete(train, -1, 1)
    train_y = train[:,-1]
    
    test_x = np.delete(test, -1, 1)
    test_y = test[:,-1]    
    
    clf = tree.DecisionTreeClassifier()
    y_pred = clf.fit(train_x, train_y).predict(test_x)
    
    print("Done!")


    print('Creating output file...')

    class_labels = np.arange(0, len(info)) 
    confusion_mat = confusion_matrix(test_y, y_pred)
    class_report = classification_report(test_y, y_pred, labels = class_labels, output_dict = True)
    output_file_creator.create_csv('Base-DT', np.arange(1, len(test_x)+1), y_pred, confusion_mat, class_report,dataset)

    print("Done!")
    
    return y_pred
