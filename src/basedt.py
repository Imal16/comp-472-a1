from sklearn import tree
import numpy as np

def run(test, train):
    print("Running Base Decision Tree...")
    
    train_x = np.delete(train, -1, 1)
    train_y = train[:,-1]
    
    #test_x = np.delete(test, -1, 1)
    #test_y = test[:,-1]    
    
    clf = tree.DecisionTreeClassifier(criterion = 'entropy')
    y_pred = clf.fit(train_x, train_y).predict(test)
    
    print("Done!")
    
    return y_pred
