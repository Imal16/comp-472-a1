from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import output_file_creator

def run(test, train, val, info, dataset):
    
    print("Running Best Decision Tree...")
    print("Running Grid Search to find the Best Decision Tree configuration...")
    
    train_x = np.delete(train, -1, 1)
    train_y = train[:,-1]
    
    test_x = np.delete(test, -1, 1)
    test_y = test[:,-1]

    default_range = np.linspace(0,10,10,dtype=int).astype(float).tolist()
    default_range2 = np.linspace(2,12,10,dtype=int).tolist()
    
    params = {'criterion' : ["gini", "entropy"], 
                  'max_depth': [10, None], 
                  'min_samples_split':default_range2, 
                  'min_impurity_decrease':default_range, 
                  'class_weight':['balanced', None]
                  }

    dt = DecisionTreeClassifier()
    
    clf = GridSearchCV(estimator = dt, param_grid = params, n_jobs=-1)
    y_pred = clf.fit(train_x, train_y).predict(test_x)
    
    print("Best parameters: ", clf.best_params_)
    print("Done!")
    

    print('Creating output file...')

    class_labels = np.arange(0, len(info))
    confusion_mat = confusion_matrix(test_y, y_pred)
    class_report = classification_report(test_y, y_pred, labels = class_labels, output_dict = True)
    output_file_creator.create_csv('Best-DT', np.arange(1, len(test_x)+1), y_pred, confusion_mat, class_report,dataset)
    
    print("Done!")


    return y_pred
