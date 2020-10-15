from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import output_file_creator

def run(test, train, val, dataset):
    
    print("Running Best Decision Tree...")
    print("Running Grid Search to find the Best Decision Tree configuration...")
    
    if dataset == '1':
        class_labels = np.arange(0,27)
    else:
        class_labels = np.arange(0,10)
    
    train_x = np.delete(train, -1, 1)
    train_y = train[:,-1]
    
    test_x = np.delete(test, -1, 1)
    test_y = test[:,-1]
    
# =============================================================================
#     if depth == '0':
#         depth = None
#     else: depth = int(depth)
# =============================================================================
    default_range = np.linspace(0,10,10,dtype=int).astype(float).tolist()
    
    
    params = {'criterion' : ["gini", "entropy"], 
                  'max_depth': [10, None], 
                  'min_samples_split':default_range, 
                  'min_impurity_decrease':default_range, 
                  'class_weight':['balanced', None]
                  }
    
    #clf = DecisionTreeClassifier(criterion=split, max_depth=depth,min_samples_split=samples,min_impurity_decrease = impurity, class_weight = weight)
    dt = DecisionTreeClassifier()
    
    clf = GridSearchCV(estimator = dt, param_grid = params)
    print(clf)
    print("Best parameters: ", clf.best_params_)
    y_pred = clf.predict(test_x)
    
    
    confusion_mat = confusion_matrix(test_y, y_pred)
    class_report = classification_report(test_y, y_pred, labels = class_labels, output_dict = True)
    
    print('creating output file')
    output_file_creator.create_csv('Best-DT', test_y, y_pred, confusion_mat, class_report,dataset)
    
    
    
    
    
    print("Done!")

    return 0
