from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

def run(test, train):
    
    print("Running Best Decision Tree...")
    print("Running Grid Search to find the Best Decision Tree configuration...")
    
    train_x = np.delete(train, -1, 1)
    train_y = train[:,-1]
    
    #test_x = np.delete(test, -1, 1)
    #test_y = test[:,-1]

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
    y_pred = clf.fit(train_x, train_y).predict(test)
    
    print("Best parameters: ", clf.best_params_)
    print("Done!")

    return y_pred
