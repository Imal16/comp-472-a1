# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:13:27 2020

@author: Ihsaan
Output formatter file
"""
import os
import csv
import numpy as np

def create_csv(filename, test_y, y_pred, confusion_mat, class_report,dataset):
    
    result = np.column_stack((test_y, y_pred))
    file_path ="../result/{}-DS{}.csv".format(filename,dataset)
    
    if os.path.exists(file_path):
        print("Removing old result file")
        os.remove(file_path)
    
    np.savetxt(file_path,result, fmt='%i',delimiter=",")
    
    with open(file_path, 'a') as result_csv:
        csvWriter = csv.writer(result_csv, delimiter = ',')
        csvWriter.writerows(confusion_mat)
    
        for classes in class_report:
             csvWriter.writerow((classes, class_report[classes]))