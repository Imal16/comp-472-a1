#####By Ihsaan Malek and Olivier Racette####
import sklearn
import csv
import numpy as np
import sys
from pathlib import Path


data_folder = Path("dataset/")


#Shorthand for numpy's loadtxt method.
#Returns matrices of integers.
#Assumes the file exists. If not, chaos ensues.
def get_data(fileName):
    return np.loadtxt(data_folder / fileName, delimiter=',', dtype='i') 


#numpy's methods to read from files don't seem to work well with the info_x files...
#Returns dictionary where key is index, value is a character
def buildSymbolTable(fileName):
    symbols = {}

    with open(data_folder / fileName) as file:
        reader = csv.DictReader(file)

        for row in reader:
            symbols[int(row['index'])] = row['symbol']
    
    return symbols


#Runs stuff
def run():
    print("Reading csv data...")

    test_no_label_1 = get_data("test_no_label_1.csv")
    test_no_label_2 = get_data("test_no_label_2.csv")
    test_with_label_1 = get_data("test_with_label_1.csv")
    test_with_label_2 = get_data("test_with_label_2.csv")
    train_1 = get_data("train_1.csv") 
    train_2 = get_data("train_2.csv") 
    val_1 = get_data("val_1.csv") 
    val_2 = get_data("val_2.csv")

    #info_1 and info_2 are special cases; seems like numpy loadtxt() and genfromtxt() really don't like to deal with those. 
    #will try out csv reader or dict reader instead.
    info_1 =  buildSymbolTable("info_1.csv")    
    info_2 = buildSymbolTable("info_2.csv")

    print("Done!")


    #From here, run the specified training algorithm

if __name__ == "__main__":
    run()