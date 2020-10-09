#####By Ihsaan Malek and Olivier Racette####
import sklearn
import csv
import numpy as np
import sys
import argparse
from pathlib import Path


data_folder = Path("dataset/")
possible_algo = ["gnb", "base-dt", "best-dt", "per", "base-mlp", "best-mlp"]


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

#Returns command line arguments object (Namespace)
#To get the value of an argument, just call object.argumentName
#Could also be a dictionary, return vars(parser.parse_args()) instead
def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("-algo", choices=possible_algo, required=True)
    parser.add_argument("-dataset", type=int, choices=[1, 2], required=True)

    #Best-DT arguments
    parser.add_argument("-split", choices=["gini", "entropy"], default="gini", required=False)
    parser.add_argument("-depth", type=int, required=False)
    parser.add_argument("-samples", type=int, required=False)
    parser.add_argument("-impurity", type=int, required=False)
    parser.add_argument("-weight", choices=["none", "balanced"], required=False)

    #Best-MLP arguments
    parser.add_argument("-func", choices=["sigmoid", "tanh", "relu", "identity"], required=False)
    parser.add_argument("-layers", type=int, required=False)
    parser.add_argument("-nodes", required=False)                      #TO BE DETERMINED, haven't learned how multi layer NN works yet
    parser.add_argument("-solver", choices=["adam", "sgd"], required=False)      

    return parser.parse_args()
    

#Runs stuff
def run(args):
    dataset = str(args.dataset)

    print("Reading csv data from dataset " + dataset + "...")

    test_no_label = get_data("test_no_label_" + dataset + ".csv")
    test_with_label = get_data("test_with_label_"+ dataset + ".csv")
    training = get_data("train_" + dataset + ".csv")
    validation = get_data("val_" + dataset + ".csv")
    info = buildSymbolTable("info_" + dataset + ".csv")    

    print("Done!")

    #From here, run the specified training algorithm

if __name__ == "__main__":
    run(getArgs())
