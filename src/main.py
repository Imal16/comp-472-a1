#####By Ihsaan Malek and Olivier Racette####

#External dependencies
#import sklearn    #might be better to import specific things in each file
import numpy as np

#Default modules
import csv
import sys
import argparse
from pathlib import Path

#Algorithms
import basedt, basemlp, bestdt, bestmlp, gnb, per


#potential globals? will be moved if needed
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

    parser.add_argument("algo", choices=possible_algo, help="Algorithm to be run. See choices list.")
    parser.add_argument("dataset", type=int, choices=[1, 2], help="Dataset number.")

    #Best-DT and Best-MLP require several more arguments. We will probably have to figure out some defaults as we go along.

    #Best-DT arguments
    parser.add_argument("-split", choices=["gini", "entropy"])
    parser.add_argument("-depth", type=int)
    parser.add_argument("-samples", type=int)
    parser.add_argument("-impurity", type=int)
    parser.add_argument("-weight", choices=["none", "balanced"])

    #Best-MLP arguments
    #parser.add_argument("-func", choices=["logistic", "tanh", "relu", "identity"])
    #parser.add_argument("-layers", type=int)                    #don't really need -layers if we can build a list of ndoes per layer. num of layer = len(list_of_nodes)
    parser.add_argument("-narch1", nargs="+", type=int)          #with nargs, user can enter multiple arguments for 1 option. '+' means 1 or more arguments. this builds a list of integers.
    parser.add_argument("-narch2", nargs="+", type=int)
    #parser.add_argument("-solver", choices=["adam", "sgd"])      

    return parser.parse_args()
    

#Runs stuff
def run():
    args = getArgs()
    dataset = str(args.dataset)

    print("Reading csv data from dataset " + dataset + "...")

    test_no_label = get_data("test_no_label_" + dataset + ".csv")
    test_with_label = get_data("test_with_label_"+ dataset + ".csv")
    training = get_data("train_" + dataset + ".csv")
    validation = get_data("val_" + dataset + ".csv")
    info = buildSymbolTable("info_" + dataset + ".csv")    

    print("Done!")

    #forgot switch statements don't exist in python. bunch of if elif coming soon.
    #theres probably a better way to do this.
    #does validation need to be passed? seems like its not really used here. 

    result = 0

    if args.algo == possible_algo[0]:
        result = gnb.run(test_with_label, training, validation)
    elif args.algo == possible_algo[1]:
        result = basedt.run(test_with_label, training, validation)
    elif args.algo == possible_algo[2]:
        result = bestdt.run(test_with_label, training, validation, args.split, args.depth, args.samples, args.impurity, args.weight)
    elif args.algo == possible_algo[3]:
        result = per.run(test_with_label, training, validation)
    elif args.algo == possible_algo[4]:
        result = basemlp.run(test_with_label, training, validation)
    elif args.algo == possible_algo[5]:
        result = bestmlp.run(test_with_label, training, validation, args.narch1, args.narch2) 

    #should the algorithms return predictions? and from here (this file) compute the rest of results, since its the same computation for every algo?
    analyze(test_with_label, validation, result)


#Seems like every algorihtm returns a similar result, prediction of y values. we can probably use the same method to analyze results of each
def analyze(test, validation, prediction):
    return 0


if __name__ == "__main__":
    run()
