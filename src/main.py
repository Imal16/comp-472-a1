#####By Ihsaan Malek and Olivier Racette####

#External dependencies
#import sklearn    #might be better to import specific things in each file
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

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
#Returns a numpy array with each symbol at the assigned index.
def buildSymbolTable(fileName):
    symbols = []

    with open(data_folder / fileName) as file:
        reader = csv.DictReader(file)

        for row in reader:
            symbols.insert(int(row['index']), row['symbol'])
    
    return np.array(symbols)


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
    parser.add_argument("-func", choices=["sigmoid", "tanh", "relu", "identity"])
    parser.add_argument("-layers", type=int)
    parser.add_argument("-nodes")                      #TO BE DETERMINED, haven't learned how multi layer NN works yet
    parser.add_argument("-solver", choices=["adam", "sgd"])      

    return parser.parse_args()


#Plots distribution of the number of instances of each class
#Assumes the label is present at the end of the data matrix!!
def calc_distrib(data, info):
    distr = np.zeros(len(info), dtype=int)

    for i in data:
        distr[i[-1]] += 1

    return distr


def plot_distrib(distrib, info, title):
    fig, ax = plt.subplots(1,1)

    x = np.arange(len(distrib))
    ax.bar(x, distrib, align="center")

    ax.xaxis.set_major_locator(plt.FixedLocator(x))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(info))

    plt.title(title)
    plt.xlabel("Symbol")
    plt.ylabel("Frequency")

    plt.show()


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


    print("Calculating data distribution...")
    train_distrib = calc_distrib(training, info)
    valid_distrib = calc_distrib(validation, info)
    test_distrib = calc_distrib(test_with_label, info)
    print("Done!")

    print("Plotting...")

    plot_distrib(train_distrib, info, "Training Distribution")
    plot_distrib(valid_distrib, info, "Validation Distribution")
    plot_distrib(test_distrib, info, "Test Distribution (Real)")

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
        result = bestmlp.run(test_with_label, training, validation, args.func, args.layers, args.nodes, args.solver)

    #should the algorithms return predictions? and from here (this file) compute the rest of results, since its the same computation for every algo?
    analyze(test_with_label, validation, result)


#Seems like every algorihtm returns a similar result, prediction of y values. we can probably use the same method to analyze results of each
def analyze(test, validation, prediction):
    return 0


if __name__ == "__main__":
    run()
