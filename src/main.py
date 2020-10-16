#####By Ihsaan Malek and Olivier Racette####

#External dependencies
#import sklearn    #might be better to import specific things in each file
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


#Default modules
import csv
import sys
import argparse
import random
from pathlib import Path

#Algorithms
import basedt, basemlp, bestdt, bestmlp, gnb, per


#potential globals? will be moved if needed
data_folder = Path("../dataset/")
possible_algo = ["gnb", "base-dt", "best-dt", "per", "base-mlp", "best-mlp"]
num_vis_samples = 5

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

    #Best-MLP arguments
    parser.add_argument("-narch1", nargs="+", type=int)          #with nargs, user can enter multiple arguments for 1 option. '+' means 1 or more arguments. this builds a list of integers.
    parser.add_argument("-narch2", nargs="+", type=int)     

    #For visualization
    parser.add_argument("-visual", help="show data visualization", action="store_true") 

    return parser.parse_args()


#Plots distribution of the number of instances of each class
#Assumes the label is present at the end of the data matrix!!
def calc_distrib(data, info):
    distr = np.zeros(len(info), dtype=int)

    for i in data:
        distr[i[-1]] += 1

    return distr



#Plots the distribution of training, validation and test in a 3 bar graph
def plot_distrib(train_distrib, val_distrib, test_distrib, info):
    fig, ax = plt.subplots()
    width = 0.25

    x = np.arange(len(info))

    ax.bar(x - width, train_distrib, width, label="Trainig")
    ax.bar(x, val_distrib, width, label="Validation")
    ax.bar(x + width, test_distrib, width, label="Test")


    #setting custom labels: put symbols (user input) instead of numbers on the x avis
    ax.xaxis.set_major_locator(plt.FixedLocator(x))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(info))


    #plot configuration
    plt.title("Distribution of Instances")
    plt.xlabel("Symbol")
    plt.ylabel("Frequency")
    ax.legend()
    ax.grid(True, axis='y', alpha=0.35)

    plt.show()


#Shows the character a sample is supposed to represent.
#Assumes the sample is an array of length 1024
def visualize_sample(sample, real, predict):
    plt.matshow(sample.reshape(32 ,32), cmap=ListedColormap(['k', 'w']))
    plt.text(-6, -3, "Prediction: " + predict + "\nReal: " + real)
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


    if args.visual:
        print("Plotting...")
        plot_distrib(train_distrib, valid_distrib, test_distrib, info)


    #forgot switch statements don't exist in python. bunch of if elif coming soon.
    #theres probably a better way to do this.
    #does validation need to be passed? seems like its not really used here. 

    result = 0

    if args.algo == possible_algo[0]:
        result = gnb.run(test_with_label, training, validation)
    elif args.algo == possible_algo[1]:
        result = basedt.run(test_with_label, training, validation, dataset)
    elif args.algo == possible_algo[2]:
        result = bestdt.run(test_with_label, training, validation,dataset)
    elif args.algo == possible_algo[3]:
        result = per.run(test_with_label, training, validation, dataset)
    elif args.algo == possible_algo[4]:
        result = basemlp.run(test_with_label, training, validation)
    elif args.algo == possible_algo[5]:
        result = bestmlp.run(test_with_label, training, validation, args.narch1, args.narch2)

    if args.visual:
        for i in range(num_vis_samples):
            index = random.randint(0, len(result)-1)
            visualize_sample(test_no_label[index], info[test_with_label[index][-1]], info[result[index]])       #holy moly brackets!

    #should the algorithms return predictions? and from here (this file) compute the rest of results, since its the same computation for every algo?
    analyze(test_with_label, validation, result)


#Seems like every algorihtm returns a similar result, prediction of y values. we can probably use the same method to analyze results of each
def analyze(test, validation, prediction):
    return 0


if __name__ == "__main__":
    run()
