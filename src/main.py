#####By Ihsaan Malek and Olivier Racette####
import sklearn
import csv
import numpy as np
from pathlib import Path


#Kinda ugly but idk if we have a better to do this
data_folder = Path("dataset/")
info_1 = data_folder / "info_1.csv"     #info_1 and info_2 are special cases since they contain headers. That row will need to be skipped.
info_2 = data_folder / "info_2.csv"
test_no_label_1 = data_folder / "test_no_label_1.csv"
test_no_label_2 = data_folder / "test_no_label_2.csv"
test_with_label_1 = data_folder / "test_with_label_1.csv"
test_with_label_2 = data_folder / "test_with_label_2.csv"
train_1 = data_folder / "train_1.csv"
train_2 = data_folder / "train_2.csv"
val_1 = data_folder / "val_1.csv"
val_2 = data_folder / "val_2.csv"


#Returns a numpy array comprised of the data within the passed csv file
#Useless now that I found np.genfromtxt() LOL
def get_data(csv_file):
    contents = []

    with open(csv_file) as file:
        reader = csv.reader(file)

        for row in reader:
            contents.append(row)

    return np.array(contents, dtype=np.int32)


#Runs stuff
def run():
    print(get_data(test_no_label_1))
    print(np.genfromtxt(test_no_label_1, delimiter=',', dtype=np.int32))


if __name__ == "__main__":
    run()