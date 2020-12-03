import numpy as np
import sklearn
import pandas as pd
import os
import glob
import csv
import features
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Argumetns for the program of similar pair finding')
parser.add_argument('-d', type=str, default="./results/", help='file path to data directory')
parser.add_argument('-f', type=str, default="tiny.txt", help='file name of the dataset. Should be an edgelist')
args = parser.parse_args()

def collect_data(file_path):
    os.chdir(file_path)
    extension = 'csv'
    all_files = [i for i in glob.glob('*_features.{}'.format(extension))]
    print(all_files)
    attributes_calculator = features.FeatureConstructor()
    ordered_attributes_list = list(attributes_calculator.attributes_map.keys())
    ordered_attributes_list.append("class")
    print(ordered_attributes_list)
    df = pd.concat([pd.read_csv(f,sep=",",names = ordered_attributes_list) for f in all_files])
    print(df)
    #df["sum_of_neighbors"].hist(bins=100)
    return df

def split_data(df):
    train, test = train_test_split(df, test_size=0.3)
    dev, test = train_test_split(test, test_size=0.33)
    return train,dev,test

def train_classifier(df):
    return model

def classify(model,df):
    return

if __name__ == "__main__":
    df=collect_data(args.d+args.f)
    train,dev,test=split_data(df)
    model=train_classifier(train)
    classify(model,test)
