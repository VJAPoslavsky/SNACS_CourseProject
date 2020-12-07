import numpy as np
import sklearn
import pandas as pd
import os
import glob
import csv
import features
import argparse
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='Argumetns for the program of similar pair finding')
parser.add_argument('-d', type=str, default="./results/", help='file path to data directory')
parser.add_argument('-f', type=str, default="haggle.txt", help='file name of the dataset. Should be an edgelist')
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
    #print(df)
    #df["sum_of_neighbors"].hist(bins=100)
    return df

def split_data(df):
    train, test = train_test_split(df, test_size=0.3)
    dev, test = train_test_split(test, test_size=0.33)
    return train,dev,test

def train_classifier(df,k):
    X,y=df[df.columns[:-1]],df["class"]
    #model = tree.DecisionTreeClassifier(random_state=42)
    if k==0:
        model=RandomForestClassifier(random_state=42)
    elif k==1:
        model=SVC(random_state=42)
    else:
        model=KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    #text_representation = tree.export_text(model,feature_names=list(X.columns))
    #print(text_representation)
    return model

def classify(model,df):
    X,y=df[df.columns[:-1]],df["class"]
    y_pred=model.predict(X)
    #print(classification_report(y, y_pred, labels=[0,1]))
    ac=accuracy_score(y, y_pred)
    pr=recall_score(y, y_pred)
    f1=f1_score(y, y_pred)
    a=roc_auc_score(y,y_pred)
    return ac,pr,f1,a

if __name__ == "__main__":
    df=collect_data(args.d+args.f)
    train,dev,test=split_data(df)
    params=df.columns[:-1]
    possible_settings=[]
    for r in range(19,len(params)+1):
        for j in itertools.combinations(params,r):
            possible_settings.append(j)
    print("settings to try:",len(possible_settings))
    df_settings=pd.DataFrame(columns=params)
    m=np.array([[c+m for m in ["Precision","Recall","F1","AUC"]] for c in ["RF_","SVM_","KNN_"]]).flatten()
    df_metrics=pd.DataFrame(columns=m)
    c=0
    for i in possible_settings:
        i=list(i)
        i.append("class")
        sets=[1 if j in set(i) else 0 for j in params]
        df_settings.loc[c]=sets
        train_s,test_s=train[i],test[i]
        m=[0 for i in range(len(m))]
        for k in range(3):
            model=train_classifier(train_s,k)
            m[k*4:(k+1)*4]=classify(model,test_s)
        df_metrics.loc[c]=m
        c+=1
    df_results=df_settings.join(df_metrics)
    print(df_results)
