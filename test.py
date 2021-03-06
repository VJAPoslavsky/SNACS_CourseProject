import numpy as np
import argparse
import pandas as pd
import networkx as nx
#k=np.load("split_0_features.npy")
#print(k)

import NetworkCharacteristics
parser = argparse.ArgumentParser(description='Argumetns for the program of similar pair finding')
parser.add_argument('-d', type=str, default="data/", help='file path to data directory')
parser.add_argument('-f', type=str, default="tiny.txt", help='file name of the dataset. Should be an edgelist')
args = parser.parse_args()

#netchars=NetworkCharacteristics.NetworkCharacteristics(graph_path=args.d+args.f, timesplit=1991)
#characteristics = netchars.extract_characteristics(args.f)

#df = pd.read_csv("data\email_dnc.txt",names=["0","1","2"])
#df["weight"] = [1 for i in range(len(df))]
#df=df.reindex(['0','1','weight', '2'],axis=1)
#df.to_csv("data\email_weighted.txt", sep=" ", index=False)
graph_path="data/retweet.txt"
print("Load graph...")
f=open(graph_path)
content=f.read().split("\n")
#content=[e.split(" ") for e in content]
print(content[:10])
graph=nx.read_edgelist(content, delimiter=" ", nodetype=int, data=(('weight',int),('timestamp',int)))
nx.write_gexf(graph,"retweet.gexf")