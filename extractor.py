#execution script for extracting the features from a given graph
import os
import random
import networkx as nx
import networkx.algorithms.community as nxcom
import numpy as np
import features
import urllib.request
import csv
import features
import itertools
import multiprocessing
from multiprocessing import Process, Pool
from joblib import Parallel, delayed, parallel_backend
import time
import argparse
import matplotlib.pyplot as plt
import NetworkCharacteristics

parser = argparse.ArgumentParser(description='Argumetns for the program of similar pair finding')
parser.add_argument('-d', type=str, default="data/", help='file path to data directory')
parser.add_argument('-f', type=str, default="tiny.txt", help='file name of the dataset. Should be an edgelist')
args = parser.parse_args()

class Extractor:
    def __init__(self, graph_path="", graph=None):
        if graph_path!="":
            print("Load graph...")
            f=open(graph_path)
            content=f.read().split("\n")
            #content=[e.split(" ") for e in content]
            print(content[:10])
            graph=nx.read_edgelist(content, delimiter=" ", nodetype=int, data=(('weight',int),('timestamp',int)))
        self.graph=graph.to_undirected()
        self.train_graph=self.graph.copy()
        self.nodes=list(self.graph.nodes())
        self.edges=list(self.graph.edges())
        self.train_edges=[]
        self.test_edges=[]
        self.p_edges=list(itertools.combinations(self.nodes, 2))
        self.p_label={e:0 for e in self.p_edges}
        self.attributes_calculator=None
        self.attribute_name="timestamp"
        #plt.hist(np.array(list(self.graph.edges(data=self.attribute_name))),100)
        #plt.show()
        self.timestamps=np.array(list(self.graph.edges(data=self.attribute_name)))
        self.timesplit=np.percentile(self.timestamps,80)
        self.page_rank=None
        print("Finished loading")


    def set_node_community(self, G, communities):
        for c, v_c in enumerate(communities):
            for v in v_c:
                # Add 1 to save 0 for external edges
                G.nodes[v]['community'] = c + 1

    def get_features_inner(self,inp):
        s=inp[0]
        p_edges=inp[1]
        if not os.path.exists(f'results/{args.f}'):
            os.makedirs(f'results/{args.f}')
            #os.makedirs(f'results/{args.f}/positive')
        f=open(f'results/{args.f}/split_{s}_features.csv', 'w+')
        f.close()
        #f=open(f'results/{args.f}/positive/split_{s}_features.csv', 'w+')
        #f.close()
        attributes_calculator = features.FeatureConstructor(self.train_graph,self.page_rank)
        attributes_list={}
        if attributes_list == {}:
                    ordered_attributes_list = attributes_calculator.attributes_map.keys()
                    for attribute in ordered_attributes_list:
                        attributes_list[attribute] = {}
        line = 0
        for pair in p_edges:
            if pair in self.test_edges:
                n1, n2  = pair
                attributes_calculator.set_nodes(n1, n2)
                column_values=np.zeros(len(ordered_attributes_list)+1)
                fet=attributes_calculator.get_features(pair)
                column_values[:-1]=fet
                #column_values[-3] = n1
                #column_values[-2] = n2
                column_values[-1] = 1#self.p_label[pair]
                line += 1
                with open(f'results/{args.f}/split_{s}_features.csv', 'a+') as file:
                    np.savetxt(file, [column_values], delimiter=",",fmt='%f')
                    file.close()
            elif pair in self.train_edges:
                continue
            else:
                n1, n2  = pair
                attributes_calculator.set_nodes(n1, n2)
                column_values=np.zeros(len(ordered_attributes_list)+1)
                fet=attributes_calculator.get_features(pair)
                column_values[:-1]=fet
                #column_values[-3] = n1
                #column_values[-2] = n2
                column_values[-1] = 0#self.p_label[pair]
                line += 1
                with open(f'results/{args.f}/split_{s}_features.csv', 'a+') as file:
                    np.savetxt(file, [column_values], delimiter=",",fmt='%f')
                    file.close()
        return 1

    def get_node_features(self):
        num_cores = multiprocessing.cpu_count()#//2
        #num_cores = 1
        #self.p_edges=self.p_edges[:5000]
        k=len(self.p_edges)//num_cores
        splits=[[i,self.p_edges[i*k:(i+1)*k]] if i<num_cores-1 else [i,self.p_edges[i*k:]] for i in range(num_cores)]
        print("starting computing metrics...")
        #parallel execution; dump resuls in text file
        t=time.time()#n_jobs=num_cores,prefer="threads"
        with parallel_backend('loky', n_jobs=num_cores):
            Parallel()(delayed(self.get_features_inner)(s) for s in splits)
        print("time taken:",time.time()-t)
        print("done computing metrics")

#        node_feature_dataset=np.zeros(len(self.p_edges),len(ordered_attributes_list))
#        line = 0
#        for pair in self.p_edges:
#            column=0
#            n1, n2  = pair
#            attributes_calculator.set_nodes(n1, n2)
#            for function in ordered_attributes_list:
#                parameters = attributes_list[function]
#                node_feature_dataset[line][column] = attributes_calculator.attributes_map[function](**parameters)
#                node_feature_dataset[line][-1] = 0
#                column+=1
#            line+=1
#        return node_feature_dataset
        return 1

    def get_graph_characterisitcs(self):
        return

    def sample(self,attribute_name="timestamp",split_date=0):
        if split_date==0:
            split_date=self.timesplit
        self.train_edges = [(x,y) for x,y,t in self.graph.edges(data=attribute_name) if t<=split_date]
        self.test_edges = [(x,y) for x,y,t in self.graph.edges(data=attribute_name) if t>split_date]
        self.train_graph.remove_edges_from(self.test_edges)
        print("Length train set",len(self.train_edges))
        print("Length test set",len(self.test_edges))
        for i in self.train_edges:
            self.p_label[i]=1
        for i in self.test_edges:
            self.p_label[i]=1
        return self.train_edges, self.test_edges

if __name__ == "__main__":
    extractor=Extractor(args.d+args.f)
    train, test=extractor.sample()
    extractor.page_rank=nx.pagerank_numpy(extractor.train_graph,weight="weight")
    netchars_train=NetworkCharacteristics.NetworkCharacteristics(graph=extractor.train_graph,timesplit=extractor.timesplit)
    #netchars_full=NetworkCharacteristics.NetworkCharacteristics(graph=extractor.graph,timesplit=np.max(extractor.timestamps))
    train_char=netchars_train.extract_characteristics(args.f)
    #full_char=netchars_full.extract_characteristics(args.f)
    #extractor.communities = nxcom.greedy_modularity_communities(extractor.train_graph)
    #extractor.set_node_community(extractor.train_graph,extractor.communities)
    #print(extractor.page_rank)
    print("computed characteristics")
    node_feature_dataset=extractor.get_node_features()
    print("Finished")