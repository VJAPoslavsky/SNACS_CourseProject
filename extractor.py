#execution script for extracting the features from a given graph
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
from joblib import Parallel, delayed

class Extractor:
    def __init__(self, graph_path="", graph=None):
        if graph_path!="":
            print("Load graph...")
            f=open(graph_path)
            content=f.read().split("\n")
            #content=[e.split(" ") for e in content]
            print(content[:10])
            graph=nx.read_edgelist(content, delimiter=" ", nodetype=int, data=(('weight',int),('timestamp',int)))
        self.graph=graph
        self.nodes=list(self.graph.nodes())
        self.edges=list(self.graph.edges())
        self.communities = nxcom.greedy_modularity_communities(self.graph)
        self.set_node_community(self.graph,self.communities)
        self.p_edges=list(itertools.combinations(self.nodes, 2))
        print("Finished loading")


    def set_node_community(self, G, communities):
        for c, v_c in enumerate(communities):
            for v in v_c:
                # Add 1 to save 0 for external edges
                G.nodes[v]['community'] = c + 1

    def get_features_inner(self,inp):
        s=inp[0]
        p_edges=inp[1]
        attributes_calculator = features.FeatureConstructor(self.graph)
        attributes_list={}
        if attributes_list == {}:
                    ordered_attributes_list = attributes_calculator.attributes_map.keys()
                    for attribute in ordered_attributes_list:
                        attributes_list[attribute] = {}
        line = 0
        node_feature_dataset = []
        for pair in p_edges:
            column=0
            n1, n2  = pair
            attributes_calculator.set_nodes(n1, n2)
            for function in ordered_attributes_list:
                parameters = attributes_list[function]
                node_feature_dataset.append(attributes_calculator.attributes_map[function](**parameters))
                column+=1
            line+=1
        np.save(f'split_{s}_features.npy',node_feature_dataset)
        return node_feature_dataset

    def get_node_features(self):
        attributes_calculator = features.FeatureConstructor(self.graph)
        attributes_list={}
        if attributes_list == {}:
                    ordered_attributes_list = attributes_calculator.attributes_map.keys()
                    for attribute in ordered_attributes_list:
                        attributes_list[attribute] = {}

        num_cores = multiprocessing.cpu_count()#//2
        k=len(self.p_edges)//num_cores
        splits=[[i,self.p_edges[i*k:(i+1)*k]] if i<num_cores-1 else [i,self.p_edges[i*k:]] for i in range(num_cores)]
        r=Parallel(n_jobs=num_cores)(delayed(self.get_features_inner)(s) for s in splits)
        print(r[:10])

        line = 0
        for pair in self.p_edges:
            column=0
            n1, n2  = pair
            attributes_calculator.set_nodes(n1, n2)
            for function in ordered_attributes_list:
                parameters = attributes_list[function]
                node_feature_dataset[line][column] = attributes_calculator.attributes_map[function](**parameters)
                node_feature_dataset[line][-1] = 0
                column+=1
            line+=1
        return node_feature_dataset

    def get_graph_characterisitcs(self):
        return

    def sample(self,attribute_name="timestamp",split_date=1015887601):
        train_edges = [(x,y) for x,y,t in self.graph.edges(data=attribute_name) if t<=split_date]
        test_edges = [(x,y) for x,y,t in self.graph.edges(data=attribute_name) if t>split_date]
        print(len(train_edges))
        print(len(test_edges))
        return train_edges, test_edges


extractor=Extractor("data/tiny.txt")#Extractor("data/ca-cit.txt")
train, test=extractor.sample()
node_feature_dataset=extractor.get_node_features()
print(node_feature_dataset[:10])