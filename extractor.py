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
        #self.communities = nxcom.greedy_modularity_communities(self.graph)
        #self.set_node_community(self.graph,self.communities)
        self.p_edges=list(itertools.combinations(self.nodes, 2))
        self.p_label={e:0 for e in self.p_edges}
        print("Finished loading")


    def set_node_community(self, G, communities):
        for c, v_c in enumerate(communities):
            for v in v_c:
                # Add 1 to save 0 for external edges
                G.nodes[v]['community'] = c + 1

    def get_features_inner(self,inp):
        s=inp[0]
        p_edges=inp[1]
        f=open(f'results/split_{s}_features.csv', 'w+')
        f.close()
        attributes_calculator = features.FeatureConstructor(self.graph)
        attributes_list={}
        if attributes_list == {}:
                    ordered_attributes_list = attributes_calculator.attributes_map.keys()
                    for attribute in ordered_attributes_list:
                        attributes_list[attribute] = {}
        line = 0
        for pair in p_edges:
            column=0
            n1, n2  = pair
            attributes_calculator.set_nodes(n1, n2)
            column_values=np.zeros(len(ordered_attributes_list)+1)
            for function in ordered_attributes_list:
                parameters = attributes_list[function]
                column_values[column] = attributes_calculator.attributes_map[function](**parameters)
                column += 1
            column_values[-1] = self.p_label[pair]
            line += 1
            with open(f'results/split_{s}_features.csv', 'a+') as file:
                np.savetxt(file, [column_values], delimiter=",",fmt='%f')
                file.close()
        return 1

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
        print("starting computing metrics...")
        #parallel execution; dump resuls in text file
        Parallel(n_jobs=num_cores)(delayed(self.get_features_inner)(s) for s in splits)
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

    def sample(self,attribute_name="timestamp",split_date=1015887601):
        train_edges = [(x,y) for x,y,t in self.graph.edges(data=attribute_name) if t<=split_date]
        test_edges = [(x,y) for x,y,t in self.graph.edges(data=attribute_name) if t>split_date]
        print(len(train_edges))
        print(len(test_edges))
        for i in train_edges:
            self.p_label[i]=1
        for i in test_edges:
            self.p_label[i]=1
        return train_edges, test_edges


extractor=Extractor("data/tiny.txt")
train, test=extractor.sample()
node_feature_dataset=extractor.get_node_features()
print("Finished")