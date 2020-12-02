#code sourced from https://github.com/alpecli/predlig;  modified to our application and needs.
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import inspect
import random
import time
from joblib import Parallel, delayed, parallel_backend
import multiprocessing

class FeatureConstructor:
    """
    A feature constructor.

    This class calculates topological attributes for a defined pair of vertices, based on the structure of the provided graph.
    """
    def __init__(self, graph, page_rank=None, node_1=None, node_2=None):
        self.graph = graph
        self.node_1 = node_1
        self.node_2 = node_2
        self.clustering_dict = None
        self.betweenness_dict = None
        self.average_neighbor_degree_dict = None
        self.load_centrality_dict = None
        self.communicability_centrality_dict = None
        self.eigenvector_centrality_dict = None
        self.closeness_vitality_dict = None
        self.core_number_dict = None
        self.page_rank= page_rank


        self.attributes_map = {
            "common_neighbors": self.common_neighbors,
            "jaccard_coefficient": self.jaccard_coefficient,
            "resource_allocation": self.resource_allocation,
            "adamic_adar_similarity": self.adamic_adar_similarity,
            "preferential_attachment": self.preferential_attachment,
            "cn_soundarajan_hopcroft": self.cn_soundarajan_hopcroft,
            "ra_index_soundarajan_hopcroft":self.ra_index_soundarajan_hopcroft,
            "betweenness_centrality": self.betweenness_centrality,
            "closeness_centrality_sum": self.closeness_centrality_sum,
            "average_clustering_coefficient": self.average_clustering_coefficient,
            "average_neighbor_degree_sum": self.average_neighbor_degree_sum,
            "clustering_coefficient_sum": self.clustering_coefficient_sum,
            "sum_of_neighbors": self.sum_of_neighbors,
            "get_shortest_path_length": self.get_shortest_path_length,
            "get_second_shortest_path_length": self.get_second_shortest_path_length,
            "cosine": self.cosine,
            "katz_measure": self.katz_measure,
            "mean_page_rank":self.mean_page_rank,
            "max_page_rank":self.max_page_rank
            #"eigenvector_centrality_sum": self.eigenvector_centrality_sum,
        }

        if(self.node_1 != None and self.node_2 != None):
            self.neighbors_1 = self.all_neighbors(self.node_1)
            self.neighbors_2 = self.all_neighbors(self.node_2)


    def get_features(self,edge):
        node_1,node_2=edge
        self.node_1 = node_1
        self.node_2 = node_2
        self.neighbors_1 = self.all_neighbors(self.node_1)
        self.neighbors_2 = self.all_neighbors(self.node_2)
        res=[self.common_neighbors(),self.jaccard_coefficient(),self.resource_allocation(),self.adamic_adar_similarity(),self.preferential_attachment(),self.cn_soundarajan_hopcroft(),self.ra_index_soundarajan_hopcroft(),self.betweenness_centrality(),self.closeness_centrality_sum(),self.average_clustering_coefficient(),self.average_neighbor_degree_sum(),self.clustering_coefficient_sum(),self.sum_of_neighbors(),self.get_shortest_path_length(),self.get_second_shortest_path_length(),self.cosine(),self.katz_measure(),self.mean_page_rank(),self.max_page_rank()]
        return res

    def set_nodes(self, node_1, node_2):
        self.node_1 = node_1
        self.node_2 = node_2
        self.neighbors_1 = self.all_neighbors(self.node_1)
        self.neighbors_2 = self.all_neighbors(self.node_2)

    def all_neighbors(self, node):
        neighbors = set()
        for neighbor in list(nx.all_neighbors(self.graph, node)):
            neighbors.add(neighbor)
        return neighbors - set([node])

    def common_neighbors(self):
        intersection=self.neighbors_1.intersection(self.neighbors_2)
        w=0
        for i in list(intersection):
            w+=self.graph.edges[[self.node_1,i]]["weight"]
            w+=self.graph.edges[[self.node_2,i]]["weight"]
        return w

    def sum_of_neighbors(self):
        w=0
        for i in self.neighbors_1:
            w+=self.graph.edges[[self.node_1,i]]["weight"]
        for i in self.neighbors_2:
            w+=self.graph.edges[[self.node_2,i]]["weight"]
        return w

    def jaccard_coefficient (self):
        try:
            return len(self.neighbors_1.intersection(self.neighbors_2))/len(self.neighbors_1.union(self.neighbors_2))
        except ZeroDivisionError:
            return 0

    def adamic_adar_similarity(self):
        measure = 0
        for neighbor in self.neighbors_1.intersection(self.neighbors_2):
            secondary_neighbors = self.all_neighbors(neighbor)
            measure += 1 / (np.log10(len(secondary_neighbors)))

        return measure

    def resource_allocation(self):
        r=nx.resource_allocation_index(self.graph, [(self.node_1, self.node_2)])
        return list(r)[0][2]

    def preferential_attachment(self):
        return len(self.neighbors_1) * len(self.neighbors_2)

    def cn_soundarajan_hopcroft(self):
        return list(nx.cn_soundarajan_hopcroft(self.graph, [(self.node_1, self.node_2)]))[0][2]

    def ra_index_soundarajan_hopcroft(self):
        return list(nx.ra_index_soundarajan_hopcroft(self.graph, [(self.node_1, self.node_2)]))[0][2]

    def katz_measure(self, beta = 0.85, min_length = 3):
        measure = 0
        try:
            sp=get_shortest_path_length()
        except:
            sp=-1
        if sp >= min_length:
            for i in nx.all_simple_paths(self.graph, self.node_1, self.node_2, min_length):
                measure += beta**len(i) * len(i)
        return measure

    def cosine(self):
        try:
            return self.common_neighbors() / self.preferential_attachment()
        except ZeroDivisionError:
            return 0

    def all_papers(self, graph, author):
        years = set()
        papers = 0
        for paper in list(nx.all_neighbors(graph, author)):
            years.add(graph.node[paper]['year'])
            papers += 1

        return papers/len(years)

    def sum_of_papers(self, graph, node_1, node_2):
        papers_1 = self.all_papers(graph, node_1)
        papers_2 = self.all_papers(graph, node_2)
        return papers_1 + papers_2

    def coefficient_clustering_sum(self):
        if (self.clustering_dict == None):
            self.clustering_dict = nx.clustering(self.graph)
            time.sleep(1)
        return self.clustering_dict[self.node_1] + self.clustering_dict[self.node_2]

    def clustering_coefficient_sum(self):
        if (self.clustering_dict == None):
            self.clustering_dict = nx.clustering(self.graph)
            time.sleep(1)
        return self.clustering_dict[self.node_1] + self.clustering_dict[self.node_2]

    def betweenness_centrality(self):
        if (self.betweenness_dict == None):
            self.betweenness_dict = nx.betweenness_centrality(self.graph)
        return self.betweenness_dict[self.node_1] * self.betweenness_dict[self.node_2]

    def closeness_centrality_sum(self):
        return nx.closeness_centrality(self.graph, self.node_1) + nx.closeness_centrality(self.graph, self.node_2)

    def average_neighbor_degree_sum(self):
        if (self.average_neighbor_degree_dict == None):
            self.average_neighbor_degree_dict = nx.average_neighbor_degree(self.graph, weight="weight")
            time.sleep(1)
        return self.average_neighbor_degree_dict[self.node_1] + self.average_neighbor_degree_dict[self.node_2]

    def average_clustering_coefficient(self):
        return nx.average_clustering(self.graph, [self.node_1, self.node_2])

    def square_clustering_coefficient_sum(self):
        return nx.square_clustering(self.graph, [self.node_1])[self.node_1] + nx.square_clustering(self.graph, [self.node_2])[self.node_2]

    def get_shortest_path_length(self):
        try:
            sp=nx.shortest_path_length(self.graph, source=self.node_1, target=self.node_2)
        except:
            sp=-1
        return sp

    def get_second_shortest_path_length(self):
        shortest_paths_edges = []
        second_shortest_path_length = -1
        try:
            for path in nx.all_shortest_paths(self.graph, source=self.node_1, target=self.node_2):
                path_edges = self.get_edges_from_path(path)
                shortest_paths_edges.append(path_edges)
                #self.graph.remove_edges_from(path_edges)

            try:
                second_shortest_path_length = nx.shortest_path_length(self.graph, self.node_1, self.node_2)
            except:
                pass
            finally:
                for path in shortest_paths_edges:
                    self.graph.add_edges_from(path)
        except:
            pass
        return second_shortest_path_length

    def mean_page_rank(self):
        return (self.page_rank[self.node_1]+self.page_rank[self.node_2])/2

    def max_page_rank(self):
        return max(self.page_rank[self.node_1],self.page_rank[self.node_2])

    def get_edges_from_path(self, path):
        return [(path[node], path[node+1]) for node in range(len(path) -1)]

    def load_centrality_sum(self):
        if (self.load_centrality_dict == None):
            self.load_centrality_dict = nx.load_centrality(self.graph)
            time.sleep(1)
        return self.load_centrality_dict[self.node_1] + self.load_centrality_dict[self.node_2]

    def communicability_centrality_sum(self):
        if (self.communicability_centrality_dict == None):
            self.communicability_centrality_dict = nx.communicability_centrality(self.graph)
            time.sleep(1)
        return self.communicability_centrality_dict[self.node_1] + self.communicability_centrality_dict[self.node_2]

    def eigenvector_centrality_sum(self):
        if (self.eigenvector_centrality_dict == None):
            self.eigenvector_centrality_dict = nx.eigenvector_centrality(self.graph)
            time.sleep(1)
        return self.eigenvector_centrality_dict[self.node_1] + self.eigenvector_centrality_dict[self.node_2]

    def closeness_vitality_sum(self):
        if (self.closeness_vitality_dict == None):
            self.closeness_vitality_dict = nx.closeness_vitality(self.graph)
            time.sleep(1)
        return self.closeness_vitality_dict[self.node_1] + self.closeness_vitality_dict[self.node_2]

    def core_number_sum(self):
        if (self.core_number_dict == None):
            self.core_number_dict = nx.core_number(self.graph)
            time.sleep(1)
        return self.core_number_dict[self.node_1] + self.core_number_dict[self.node_2]