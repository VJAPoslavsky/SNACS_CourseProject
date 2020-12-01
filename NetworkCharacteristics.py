import networkx as nx
import random
import numpy as np
import pandas as pd
import itertools

class NetworkCharacteristics:
    def __init__(self,graph):
        self.functions = [
            self.number_of_nodes,
            self.number_of_edges,
            self.number_of_CC,
            #number_of_WCC,
            #number_of_SCC,
            #nodes_in_WCC,
            #edges_in_WCC,
            #nodes_in_SCC,
            #edges_in_SCC,
            self.density,
            self.average_clustering_coefficient,
            #degree_assortativity_coefficient,
            self.max_degree,
            self.min_degree,
            self.average_degree,
            self.number_of_triangles,
            self.cluster_transitivity,
            self.connectivity,
            k_components,
            self.average_betweenness_centrality,
            self.average_degree_centrality,
            self.average_eigenvector_centrality,
            self.average_closeness_centrality,
            self.average_harmonic_centrality,
            #average_katz_centrality,
            self.average_shortest_path,
            self.number_of_bridges,
            self.average_eccentricity,
            self.eccentricity
        ]
        self.graph=graph
        self.nodes=list(self.graph.nodes())
        self.edges=list(self.graph.edges())
        self.timesplit=timesplit
        #self.p_edges=list(itertools.combinations(self.nodes, 2))
        #self.p_label={e:0 for e in self.p_edges}

    def number_of_nodes(self):
        return self.graph.number_of_nodes()

    def number_of_edges(self):
        return self.graph.number_of_edges()

    def number_of_CC(self):
        return nx.number_connected_components(self.graph)

    def number_of_WCC(self):
        return nx.number_weakly_connected_components(self.graph)

    def number_of_SCC(self):
        return x.number_strongly_connected_components(self.graph)

    def nodes_in_WCC(self):
        G_largest_wcc = max(nx.weakly_connected_components(self.graph), key=len)
        return G_largest_wcc.number_of_nodes()

    def edges_in_WCC(self):
        G_largest_wcc = max(nx.weakly_connected_components(self.graph), key=len)
        return G_largest_scc.number_of_nodes()

    def nodes_in_SCC(self):
        G_largest_scc = nx.Graph(self.graph.subgraph(largest_scc))
        return G_largest_scc.number_of_nodes()

    def edges_in_SCC(self):
        G_largest_scc = nx.Graph(self.graph.subgraph(largest_scc))
        return G_largest_scc.number_of_edges()

    def density(self):
        return nx.density(self.graph)

    def average_clustering_coefficient(self):
        return nx.average_clustering(self.graph)

    def degree_assortativity_coefficient(self):
        return degree_assortativity_coefficient(self.graph, nodes=G_sample)

    def average_clustering_coefficient(self):
        return nx.average_clustering(self.graph)

    def degree_assortativity_coefficient(self):
        return degree_assortativity_coefficient(self.graph, nodes=G_sample)

    def max_degree(self):
        degree_df = pd.DataFrame(self.graph.degree())
        degrees = degree_df.iloc[:,1]
        degrees = pd.to_numeric(degrees)
        return max(degrees)

    def min_degree(self):
        degree_df = pd.DataFrame(self.graph.degree())
        degrees = degree_df.iloc[:,1]
        degrees = pd.to_numeric(degrees)
        return min(degrees)

    def average_degree(self):
        degree_df = pd.DataFrame(self.graph.degree())
        degrees = degree_df.iloc[:,1]
        degrees = pd.to_numeric(degrees)
        return np.mean(degrees)

    def number_of_triangles(self):
        return sum(nx.triangles(self.graph).values()) / 3

    def cluster_transitivity(self):
        return nx.transitivity(self.graph)

    def connectivity(self):
        return nx.node_connectivity(self.graph)

    def k_components(self):
        return nx.k_components(self.graph)

    def average_betweenness_centrality(self):
        betweenness_centrality = nx.betweenness_centrality(self.graph, normalized=True, k=1000)
        betweenness_centrality = np.fromiter(betweenness_centrality.values(), dtype=float)
        return np.mean(betweenness_centrality)

    def average_degree_centrality(self):
        degree = nx.degree_centrality(self.graph)
        degree = np.fromiter(degree.values(), dtype=float)
        return np.mean(degree)

    def average_eigenvector_centrality(self):
        eigen = nx.eigenvector_centrality(self.graph)
        eigen = np.fromiter(eigen.values(), dtype=float)
        return np.mean(eigen)

    def average_closeness_centrality(self):
        closeness = nx.closeness_centrality(self.graph)
        closeness = np.fromiter(closeness.values(), dtype=float)
        return np.mean(closeness)

    def average_harmonic_centrality(self):
        harmonic = nx.harmonic_centrality(self.graph)
        harmonic = np.fromiter(harmonic.values(), dtype=float)
        return np.mean(harmonic)

    def average_katz_centrality(self):
        katz = nx.katz_centrality(self.graph)
        katz = np.fromiter(katz.values(), dtype=float)
        return np.mean(katz)

    def average_shortest_path(self):
        nodes_shortest_paths = []
        for C in (self.graph.subgraph(c).copy() for c in nx.connected_components(self.graph)):
            row = (C.number_of_nodes(), nx.average_shortest_path_length(C))
            nodes_shortest_paths.append(row)
        df = pd.DataFrame(nodes_shortest_paths, columns=['num_of_nodes', 'avg_path_length'])
        df['mult'] = df['num_of_nodes'] * df['avg_path_length']
        return df['mult'].sum() / df['num_of_nodes'].sum()

    def number_of_bridges(self):
        bridges = list(nx.bridges(self.graph))
        return len(bridges)

    def average_eccentricity(self):
        nodes_eccentricities = []
        for C in (self.graph.subgraph(c).copy() for c in nx.connected_components(self.graph)):
            num_nodes = C.number_of_nodes()
            eccentricity = nx.eccentricity(C)
            eccentricity = np.fromiter(eccentricity.values(), dtype=float)
            mean_eccentricity = np.mean(eccentricity)
            row = (num_nodes, mean_eccentricity)
            nodes_eccentricities.append(row)
        df = pd.DataFrame(nodes_eccentricities, columns=['num_of_nodes', 'avg_eccentricity'])
        df['mult'] = df['num_of_nodes'] * df['avg_eccentricity']
        return df['mult'].sum() / df['num_of_nodes'].sum()

    def eccentricity(self):
        max_eccentricity = 0
        for C in (self.graph.subgraph(c).copy() for c in nx.connected_components(self.graph)):
            eccentricities = nx.eccentricity(C)
            eccentricities = np.fromiter(eccentricities.values(), dtype=float)
            curr_max = max(eccentricities)
            if (curr_max > max_eccentricity):
                max_eccentricity = curr_max
        return max_eccentricity

    def initialize_characteristics_df(self):
        columns = []
        for func in self.functions:
            columns.append(func.__name__)
        columns.insert(0, 'timesplit')
        df = pd.DataFrame(columns=columns)
        return df

    def extract_characteristics(self):
        df = self.initialize_characteristics_df()
        df_row = []
        for func in self.functions:
            result = func()
            df_row.append(result)
        df_row.insert(0, self.timesplit)
        df = df.append(dict(zip(df.columns, df_row)), ignore_index=True)
        df.to_csv('Results/Characteristics_DBLP_Timesplit_' + str(self.timesplit) + '.csv', index=False)
        return df

netchars = NetworkCharacteristics(graph_path='Datasets/Collaboration/DBLP_Graph.csv', timesplit=1991)
characteristics = netchars.extract_characteristics()
