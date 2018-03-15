#!/usr/bin/env python

import numpy as np

class Graph(object):

    def __init__(self, num_nodes):
        self.adjacency_list = []
        for i in range(num_nodes):
            self.adjacency_list.append( [] )

    def add_edge(self, i, j):
        if not (i in self.adjacency_list[j]):
            self.adjacency_list[i].append(j)
            self.adjacency_list[j].append(i)
            return True
        return False

    def adjacent_to(self, i):
        for j in self.adjacency_list[i]:
            yield j

    def __len__(self):
        return len(self.adjacency_list)

    def degree(self, i):
        return len(self.adjacency_list[i])

import itertools
def generate_graph( num_nodes, graph_density=0.2 ):
    graph = Graph(num_nodes)

    total_possible_graph_edges = num_nodes*(num_nodes-1)/2
    total_graph_edges = int(total_possible_graph_edges * graph_density)
    num_graph_edges = 0

    # Add initial edges to graph for cycle pattern
    for i in range(num_nodes-1):
        graph.add_edge(i,i+1)
        num_graph_edges += 1

    graph.add_edge(0,num_nodes-1)
    num_graph_edges += 1

    possible_edges_to_add = np.array([(i,j) for i,j in itertools.combinations(range(num_nodes), 2)], dtype=tuple)
    np.random.shuffle(possible_edges_to_add)

    i = 0
    while num_graph_edges < total_graph_edges and i < len(possible_edges_to_add):
        edge_added_successfully = graph.add_edge( possible_edges_to_add[i][0], possible_edges_to_add[i][1] )
        if edge_added_successfully:
            num_graph_edges += 1
        i += 1

    return graph
