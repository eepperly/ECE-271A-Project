#!/usr/bin/env python

class Graph(object):

    def __init__(self, num_nodes):
        self.adjacency_list = []
        for i in range(num_nodes):
            self.adjacency_list.append( [] )

    def add_edge(self, i, j):
        if not (i in self.adjacency_list[j]):
            self.adjacency_list[i].append(j)
            self.adjacency_list[j].append(i)

    def adjacent_to(self, i):
        for j in self.adjacency_list[i]:
            yield j

    def __len__(self):
        return len(self.adjacency_list)

    def degree(self, i):
        return len(self.adjacency_list[i])
