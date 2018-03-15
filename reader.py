#!/usr/bin/env

'''
Code to read data structures for text files
'''

import re
from graph import Graph
import numpy as np

def read_graph( fName ):

    graph_line_re = r'(\d+),\s*(\d+)[,\n].+'
    empty_line_re = r'\s*'

    edges = []
    
    with open(fName, "r") as f:
        for line in f:
            match_group = re.match(graph_line_re, line)

            if match_group:
                edges.append( (int(match_group.group(1))-1, int(match_group.group(2))-1) )
            elif not re.match(empty_line_re, line):
                raise RuntimeError("Encountered unexpected line '{}' in parsing graph".format(line))

    num_nodes = max([max(a[0], a[1]) for a in edges])+1

    graph = Graph(num_nodes)

    for edge in edges:

        graph.add_edge( edge[0], edge[1] )

    return graph

def read_matrix( fName, return_as_array=False ):

    data = []

    with open(fName, "r") as f:

        for line in f:

            if line.rstrip() == "":
                continue

            data.append(map(float, line.split(',')))

    if return_as_array:
        return np.array( data, ndmin=2 )

    else:
        return np.matrix( data )

def read_array( fName ):
    return read_matrix( fName, return_as_array=True )

def write_array( fName, array ):

    np.savetxt( fName, array, delimiter="," )
    # with open(fName, "w") as write_file:
    #     for i in range(array.shape[0]):
    #         write_file.write(",".join(map(str, array[i,:].flatten().tolist()[0]))+"\n")
            

def read_number( fName ):

    with open(fName, "r") as f:

        for line in f:

            return float(line.rstrip())

