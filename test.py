#!/usr/bin/env python2

import numpy as np
from graph import Graph, generate_graph
from basis_pursuit import basis_pursuit, basis_pursuit_solver, basis_pursuit_exact

import itertools
import cvxpy
import scipy

# Parse

import argparse

parser = argparse.ArgumentParser(description='Test distributed primal-dual algorithm by solving a basis pursuit problem.')
parser.add_argument("--num_steps", help="number of steps to run primal-dual algorithm", type=int, default=100)
parser.add_argument("--num_nodes", help="number of agents", type=int, default=100)
parser.add_argument("--decision_variable_size", help="number of decision variables per agent", type=int, default=10)
parser.add_argument("--range_dimension", help="dimension of the range of the problem", type=int, default=300)
parser.add_argument("--graph_density", help="fraction of possible edges that will be filled", type=float, default=0.2)
parser.add_argument("--gamma", help="a numerical parameter", type=float, default=None)
parser.add_argument("--solver_param", help="a numerical parameter", type=float, default=1.0)
parser.add_argument("--debugging_output", help="print additional debugging information", action="store_true")
parser.add_argument("--from_folder", help="get initial data and numerical parameters from folder", type=str, default=None)

args = parser.parse_args()

num_nodes = args.num_nodes
decision_variable_size = args.decision_variable_size
range_dimension = args.range_dimension
num_steps = args.num_steps
graph_density = args.graph_density

gamma = args.gamma
if gamma is None:
    gamma = 2*float(num_nodes) / int(graph_density * num_nodes * (num_nodes-1)/2)

solver_param = args.solver_param

debugging_output = args.debugging_output
non_smooth_one_norm = True

from_folder = args.from_folder

if from_folder is None:
    # Formulate problem

    matrices = []

    for i in range(num_nodes):
        matrices.append( np.matrix(np.random.normal(size=(range_dimension, decision_variable_size))) )

    vector = sum([matrices[i]*np.random.normal(size=(decision_variable_size,1)) for i in range(num_nodes)])

    # Construct graph

    graph = generate_graph(num_nodes, graph_density=graph_density)

    # Solve!
    primal, dual, primal_history, dual_history = basis_pursuit( matrices, vector, graph, num_iters=num_steps, solver_param=solver_param, gamma=gamma, non_smooth_one_norm=non_smooth_one_norm, debug_output=debugging_output )

else:

    folder = from_folder
    
    from reader import read_graph, read_array, read_matrix, read_number
    import os
    
    big_matrix = read_matrix( os.path.join(folder, "big_R.dat") )
    vector = read_matrix( os.path.join(folder, "little_r.dat") )

    gamma = read_number( os.path.join(folder, "gamma.dat") )

    graph = read_graph( os.path.join(folder, "graph.dat") )

    kappas = read_array( os.path.join(folder, "kappa.dat") )
    taus = read_array( os.path.join(folder, "tau.dat") )

    kappas.tolist()
    taus.tolist()

    kappas = kappas[0]
    taus = taus[0]

    initial_primal_guess_matrix = read_array( os.path.join(folder, "xi.dat") )
    initial_dual_guess_matrix = read_array( os.path.join(folder, "y.dat") )

    num_nodes = initial_primal_guess_matrix.shape[1]

    initial_primal_guess = [initial_primal_guess_matrix[:,[i]] for i in range(num_nodes)]
    initial_dual_guess = [initial_dual_guess_matrix[:,[i]] for i in range(num_nodes)]

    primal_len = len(initial_primal_guess[0])
    matrices = [big_matrix[:,i*primal_len:(i+1)*primal_len] for i in range(num_nodes)]
    vectors = [vector/float(num_nodes) for _ in range(num_nodes)]
    
    primal, dual, primal_history, dual_history = basis_pursuit_solver(matrices, vectors, gamma, taus, kappas, graph, initial_primal_guess, initial_dual_guess, num_steps, debug_output=debugging_output, non_smooth_one_norm=non_smooth_one_norm)

    
# Get CVXPY solution

optimal = basis_pursuit_exact(matrices, vector)
optimal_norm = np.linalg.norm(optimal, ord=1)

vector_norm = np.linalg.norm(vector, ord=1)
R = np.hstack(matrices)

fName = "results.dat"

with open(fName, "w") as write_file:
    for i in range(len(primal_history)):

        primal_val = np.vstack(primal_history[i])
        optimality = (np.linalg.norm(primal_val, ord=1) - optimal_norm)/optimal_norm
        infeasibility = np.linalg.norm( R*primal_val - vector, ord=1 ) / vector_norm

        dual_consensus = -np.inf
        for j, k in itertools.combinations(range(num_nodes), 2):
            dual_consensus = max(dual_consensus, np.linalg.norm(dual_history[i][j] - dual_history[i][k], ord=1))

        write_file.write("{:18}\t{:18}\t{:18}\n".format(optimality, infeasibility, dual_consensus))
