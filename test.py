#!/usr/bin/env python2

import numpy as np
from graph import Graph
from basis_pursuit import basis_pursuit

import cvxpy
import scipy

debug_output = False
default_steps = 4

#R1 = np.matrix('1 2;3 4')
#R2 = np.matrix('1 2 3;4 5 6')
#R3 = np.matrix('1 2 3 4;5 6 7 8')

R1 = np.matrix(np.random.rand(2,2))
R2 = np.matrix(np.random.rand(2,3))
R3 = np.matrix(np.random.rand(2,4))

r = np.random.rand(2,1)

matrices = [R1, R2, R3]

vector = r

graph = Graph(3)
graph.add_edge(0,1)
graph.add_edge(0,2)
graph.add_edge(1,2)

import sys
if len(sys.argv) > 1:
    steps = int(sys.argv[1])
else:
    steps = default_steps

primal, dual, primal_history, dual_history = basis_pursuit( matrices, vector, graph, num_iters=steps, solver_param=1.0, gamma=1.0, debug_output=debug_output )

print "Primal"
print primal
print

print "Dual"
print dual
print

# R = scipy.linalg.block_diag(*matrices)
# r = np.vstack(vectors)
R = np.hstack(matrices)

var = cvxpy.Variable(R.shape[1])
objective = cvxpy.Minimize( cvxpy.norm(var, 1) )
constraints = [R*var == r]

prob = cvxpy.Problem( objective, constraints )
prob.solve()
assert prob.status == cvxpy.OPTIMAL

print "Primal CVXPY"
print var.value

import matplotlib.pyplot as plt

plt.plot([sum([np.linalg.norm(vec, ord=1) for vec in primal]) for primal in primal_history])
plt.show()
