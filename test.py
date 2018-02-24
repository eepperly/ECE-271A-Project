#!/usr/bin/env python2

import numpy as np
from graph import Graph
from basis_pursuit import basis_pursuit

import cvxpy
import scipy

R1 = np.matrix('1 2;3 4')
R2 = np.matrix('1 2 3;4 5 6')
r1 = np.matrix('5;6')
r2 = np.matrix('7;8')

matrices = [R1, R2]
vectors = [r1,r2]

graph = Graph(2)
graph.add_edge(0,1)

primal, dual = basis_pursuit( matrices, vectors, graph, num_iters=10000 )

print "Primal"
print primal
print

print "Dual"
print dual
print

R = scipy.linalg.block_diag(*matrices)
r = np.vstack(vectors)

var = cvxpy.Variable(R.shape[1])
objective = cvxpy.Minimize( cvxpy.norm(var, 1) )
constraints = [R*var == r]

prob = cvxpy.Problem( objective, constraints )
prob.solve()
assert prob.status == cvxpy.OPTIMAL

print "Primal CVXPY"
print var.value
