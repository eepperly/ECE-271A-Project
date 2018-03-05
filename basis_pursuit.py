#!/usr/bin/env python

from smooth_func import SmoothZeroFunc
from nonsmooth_func import NonSmoothOneNorm
from distributed_primal_dual import DistributedProblem
from cone import ZeroCone

import numpy as np

def basis_pursuit( matrices, vector, graph, num_iters=10, solver_param=0.1, gamma=0.1, debug_output=False ):

    n = len(matrices)

    smooth_funcs = []

    for i in range(n):
        smooth_funcs.append( SmoothZeroFunc(matrices[i].shape[1]) )
        
    non_smooth_funcs = n * [NonSmoothOneNorm()]

    cone = ZeroCone()

    # vectors = (n-1)*[np.zeros(vector.shape)] + [vector]
    vectors = n*[vector/float(n)]
    
    basis_pursuit_problem = DistributedProblem(smooth_funcs, non_smooth_funcs, matrices, vectors, graph, cone)

    if not isinstance(solver_param, list):
        solver_param = n*[solver_param]
    
    gamma = gamma

    taus = [1/s for s in solver_param]
    kappas = [solver_param[i]/(2*solver_param[i]*gamma*graph.degree(i) + np.linalg.norm(matrices[i], ord=2)**2) for i in range(n)]

    initial_primal_guess = []
    initial_dual_guess = []

    for i in range(n):

        initial_primal_guess.append( np.zeros( (matrices[i].shape[1], 1) ) )
        initial_dual_guess.append( np.zeros( (matrices[i].shape[0], 1) ) )

    return basis_pursuit_problem.solve(num_iters, initial_primal_guess, initial_dual_guess, gamma, taus, kappas, debug_output=debug_output)

