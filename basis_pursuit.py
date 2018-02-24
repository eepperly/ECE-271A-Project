#!/usr/bin/env python

from smooth_func import SmoothZeroFunc
from nonsmooth_func import NonSmoothOneNorm
from distributed_primal_dual import DistributedProblem
from cone import ZeroCone

import numpy as np

def basis_pursuit( matrices, vectors, graph, num_iters=10 ):

    n = len(matrices)

    smooth_funcs = []

    for i in range(n):
        smooth_funcs.append( SmoothZeroFunc(matrices[i].shape[1]) )
    
    non_smooth_funcs = n * [NonSmoothOneNorm()]

    cone = ZeroCone()

    basis_pursuit_problem = DistributedProblem(smooth_funcs, non_smooth_funcs, matrices, vectors, graph, cone)

    gamma = 0.01
    taus = n*[0.01]
    kappas = n*[0.01]

    initial_primal_guess = []
    initial_dual_guess = []

    for i in range(n):

        initial_primal_guess.append( np.zeros( (matrices[i].shape[1], 1) ) )
        initial_dual_guess.append( np.zeros( (matrices[i].shape[0], 1) ) )

    return basis_pursuit_problem.solve(num_iters, initial_primal_guess, initial_dual_guess, gamma, taus, kappas)

