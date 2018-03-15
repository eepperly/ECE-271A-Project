#!/usr/bin/env python

from smooth_func import SmoothZeroFunc, SmoothOneNorm
from nonsmooth_func import NonSmoothOneNorm, NonSmoothZeroFunc
from distributed_primal_dual import DistributedProblem
from cone import ZeroCone

import numpy as np

def basis_pursuit( matrices, vector, graph, num_iters=10, solver_param=0.1, gamma=0.1, debug_output=False, non_smooth_one_norm=True, cautious=False ):

    n = len(matrices)

    # vectors = (n-1)*[np.zeros(vector.shape)] + [vector]
    vectors = n*[vector/float(n)]

    if not isinstance(solver_param, list):
        solver_param = n*[solver_param]
    
    multiplier = 0.5 if cautious else 1.0

    import scipy
    matrix_norm = lambda mat: np.linalg.norm(mat.todense(), ord=2) if isinstance(mat, scipy.sparse.spmatrix) else np.linalg.norm(mat, ord=2)
    
    taus = [multiplier*solver_param[i] / matrix_norm(matrices[i]) for i in range(n)]
    kappas = [multiplier/(2*gamma*graph.degree(i) + solver_param[i]*matrix_norm(matrices[i])) for i in range(n)]

    initial_primal_guess = []
    initial_dual_guess = []

    for i in range(n):

        initial_primal_guess.append( np.zeros( (matrices[i].shape[1], 1) ) )
        initial_dual_guess.append( np.zeros( (matrices[i].shape[0], 1) ) )

    return basis_pursuit_solver(matrices, vectors, gamma, taus, kappas, graph, initial_primal_guess, initial_dual_guess, num_iters, debug_output=debug_output, non_smooth_one_norm=non_smooth_one_norm)

def basis_pursuit_solver(matrices, vectors, gamma, taus, kappas, graph, initial_primal_guess, initial_dual_guess, num_iters, debug_output=False, non_smooth_one_norm=True):

    n = len(matrices)
    
    if non_smooth_one_norm:
        smooth_funcs = []

        for i in range(n):
            smooth_funcs.append( SmoothZeroFunc(matrices[i].shape[1]) )

        non_smooth_funcs = n * [NonSmoothOneNorm()]

    else:
        smooth_funcs = []

        for i in range(n):
            smooth_funcs.append( SmoothOneNorm(matrices[i].shape[1]) )

        non_smooth_funcs = n * [NonSmoothZeroFunc()]
        
    cone = ZeroCone()

    basis_pursuit_problem = DistributedProblem(smooth_funcs, non_smooth_funcs, matrices, vectors, graph, cone)

    return basis_pursuit_problem.solve(num_iters, initial_primal_guess, initial_dual_guess, gamma, taus, kappas, debug_output=debug_output)

def basis_pursuit_exact(matrices, vector):

    import cvxpy, scipy

    R = scipy.sparse.hstack(matrices) if isinstance(matrices[0], scipy.sparse.spmatrix) else np.hstack(matrices)

    var = cvxpy.Variable(R.shape[1])
    objective = cvxpy.Minimize( cvxpy.norm(var, 1) )
    constraints = [R*var == vector]
    
    prob = cvxpy.Problem( objective, constraints )
    prob.solve()
    assert prob.status == cvxpy.OPTIMAL

    return var.value
