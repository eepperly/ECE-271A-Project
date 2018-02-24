#!/usr/bin/env python

from abc import ABCMeta
from abc import abstractmethod

import numpy as np
import cvxpy

class NonSmoothFunc(object):

    @abstractmethod
    def evaluate( self, vec ):
        pass

    @abstractmethod
    def prox( self, vec, scaling=1.0 ):
        pass

class NonSmoothZeroFunc(NonSmoothFunc):

    def evaluate( self, vec ):
        return 0.0

    def prox( self, vec, scaling=1.0 ):
        return var

class NonSmoothOneNorm(NonSmoothFunc):

    def evaluate( self, vec ):
        return np.linalg.norm(vec, ord=1)

    def prox( self, vec, scaling=1.0 ):
        var = cvxpy.Variable(len(vec))
        objective = cvxpy.Minimize( scaling*cvxpy.norm(var, 1) + 0.5* cvxpy.sum_squares(vec - var) )
        prob = cvxpy.Problem( objective )

        prob.solve()
        assert prob.status == cvxpy.OPTIMAL

        return var.value
