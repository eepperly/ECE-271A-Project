#!/usr/bin/env python

from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import cvxpy

class Cone(object):

    __metaclass__ = ABCMeta
    
    @abstractmethod
    def isIn( self, num ):
        pass

    def inPrimal( self, num ):
        return self.isIn( num )
    
    @abstractmethod
    def inDual( self, num ):
        return True

    def inPolar( self, num ):
        return self.inDual(-num)
    
    @abstractmethod
    def projectPrimal( self, num ):
        return None

    @abstractmethod
    def projectDual( self, num ):
        return None

    def projectPolar( self, num ):
        return -self.projectDual(-num)

class PolyhedralCone(Cone):

    def __init__(self, matrix):
        self.matrix = matrix # polyhedral cone refers to the set of all x such that matrix*x >= 0

    @staticmethod
    def getPositiveCone(self, size):
        import scipy
        return PolyhedralCone(scipy.sparse.eye(size))
        
    def isIn( self, vec ):
        return np.all( self.matrix * vec >= 0 )

    def inDual( self, vec ):
        raise NotImplementedError("Dual variable querying has not been implemented yet!")

    def projectPrimal( self, vec ):
        if self.isIn(vec):
            return vec
        
        projected = cvxpy.Variable(len(vec))
        projection_distance_squared = cvxpy.Minimize(cvxpy.sum_squares( vec - projected ))
        constraints = [self.matrix*projected >= 0.0]
        prob = cvxpy.Problem(projection_distance_squared, constraints)
        prob.solve()
        assert prob.status == cvxpy.OPTIMAL

        return projected.value

    def projectDual( self, vec ):
        var = cvxpy.Variable(self.matrix.shape[0])
        projection_distance_squared = cvxpy.Minimize(cvxpy.sum_squares( vec - self.matrix.tranpose()*var ))
        constraints = [var >= 0.0]
        prob = cvxpy.Problem(projection_distance_squared, constraints)
        prob.solve()
        assert prob.status == cvxpy.OPTIMAL

        return self.matrix.transpose()*var.value

TOLERANCE = 1e-11
    
def NullSpaceCone(Cone):

    def __init__(self, matrix):
        self.matrix = matrix

    def isIn(self, vec):
        return np.linalg.norm(self.matrix * vec, ord=np.inf) < TOLERANCE * np.linalg.norm(vec)

    def inDual(self, vec):
        raise NotImplementedError("Querying in dual not supported yet")

    def projectPrimal(self, vec):
        if self.isIn(vec):
            return vec
                
        projected = cvxpy.Variable(len(vec))
        projection_distance_squared = cvxpy.Minimize(cvxpy.sum_squares( vec - projected ))
        constraints = [self.matrix*projected == 0.0]
        prob = cvxpy.Problem(projection_distance_squared, constraints)
        prob.solve()
        assert prob.status == cvxpy.OPTIMAL

        return projected.value

    def projectDual( self, vec ):
        var = cvxpy.Variable(self.matrix.shape[0])
        projection_distance_squared = cvxpy.Minimize(cvxpy.sum_squares( vec - self.matrix.tranpose()*var ))
        constraints = [var == 0.0]
        prob = cvxpy.Problem(projection_distance_squared, constraints)
        prob.solve()
        assert prob.status == cvxpy.OPTIMAL

        return self.matrix.transpose()*var.value

class ZeroCone(Cone):

    def isIn(self, vec):
        return np.linalg.norm(vec, ord=np.inf) < TOLERANCE

    def inDual(self, vec):
        return True

    def projectPrimal(self, vec):
        return np.zeros(vec.shape)

    def projectDual(self, vec):
        return vec
