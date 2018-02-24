#!/usr/bin/env python

from abc import ABCMeta
from abc import abstractmethod
import numpy as np

class SmoothFunc(object):

    @abstractmethod
    def evaluate( self, vec ):
        pass

    @abstractmethod
    def gradient( self, vec ):
        pass

class SmoothAffineFunc(SmoothFunc):

    def __init__( self, c, d ):
        self.c = c
        self.d = d
    
    def evaluate( self, vec ):
        return np.dot( self.c, vec ) + d

    def gradient( self, vec ):
        return self.c

class SmoothZeroFunc(SmoothFunc):

    def __init__( self, output_size ):
        self.output_size = output_size
    
    def evaluate( self, vec ):
        return 0.0

    def gradient( self, vec ):
        return np.zeros( (self.output_size, 1) )
