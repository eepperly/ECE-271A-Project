#!/usr/bin/env python

class DistributedProblem(object):

    def __init__(self, smooth_funcs, non_smooth_funcs, constraint_matrices, constraint_vectors, graph, cone):

        self.smooth_funcs = smooth_funcs
        self.non_smooth_funcs = non_smooth_funcs
        self.constraint_matrices = constraint_matrices
        self.constraint_vectors = constraint_vectors
        self.graph = graph
        self.cone = cone

        self.size = len(self.smooth_funcs)
        
        assert self.size == len(self.non_smooth_funcs)
        assert self.size == len(self.constraint_vectors)
        assert self.size == len(self.constraint_matrices)
        assert self.size == len(self.graph)
        
    def solve(self, num_iters, initial_primal_guess, initial_dual_guess, gamma, taus, kappas):

        assert self.size == len(taus)
        assert self.size == len(kappas)

        primal = initial_primal_guess
        dual = initial_dual_guess

        s = [2.0*dual_var for dual_var in dual]
        new_s = []

        dual_history = [ dual ]

        for _ in range(num_iters):

            new_primal = []
            new_dual = []
            
            for i in range(self.size):
                # # Print outs for debugging
                # print "Primal", primal[i].shape
                # print primal[i]

                # print "Smooth Function Gradient", self.smooth_funcs[i].gradient(primal[i]).shape
                # print self.smooth_funcs[i].gradient(primal[i])

                # print "Constraint Matrix", self.constraint_matrices[i].shape
                # print self.constraint_matrices[i]

                # print "Dual", dual[i].shape
                # print dual[i]
                
                new_primal.append( self.non_smooth_funcs[i].prox(primal[i] - taus[i] * (self.smooth_funcs[i].gradient(primal[i]) + self.constraint_matrices[i].transpose()*dual[i]), scaling=taus[i]) )
                
                p = 0.0
                for j in self.graph.adjacent_to(i):
                    p += s[j] - s[i]

                new_dual.append( self.cone.projectPolar( dual[i] + kappas[i]*(self.constraint_matrices[i]*(2*new_primal[i] - primal[i]) - self.constraint_vectors[i] + gamma*p)) )

                new_s_value = 2 * new_dual[i]

                for j in range(len(dual_history)):
                    new_s_value += dual_history[j][i]

                new_s.append( new_s_value )
                
            primal = new_primal
            dual_history.append(new_dual)
            dual = new_dual
            s = new_s

        return primal, dual
