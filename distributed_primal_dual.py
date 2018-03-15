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
        
    def solve(self, num_iters, initial_primal_guess, initial_dual_guess, gamma, taus, kappas, debug_output=False):

        assert self.size == len(taus)
        assert self.size == len(kappas)

        primal = initial_primal_guess
        dual = initial_dual_guess

        s = [2.0*dual_var for dual_var in dual]
        new_s = []

        dual_history = [ dual ]
        primal_history = [ primal ]

        if debug_output:
            print "Solving distributed problem"
            print "Gamma", gamma
            print "Taus", " ".join(map(str, taus))
            print "Kappas", " ".join(map(str, kappas))
            print
        
        for _ in range(num_iters):

            new_primal = []
            new_dual = []

            if debug_output:
                print "Step", _
                print "----------"
                print
            
            for i in range(self.size):

                if debug_output:
                    print "Agent", i
                    print

                    print "Primal", primal[i].shape
                    print primal[i]
                    print

                    print "Smooth Function Gradient", self.smooth_funcs[i].gradient(primal[i]).shape
                    print self.smooth_funcs[i].gradient(primal[i])
                    print

                    print "Constraint Matrix", self.constraint_matrices[i].shape
                    print self.constraint_matrices[i]
                    print

                    print "Constraint Vector", self.constraint_vectors[i].shape
                    print self.constraint_vectors[i]
                    print
                    
                    print "Dual", dual[i].shape
                    print dual[i]
                    print

                    print "s", s[i].shape
                    print s[i]
                    print

                new_primal.append( self.non_smooth_funcs[i].prox(primal[i] - taus[i] * (self.smooth_funcs[i].gradient(primal[i]) + self.constraint_matrices[i].transpose()*dual[i]), scaling=taus[i]) )
                
                p = 0.0
                for j in self.graph.adjacent_to(i):
                    p += s[j] - s[i]

                if debug_output:
                    print "p", p.shape
                    print p
                    print
                    
                # print "New dual = ", dual[i], "+", kappas[i], "* (", self.constraint_matrices[i], "+ 2 * ", new_primal[i], "-", primal[i], ") -", self.constraint_vectors, "+", gamma, "*", p

                new_dual.append( self.cone.projectPolar( dual[i] + kappas[i]*(self.constraint_matrices[i]*(2*new_primal[i] - primal[i]) - self.constraint_vectors[i] + gamma*p) ) )

                # new_s_value = 2 * new_dual[i]

                # for j in range(len(dual_history)):
                #     new_s_value += dual_history[j][i]

                new_s.append( s[i] + 2.0*new_dual[i] - dual[i] )
                
                if debug_output:
                    print "new_primal", new_primal[i].shape
                    print new_primal
                    print
                    
                    print "new_dual", new_dual[i].shape
                    print new_dual[i]
                    print
                
                    print "new_s", new_s[i].shape
                    print new_s[i]
                    print

                    print

            if debug_output:
                print
                
            primal_history.append(new_primal)
            primal = new_primal
            dual_history.append(new_dual)
            dual = new_dual
            s = new_s
            new_s = []

        return primal, dual, primal_history, dual_history
