function optimal = static_distributed_pda(primal_initial_guess,...
    dual_initial_guess, gamma, taus, kappas, num_nodes, ...
    adjacency_matrix, Rs, rs, cone_matrix, nonsmooth_local_objectives,...
    smooth_local_objectives, num_steps)
%STATIC_DISTRIBUTED_PDA Compute the solution to a distributed,
%conically-contstrained, convex optimization problem
%   Consider a static graph G with N nodes. The goal of this program is to
%   solve a convex optimization problem of the form:
%
%       minimize ( sum i from 1 to N of F_i(xi_i) )
%       subject to ( sum i from 1 to N of R_i*xi_i - r_i ) is in K
%
%   Where F_i (Phi_i in the paper) are convex functions, R_i is a matrix,
%   r_i is a vector, and K is a proper cone. The decision variables here
%   are xi, whose components are distributed between the nodes as xi =
%   (xi_1,...,xi_N).
%
%   We assume that F_i can be decomposed as a sum of a smooth function
%   f_i and a possibly nonsmooth function rho_i.
%
%   Variables:
%       primal_initial_guess - consists of a guess for all the decision
%           variables, xi, for the primal problem
%       dual_initial_guess - consists of a guess for the dual variables
%       gamma - a numerical parameter
%       taus, kappas - a cell array of N parameters
%       num_nodes - the number of nodes, N
%       adjacency_matrix - a matrix representing the connectivity of the
%           graph: (i,j) is in the graph if adjacency_matrix(i,j) == 1
%       Rs, rs - the collection of R matrices and r vectors, as cell
%           arrays, indexed from 1 to N
%       cone_matrix - the cone is assumed to be represented as the set of
%           vectors such that cone_matrix*x >= 0
%       nonsmooth_local_objectives, smooth_local_objectives - a cell array
%           of functions, indexed i from 1 to N, that map xi_i to
%           rho_i(xi_i) and f_i(xi_i) respectively
%       num_steps: number of steps to take

cell_array_multiply = @(cell_array, num)  cellfun(@(x) x*num,cell_array,...
    'un',0);

primal = primal_initial_guess; % xi in the paper
dual = dual_initial_guess; % y in the paper
sum = cell_array_multiply(dual_initial_guess, 2);

scaled_nonsmooth_local_objectives =