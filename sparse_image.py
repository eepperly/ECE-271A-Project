#!/usr/bin/env python2

import matplotlib.pyplot as plt
from reader import read_array
import numpy as np
from scipy.sparse import csc_matrix
from graph import generate_graph
from basis_pursuit import basis_pursuit, basis_pursuit_exact

# Parse

import argparse

parser = argparse.ArgumentParser(description="Test distributed primal-dual algorithm on a sparse image sensing problem")
parser.add_argument("--num_agents", help="number of agents", type=int, default=4)
parser.add_argument("--num_steps", help="number of steps to run primal-dual algorithm", type=int, default=100)
parser.add_argument("--num_measurements", help="total number of measurements", type=int, default=None)
parser.add_argument("--fraction_measurements", help="fraction of measurements performed", type=float, default=0.2)
parser.add_argument("--gamma", help="a numerical parameter", type=float, default=None)
parser.add_argument("--solver_param", help="a numerical parameter", type=float, default=1.0)
parser.add_argument("--image_file", help="which file to load for image", type=str, default="smile.dat")
parser.add_argument("--graph_density", help="fraction of possible edges that will be filled", type=float, default=0.2)

args = parser.parse_args()

num_agents = args.num_agents
num_steps = args.num_steps

num_measurements = args.num_measurements
fraction_measurements = args.fraction_measurements

graph_density = args.graph_density

gamma = args.gamma
if gamma is None:
    gamma = 2*float(num_agents) / int(graph_density * num_agents * (num_agents-1)/2)

solver_param = args.solver_param

non_smooth_one_norm = True

# Set up problem

image_file = args.image_file
image = read_array( image_file )

num_values = image.shape[0] * image.shape[1]

if num_measurements is None:
    num_measurements = int( fraction_measurements * num_values )
num_measurements = max(num_measurements, image.shape[0])

array_index_to_vec_index = lambda a: a[0]*image.shape[1] + a[1]
vec_index_to_array_index = lambda i: (i%image.shape[1], i//image.shape[1])

measurements_made = 0
measurement_rows = []
measurement_cols = []
intensities = []

for i in range(image.shape[0]):

    measurement_rows += image.shape[1] * [measurements_made]
    measurement_cols += map(array_index_to_vec_index, [(i,j) for j in range(image.shape[1])])
    intensities.append( np.sum(image[i,:]) )
    measurements_made += 1

while measurements_made < num_measurements:

    print float(measurements_made) / num_measurements
    
    # Generate line
    x_intercept = image.shape[1] * np.random.rand()
    y_intercept = image.shape[0] * np.random.rand()
    top_or_bot = np.random.randint(2)
    left_or_right = np.random.randint(2)
    slope = (top_or_bot * image.shape[0] - y_intercept) / (x_intercept - left_or_right * image.shape[1])
    intercept = y_intercept if not left_or_right else y_intercept - slope * image.shape[1]

    # Get measurement
    intensity = 0

    from math import floor, ceil
    for j in range(image.shape[1]):
        for i in range(int(floor(slope*j+intercept)), int(ceil(slope*(j+1)+intercept))+1):
            if 0 <= i and i < image.shape[0]:
                measurement_rows.append( measurements_made )
                measurement_cols.append( array_index_to_vec_index((i,j)) )
                intensity += image[i][j]

    measurements_made += 1
    intensities.append( intensity )

big_matrix = csc_matrix( (np.ones((len(measurement_rows,))), (measurement_rows, measurement_cols)), shape=(len(intensities), image.shape[0]*image.shape[1]) )

print big_matrix.shape

big_vector = np.array(intensities).reshape(len(intensities), 1)

matrices = [big_matrix[:,i*(num_values//num_agents):(i+1)*(num_values//num_agents)] for i in range(num_agents-1)] + [big_matrix[:,(num_agents-1)*(num_values//num_agents):]]

graph = generate_graph( num_agents, graph_density=graph_density )

primal, dual, primal_history, dual_history = basis_pursuit( matrices, big_vector, graph, num_iters=num_steps, solver_param=solver_param, gamma=gamma, non_smooth_one_norm=True, debug_output=False )

final_image = np.vstack(primal).reshape(image.shape[0], image.shape[1])
final_image = np.maximum( final_image, np.zeros(final_image.shape) )

plt.imshow(final_image, cmap='gray')
plt.show()

from reader import write_array
write_array("dpda_image.dat", final_image)

optimal = basis_pursuit_exact(matrices, big_vector)
image = optimal.reshape(image.shape[0], image.shape[1])

plt.imshow(image, cmap='gray')
plt.show()
write_array("cvx_image.dat", image)
