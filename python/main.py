import time
import random

import glob
import numpy as np
import pickle

import graph
import milp
import heuristic
import disp
import utils

#
# Functions
#

def use_alg(module, alg, problem_data):
    
    # Unpack variables
    verts, dist, C = problem_data
    
    # Start time
    start_time = time.time()

    # Run the function
    solution_data = module.func(verts, dist, C)
    
    # Time to run algorithm
    delay = time.time() - start_time
    
    # Unpack solution values
    dummy_edges, dummy_verts, edges, fnum, lcm, minimum, N, ta, targets, tour, x, y = solution_data
    
    # Display
    disp.everything(alg, problem_data, solution_data)
    # disp.matrix(x, dummy_verts)
    # disp.table(dummy_verts, ta, y)
    # disp.timeline(alg, verts, dist, lcm, minimum, dummy_verts, ta, y, targets)
    # disp.fuel(alg, verts, dummy_verts, tour, ta, y, minimum, C, N, targets, lcm)
    # disp.dummy(alg, verts, dist, dummy_verts, edges, dummy_edges, minimum, N, fnum, targets)
    # disp.route(alg, verts, dist, edges, minimum)
    
    return solution_data
    
    # title  = func_name + ' Algorithm'
    # title += '  ({:.5f} sec)'.format(delay)
    # title += '  ({:.5f} units)'.format(length)


#%% Generate Problem Instances

graph.generate(
    num_instances = 2,
    num_targets = 4,
    C = 2.5,
    # seed = seed,
    # window = 9,
)

#%% Solve Problem Instances

for i in range(utils.num_saved()):

    problem_data = utils.load(i)

    # MILP
        
    # Solve the problem instance and store the solution variables
    solution_data = use_alg(milp, 'MILP', problem_data)
    
    # Save the solution variables
    utils.save(solution_data, i, 'MILP')


    # Heuristic
        
    # Solve the problem instance and store the solution variables
    solution_data = use_alg(heuristic, 'Heuristic', problem_data)
    
    # Save the solution variables
    utils.save(solution_data, i, 'Heuristic')
















