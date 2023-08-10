import math
from itertools import pairwise

import numpy as np

import graph
import disp
import utils

#
# Functions
#

def solve(genes):
    """
    Solve a problem instance using certain genes and return the length of the
    resulting tour.
    """
    
    global dummy_verts, ts, te, f, N, Sp, F, Vp, lcm, targets, fnum
    
    # Arrival times and remaining fuel for each vertex
    ta = np.zeros(len(dummy_verts))
    y = np.ones(len(dummy_verts)) * C
    
    # Initial depot
    i = 0
    
    # tour starts with initial depot
    tour = [i]
    
    # Stack for each vertex used for back tracking
    stack = [list(range(1,N+1)) for i in range(N+1)]
    
    # The next fuel depot to be visited
    cfuel = N+2
    
    j = None
    
    # Loop until all targets are part of the tour
    while len(set(Sp) - set(tour)) != 0:
        
        #
        # Find a valid j or backtrack i
        #
    
        if len(stack[i]) != 0:
            
            # jj is all posible values of j
            jj = stack[i]
            
            # ta[jj] is ta for all possible values of j
            ta[jj] = np.maximum(ta[i]+f[i,jj], ts[jj])
            
            # A target cannot be reached, must backtrack
            # This is due to the triangle inequality
            if np.any(ta[jj] > te[jj]):
                stack[i] = []
                continue
            
            # y[jj] is ta for all possible values of j
            # There is two cases for when i is not a target
            if i in Sp:
                y[jj] = y[i] - (ta[jj] - ta[i])
            else:
                y[jj] = C - (ta[jj] - ta[i])
                
            # Matrix of values to compare
            z = np.array([te[jj], ts[jj], f[i,jj], f[jj,0], ta[jj], -y[jj]])
            
            # Normalize between each target
            z = (z-np.min(z,axis=0,keepdims=True))/(np.max(z,axis=0,keepdims=True)-np.min(z,axis=0,keepdims=True))
            
            # Weights to visit each target
            weights = np.sum(genes * z.T, axis=1)
            
            # New value for j
            j = stack[i][np.argmin(weights)]
            stack[i].remove(j)
            
            
        # Backtrack i
        else:
            # count += 1
            # text  = f'{count:3d}: '
            text  = ' -> '.join(map(str,tour))
            text += ' >< anything'
            print(text)
            stack[i] = list(range(1,N+1)) # Reset the stack of i
            tour.pop()
            i = tour[-1]
            
            # Remove refueling
            if dummy_verts[i]['index'] == 0 and len(tour) > 1 :
                cfuel -= 1
                tour.pop()
                i = tour[-1]
            
            continue
    
        #
        # Valid j is found
        #
    
        # count += 1
        # text  = f'{count:3d}: '
        text  = ' -> '.join(map(str,tour))
        text += f' -> {j:d}'
        print(text)
        
        # The UAV has insufficient fuel to return to the depot
        if y[j] < f[j,0]:
            
            # Return to depot
            tour.append(cfuel)
            y[cfuel] = y[i] - f[i,cfuel] 
            ta[cfuel] = ta[i] + f[i,cfuel]
            
            # Travel to target
            y[j] = C - f[cfuel,j]
            ta[j] = ta[cfuel] + f[cfuel,j]
            if ta[j] < ts[j]:
                ta[j] = ts[j]
                
            # Refueling causes the UAV to miss the end time
            if ta[j] > te[j]:
                continue
                
            cfuel += 1
            
            text  = ' -> '.join(map(str,tour))
            text += f' -> {j:d}'
            print(text)
            
        
        # Current target changes from j to i
        tour.append(j)
        i = j
        
        # Remove all elements in the tour from the curent stack
        stack[i] = list(set(stack[i])-set(tour))
        
        print(stack[j])
        
        
    # Return to the start of the tour
    j = N+1
    ta[j] = ta[i] + f[i,j]
    tour.append(j)    
    
    # DEBUG
    if __name__ == '__main__': globals().update(locals())
    
    return tour, ta, y
    
#
# Function for solving
#    

def func(verts, dist, C):

    depot   = verts[0]
    targets = verts[1:]
    
    #
    # Transform Into Dummy Graph
    #
    
    dummy_verts = []
    
    # LCM of all time windows
    lcm = math.lcm(*(t['w'] for t in targets))
    
    # Starting depot
    dummy_depot = depot.copy()
    dummy_depot['dummy index'] = len(dummy_verts)
    dummy_depot['s'] = 0
    dummy_depot['e'] = lcm
    dummy_verts.append(dummy_depot)
    
    # Dummy targets
    for target in targets:
        for i in range(lcm // target['w']):
            dummy_target = target.copy()
            dummy_target['dummy index'] = len(dummy_verts)
            dummy_target['s'] = target['w'] *  i
            dummy_target['e'] = target['w'] * (i + 1)
            dummy_verts.append(dummy_target)
            
    # Number of targets
    N = len(dummy_verts) - 1
            
    # Number of fuel depots
    fnum = N - 1
    
    # End depot
    dummy_depot = depot.copy()
    dummy_depot['dummy index'] = len(dummy_verts)
    dummy_depot['s'] = 0
    dummy_depot['e'] = lcm
    dummy_verts.append(dummy_depot)
    
    # Refuel sites
    for i in range(fnum): 
        dummy_depot = depot.copy()
        dummy_depot['dummy index'] = len(dummy_verts)
        dummy_depot['s'] = 0
        dummy_depot['e'] = lcm
        dummy_verts.append(dummy_depot)
    
    #
    # Constants
    #
    
    # Target indicies
    Sp = list(range(1, N+1))
    
    # Fuel site indicies
    F = list(range(N+2, N+2 + fnum))
    
    # Vertex indicies
    Vp = list(range(len(dummy_verts)))
    
    # Distance matrix
    f = np.zeros((len(dummy_verts),len(dummy_verts)))
    for i,u in enumerate(dummy_verts):
        for j,v in enumerate(dummy_verts):
            if i != j:
                f[i,j] = dist[u['index'],v['index']]
    
    # Start times
    ts = np.array([v['s'] for v in dummy_verts])
    
    # End times
    te = np.array([v['e'] for v in dummy_verts])
    
    globals().update(locals())
    
    #
    # Solve
    # TODO add genetics
    #
    
    all_genes = np.random.random((3,6))
    
    # for i in all_genes.size[0]:
    
    tour, ta, y = solve(all_genes[0])
    
    dummy_edges = list(pairwise(tour))
    edges = [(dummy_verts[i]['index'],dummy_verts[j]['index']) for i,j in dummy_edges]
    x = None
    minimum = ta[tour[-1]] # TODO make distance based
    
    # DEBUG
    if __name__ == '__main__': globals().update(locals())
    
    return dummy_edges, dummy_verts, edges, fnum, lcm, minimum, N, ta, targets, tour, x, y



if __name__ == '__main__': 
    problem_data = utils.load(0)
    verts, dist, C = problem_data
    solution_data = func(verts, dist, C)
    disp.everything('H', problem_data, solution_data)







